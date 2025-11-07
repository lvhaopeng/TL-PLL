import torch
from torch import nn
import torch.nn.functional as F
import math
from torchsummary import summary

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, strides=1): # 移除 norm_type 参数
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels) #  直接使用 BatchNorm2d
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels) #  直接使用 BatchNorm2d
        if use_1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLU(y + x)
        return y

class CrossModalAttention(nn.Module): # Corrected CrossModalAttention Class - Handles 2D and 3D query_features
    def __init__(self, query_dim_spatial, query_dim_channel, key_value_dim, attention_dim=128, num_heads=8): # Modified __init__
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.query_dim_channel = query_dim_channel # Added query_dim_channel - Input channel dimension of query features
        self.query_dim_spatial = query_dim_spatial # Added query_dim_spatial - Spatial dimension of query features (H*W)

        self.query_fc = nn.Linear(query_dim_channel, attention_dim) # query_fc now takes channel dim as input
        self.key_fc = nn.Linear(key_value_dim, attention_dim)
        self.value_fc = nn.Linear(key_value_dim, attention_dim)

        self.out_fc = nn.Linear(attention_dim, query_dim_channel) # out_fc outputs channel dimension

    def forward(self, query_features, key_value_features): # query_features: (B, C or B, C, S), key_value_features: (B, D_tab)
        batch_size = query_features.size(0)

        if len(query_features.size()) == 3: # 3D query features (b3, b4 layers)
            num_channels, spatial_dim = query_features.size(1), query_features.size(2)
            Q = self.query_fc(query_features.transpose(1, 2)).view(batch_size, spatial_dim, self.num_heads, self.attention_dim // self.num_heads) # (B, S, H, d_k)
        elif len(query_features.size()) == 2: # 2D query features (b5 layer)
            num_channels = query_features.size(1) # C
            spatial_dim = 1 # Spatial dim = 1 for flattened feature
            Q = self.query_fc(query_features).view(batch_size, spatial_dim, self.num_heads, self.attention_dim // self.num_heads) # (B, 1, H, d_k) - Reshape for b5, spatial_dim=1
        else:
            raise ValueError("Unsupported query_features dimension")


        K = self.key_fc(key_value_features).view(batch_size, 1, self.num_heads, self.attention_dim // self.num_heads) # (B, 1, H, d_k) - Tabular as single sequence
        V = self.value_fc(key_value_features).view(batch_size, 1, self.num_heads, self.attention_dim // self.num_heads) # (B, 1, H, d_v)

        Q = Q.transpose(1, 2) # (B, H, S, d_k)
        K = K.transpose(1, 2) # (B, H, 1, d_k)
        V = V.transpose(1, 2) # (B, H, 1, d_v)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attention_dim // self.num_heads) # (B, H, S, 1) - Spatial attention scores
        attention_weights = F.softmax(attention_scores, dim=-2) # (B, H, S, 1) - Softmax over spatial locations (S)

        # Element-wise multiplication and summation for context vector
        V_expanded = V.expand(-1, -1, spatial_dim, -1) # Expand V to (B, H, S, d_v)
        context_vector = (attention_weights * V_expanded).sum(dim=2) # (B, H, d_v) - Sum over spatial dimension

        context_vector = context_vector.transpose(1, 2).contiguous().view(batch_size, 1, self.attention_dim) # (B, 1, attention_dim) - Reshape for output
        fused_features = self.out_fc(context_vector).squeeze(1) # (B, query_dim_channel) - Project to channel dimension

        return fused_features, attention_weights

class TabularFeatureExtractor(nn.Module):  # 表格特征提取器代码
    def __init__(self, tabular_input_dim, hidden_dims=[256, 256, 256], output_dim=256):
        super(TabularFeatureExtractor, self).__init__()
        layers = []
        input_dim = tabular_input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))  # 修改：使用 LayerNorm 替代 BatchNorm1d
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, tabular_input):
        tabular_features = self.mlp(tabular_input)
        return tabular_features


class ResNet18(nn.Module):
    def __init__(self, Residual, tabular_input_dim):
        super(ResNet18, self).__init__()
        self.conv1_layer = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.b1 = nn.Sequential(
            self.conv1_layer,
            nn.ReLU(),
            self.bn1,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(Residual(64, 64, use_1conv=False, strides=1),
                                Residual(64, 64, use_1conv=False , strides=1))

        self.b3 = nn.Sequential(Residual(64, 128, use_1conv=True, strides=2),
                                Residual(128, 128, use_1conv=False, strides=1))

        self.b4 = nn.Sequential(Residual(128, 256, use_1conv=True, strides=2),
                                Residual(256, 256, use_1conv=False, strides=1))

        self.b5 = nn.Sequential(Residual(256, 512, use_1conv=True, strides=2),
                                Residual(512, 512, use_1conv=False, strides=1))

        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten())

        self.tabular_embed = TabularFeatureExtractor(tabular_input_dim=29, hidden_dims=[128, 256], output_dim=256) # 根据实际 tabular 输入维度调整 tabular_input_dim=30

        # 新增 Block3 中期融合注意力层， 注意力维度 attention_dim 设置为 128
        self.mid_fusion_attention_b3 = CrossModalAttention(query_dim_spatial= 35 * 26, query_dim_channel=128, key_value_dim=256, attention_dim=128)

        self.attention_layer_b4 = CrossModalAttention(query_dim_spatial= 18 * 13, query_dim_channel=256, key_value_dim=256, attention_dim=128)
        self.attention_layer_b5 = CrossModalAttention(query_dim_spatial= 1 * 1, query_dim_channel=512, key_value_dim=256, attention_dim=128)

        self.fusion_layer = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(256, 1)

    def forward(self, image_input, tabular_input, raw_v_delta, L_max, alpha, v_delta_opt, label_min, label_max): # 新增物理参数和标签归一化参数
        # --- 原来的模型前向传播部分保持不变 ---
        x = self.b1(image_input)
        x = self.b2(x)
        x_b3 = self.b3(x) # Block3 输出特征图

        x_tabular_embed = self.tabular_embed(tabular_input) # 获取表格数据 embedding

        # 中期融合： Block3 特征图和表格特征通过注意力机制融合
        # 注意力机制计算中仍然使用标准化后的表格特征
        x_b3_fused_attention, attention_weights_b3 = self.mid_fusion_attention_b3(x_b3.flatten(2),
                                                                                  x_tabular_embed)

        # 将注意力输出升维到 (B, C, 1, 1) 以便进行广播加法
        x_b3_fused_attention_reshape = x_b3_fused_attention.unsqueeze(-1).unsqueeze(-1)

        # 使用逐元素相加进行融合
        x_b3_fused = x_b3 + x_b3_fused_attention_reshape

        x_b4 = self.b4(x_b3_fused) # 使用融合后的特征作为 Block4 的输入
        x_b4_fused_attention, attention_weights_b4 = self.attention_layer_b4(x_b4.flatten(2), x_tabular_embed)
        x_b4_fused = x_b4 + x_b4_fused_attention.unsqueeze(-1).unsqueeze(-1)

        x_b5 = self.b5(x_b4_fused)
        x_b5_fused_attention, attention_weights_b5 = self.attention_layer_b5(x_b5.flatten(2), x_tabular_embed)
        x_b5_fused = x_b5 + x_b5_fused_attention.unsqueeze(-1).unsqueeze(-1)

        x = self.b6(x_b5_fused)

        if len(x.shape) != 2:
            x = x.view(x.size(0), -1)

        # 后期融合： 拼接最终图像特征和表格数据特征
        fused_features = torch.cat([x,
                                    x_tabular_embed], dim=1)

        fused_features = self.fusion_layer(fused_features) # 融合层代码

        # 模型预测输出 (归一化后的)
        L_model_normalized = self.output_layer(fused_features) # shape: (B, 1)

        # --- 新增：计算物理估计寿命 L_phys ---
        # 经验公式： L_pred = L_max * exp(-alpha * (V_delta - V_delta_opt)^2)
        # 注意：这里使用原始的 raw_v_delta 进行计算
        # 需要确保 L_max, alpha, v_delta_opt 是 tensor 且与 raw_v_delta 在同一设备
        L_max_t = torch.tensor(L_max, dtype=torch.float32).to(raw_v_delta.device)
        alpha_t = torch.tensor(alpha, dtype=torch.float32).to(raw_v_delta.device)
        v_delta_opt_t = torch.tensor(v_delta_opt, dtype=torch.float32).to(raw_v_delta.device)
        label_min_t = torch.tensor(label_min, dtype=torch.float32).to(raw_v_delta.device)
        label_max_t = torch.tensor(label_max, dtype=torch.float32).to(raw_v_delta.device)


        # 计算物理估计值 (原始尺度)
        L_phys_raw = L_max_t * torch.exp(-alpha_t * (raw_v_delta - v_delta_opt_t).pow(2)) # raw_v_delta shape: (B,)

        # 将 L_phys_raw 从 (B,) 调整为 (B, 1) 以匹配 L_model_normalized 的形状
        L_phys_raw = L_phys_raw.unsqueeze(-1) # shape: (B, 1)

        # 将物理估计值归一化到与模型输出相同的尺度
        # 确保除数不为零
        range_label = label_max_t - label_min_t
        L_phys_normalized = (L_phys_raw - label_min_t) / (range_label if range_label != 0 else 1.0) # shape: (B, 1)

        # --- 修改返回值，增加 L_phys_normalized ---
        # 返回模型预测 (归一化), 物理估计 (归一化), 以及注意力权重
        return L_model_normalized, L_phys_normalized, attention_weights_b3, attention_weights_b4, attention_weights_b5

def build_resnet(tabular_input_dim=32): # 移除 norm_type 参数 - Default tabular_input_dim is now 32
    return ResNet18(Residual, tabular_input_dim=tabular_input_dim)

