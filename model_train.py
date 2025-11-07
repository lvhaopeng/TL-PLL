import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from PIL import Image
import torch.nn as nn
from torch import nn, optim
from torch.optim import AdamW
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import seaborn as sns  # 引入 seaborn 库，用于绘制更美观的统计图

# 引入模型和数据集类
from model import build_resnet as build_resnet_bn # 从 model_BatchNorm_simplified.py 导入 BatchNorm 模型构建函数，并命名为 build_resnet_bn
#from model_NoNorm import build_resnet as build_resnet_nonorm # 从 model_NoNorm.py 导入 NoNorm 模型构建函数，并命名为 build_resnet_nonorm
from dataset import MultiModalDataset

# 定义训练集复杂图片增强函数
import torchvision.transforms.functional as F # 引入 functional 模块

def train_transform_complex(image, tabular_data, label, raw_v_delta, label_min, label_max): # <-- 在这里新增 raw_v_delta 参数
    """
    应用于训练集图片的复杂增强操作：
    ... (文档字符串不变) ...
    Args:
        image (PIL.Image): 原始图片
        tabular_data (torch.Tensor): 已经标准化好的表格数据
        label (float): 原始标签
        raw_v_delta (torch.Tensor): 原始 V_delta 张量 (单个值) # <-- 更新文档字符串
        label_min (float): 标签最小值 (用于归一化)
        label_max (float): 标签最大值 (用于归一化)
    Returns:
        list: 包含 6 个 (图片张量, 表格数据张量, 标签张量, 原始 V_delta 张量) 元组的列表 # <-- 更新文档字符串
    """
    augmented_samples = []
    # 标签归一化 (这部分保持不变)
    label_normalized = (label - label_min) / (label_max - label_min)
    label_tensor = torch.tensor([label_normalized], dtype=torch.float32)

    # 转换为灰度图 (这部分保持不变)
    image = image.convert("L")

    # 定义后续通用的图片处理流程 (ToTensor 和 Normalize) (这部分保持不变)
    post_crop_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 1. 随机裁剪 5 个区域 (这部分保持不变)
    for _ in range(5):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(336, 336))
        cropped_image = F.crop(image, i, j, h, w)

        # 2. 对裁剪后的区域进行 180° 旋转操作 (这部分保持不变)
        img_rotated = F.rotate(cropped_image, 180)

        # 3. 对裁剪后的区域进行镜像操作 (新增水平镜像)
        img_mirrored = F.hflip(cropped_image)  # *** 新增：水平镜像操作 ***

        # 4. 转换为 Tensor 并归一化，并组合样本
        # 添加原始裁剪图片
        # 修改：将 raw_v_delta 添加到元组中 (这部分在之前的修改中已经包含，再次确认)
        augmented_samples.append((post_crop_transform(cropped_image), tabular_data, label_tensor, raw_v_delta))
        # 添加 180 度旋转后的裁剪图片
        augmented_samples.append((post_crop_transform(img_rotated), tabular_data, label_tensor, raw_v_delta))
        # 修改：将 raw_v_delta 添加到元组中 (这部分在之前的修改中已经包含，再次确认)
        augmented_samples.append((post_crop_transform(img_rotated), tabular_data, label_tensor, raw_v_delta))

    return augmented_samples # 返回包含 15 个样本元组的列表 (这部分保持不变)
def val_test_transform_simple(image, tabular_data, label, raw_v_delta, label_min, label_max): # <-- 在这里新增 raw_v_delta 参数
    """
    应用于验证集和测试集图片的简单预处理操作：
    ... (文档字符串不变) ...
    Args:
        image (PIL.Image): 原始图片
        tabular_data (torch.Tensor): 已经标准化好的表格数据
        label (float): 原始标签
        raw_v_delta (torch.Tensor): 原始 V_delta 张量 (单个值) # <-- 更新文档字符串
        label_min (float): 标签最小值 (用于归一化)
        label_max (float): 标签最大值 (用于归一化)
    Returns:
        list: 包含 1 个 (图片张量, 表格数据张量, 标签张量, 原始 V_delta 张量) 元组的列表 # <-- 更新文档字符串
    """
    # 标签归一化 (这部分保持不变)
    label_normalized = (label - label_min) / (label_max - label_min)
    label_tensor = torch.tensor([label_normalized], dtype=torch.float32)

    # 转换为灰度图并 Resize (这部分保持不变)
    image = image.convert("L")
    image = transforms.Resize((336, 336))(image)

    # 转换为 Tensor 并归一化 (这部分保持不变)
    image_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])(image)

    # 修改返回值，增加 raw_v_delta (这部分在之前的修改中已经包含，再次确认)
    return [(image_tensor, tabular_data, label_tensor, raw_v_delta)] # 返回包含单个样本的列表

def custom_collate_fn(batch):
    """
    处理 from __getitem__ 返回列表的情况，先展平 batch
    """
    # batch 是一个列表，每个元素是 __getitem__ 返回的列表 (包含 1 或 6 个样本元组)
    # 每个样本元组现在是 (图片张量, 标准化表格张量, 标签张量, 原始 V_delta 张量)
    # 先将所有子列表中的样本元组展平到一个列表中
    flattened_batch = [item for sublist in batch for item in sublist]

    # 然后按照原始 collate_fn 的逻辑处理展平后的样本列表
    # 解包时需要解包 4 个元素
    images, tabular_data, labels, raw_v_deltas = zip(*flattened_batch)
    images = torch.stack(images)
    tabular_data = torch.stack(tabular_data)
    labels = torch.stack(labels)
    raw_v_deltas = torch.stack(raw_v_deltas) # 堆叠原始 V_delta 张量

    # 修改返回值，增加 raw_v_deltas
    return images, tabular_data, labels, raw_v_deltas

# 修改训练集复杂图片增强函数，接收 raw_v_delta 参数并传递
def train_val_test_data_process(image_dir, csv_file, v_delta_col_name='V_delta', batch_size=4, random_seed=42): # 新增 v_delta_col_name 参数
    # 1. 读取 CSV 数据到 DataFrame
    df = pd.read_csv(csv_file)

    # 获取 V_delta 列的索引
    if v_delta_col_name in df.columns:
        v_delta_column_index = df.columns.get_loc(v_delta_col_name)
        print(f"Found V_delta column at index: {v_delta_column_index}")
    else:
        v_delta_column_index = None
        print(f"Warning: V_delta column '{v_delta_col_name}' not found in CSV. Physics regularization will not include V_delta.")


    # 计算整体的标签 min/max 值 (用于后续反归一化)
    label_min = df.iloc[:, -1].min()
    label_max = df.iloc[:, -1].max()

    # 2. 在划分数据集 *之前*， 对 DataFrame 进行 *完全随机打乱* (Shuffle)
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # 3. 划分训练集、验证集和测试集的 DataFrame 切片
    train_size = int(0.7 * len(df_shuffled))
    val_size = int(0.15 * len(df_shuffled))
    test_size = len(df_shuffled) - train_size - val_size

    train_df = df_shuffled.iloc[:train_size].reset_index(drop=True)
    val_df = df_shuffled.iloc[train_size:train_size + val_size].reset_index(drop=True)
    test_df = df_shuffled.iloc[train_size + val_size:].reset_index(drop=True)

    # 新增代码：获取并打印训练集标签的最大值
    train_label_max_value = train_df.iloc[:, -1].max()
    print(f"训练集标签最大值: {train_label_max_value:.4f}")  # 打印出来查看

    # 4. 创建 *独立的* MultiModalDataset 实例，并传入对应的 transform、归一化参数和 V_delta 列索引
    # 在训练集 Dataset 创建时， StandardScaler 会进行 fit_transform 并计算 mean/scale
    train_dataset = MultiModalDataset(image_dir=image_dir, dataframe_slice=train_df,
                                      v_delta_column_index=v_delta_column_index, # 传入 V_delta 列索引
                                      transform=train_transform_complex,
                                      label_min=label_min, label_max=label_max)

    # 获取训练集计算出的 scaler (用于验证集和测试集的 transform)
    scaler = train_dataset.scaler

    val_dataset = MultiModalDataset(image_dir=image_dir, dataframe_slice=val_df,
                                    v_delta_column_index=v_delta_column_index, # 传入 V_delta 列索引
                                    transform=val_test_transform_simple, scaler=scaler,
                                    label_min=label_min, label_max=label_max)

    test_dataset = MultiModalDataset(image_dir=image_dir, dataframe_slice=test_df,
                                     v_delta_column_index=v_delta_column_index, # 传入 V_delta 列索引
                                     transform=val_test_transform_simple, scaler=scaler,
                                     label_min=label_min, label_max=label_max)


    # 5. 创建 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 返回 DataLoaders, 标签的 min/max 值, 和 V_delta 列索引
    return train_dataloader, val_dataloader, test_dataloader, label_min, label_max, v_delta_column_index # 修改返回值

# 修改 train_model_process 函数，接受 label_min 和 label_max 参数
def train_model_process(model, train_dataloader, val_dataloader, test_dataloader,
                        label_min, label_max, v_delta_column_index, # 新增 V_delta 列索引
                        L_max, alpha, v_delta_opt, lambda_reg, # 新增物理公式参数和正则化权重
                        num_epoch, avg_true):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    # 定义用于任务损失和物理损失的准则 (可以使用相同的，通常是 MSE 或 SmoothL1)
    criterion_task = nn.SmoothL1Loss()
    criterion_phys = nn.SmoothL1Loss() # 或者 nn.MSELoss()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf') # 这里的 best_loss 是总损失

    train_loss_all = [] # 总损失
    train_task_loss_all = [] # 任务损失
    train_phys_loss_all = [] # 物理损失
    val_loss_all = [] # 总损失
    val_task_loss_all = [] # 任务损失
    val_phys_loss_all = [] # 物理损失

    train_mae_all = []
    val_mae_all = []
    train_r2_all = []
    val_r2_all = []
    train_outputs_all_epochs = []
    train_labels_all_epochs = []
    val_outputs_all_epochs = []
    val_labels_all_epochs = []

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True)


    for epoch in range(num_epoch):
        print(f'Epoch {epoch + 1}/{num_epoch}')
        print('-' * 10)
        print(f"真实值平均值: {avg_true:.4f}")

        # 训练阶段
        model.train()
        epoch_train_total_loss = 0.0
        epoch_train_task_loss = 0.0
        epoch_train_phys_loss = 0.0
        epoch_train_mae = 0.0
        train_samples = 0
        all_train_outputs = []
        all_train_labels = []

        # 修改：DataLoader 返回 4 个值
        for images, tabular_data, labels, raw_v_deltas in train_dataloader:
            # 注意：这里的 images, tabular_data, labels 已经是 batch*增强数 的大小
            images = images.to(device).float()
            tabular_data = tabular_data.to(device).float()
            labels = labels.to(device).float().squeeze(-1)
            raw_v_deltas = raw_v_deltas.to(device).float() # 将原始 V_delta 移到设备上

            optimizer.zero_grad()
            # 修改：调用模型时传入物理公式参数和标签归一化参数
            L_model_normalized, L_phys_normalized, _, _, _ = model(
                images, tabular_data, raw_v_deltas,
                L_max=L_max, alpha=alpha, v_delta_opt=v_delta_opt,
                label_min=label_min, label_max=label_max
            ) # L_model_normalized shape: (B, 1), L_phys_normalized shape: (B, 1)

            # 确保 labels 和 L_model_normalized/L_phys_normalized 的形状匹配 (都是 (B, 1) 或 (B,))
            # L_model_normalized 和 L_phys_normalized 都是 (B, 1)， labels 是 (B,)
            # 将 labels 调整为 (B, 1) 以匹配损失函数输入
            labels_normalized = labels.unsqueeze(-1) # shape: (B, 1)


            # 计算任务损失 (模型预测 vs 真实标签)
            task_loss = criterion_task(L_model_normalized, labels_normalized)

            # 计算物理损失 (模型预测 vs 物理估计)
            phys_loss = criterion_phys(L_model_normalized, L_phys_normalized)

            # 计算总损失
            total_loss = task_loss + lambda_reg * phys_loss # 加入物理正则项

            total_loss.backward()
            optimizer.step()

            # 累加损失 (注意：这里累加的是每个 batch 的总损失，用于计算平均损失)
            epoch_train_total_loss += total_loss.item() * labels.size(0) # 乘以原始样本数 (或增强后的总样本数)
            epoch_train_task_loss += task_loss.item() * labels.size(0)
            epoch_train_phys_loss += phys_loss.item() * labels.size(0)


            # 计算原始尺度的 MAE 和 R² (需要反归一化模型输出和真实标签)
            # 注意：这里使用模型输出 L_model_normalized 进行反归一化计算 MAE/R^2
            outputs_original = L_model_normalized.squeeze(-1) * (label_max - label_min) + label_min # shape: (B,)
            labels_original = labels * (label_max - label_min) + label_min  # shape: (B,)

            epoch_train_mae += mean_absolute_error(labels_original.cpu().numpy(), outputs_original.cpu().detach().numpy()) * labels.size(0) # 乘以当前 batch 的样本数累加
            all_train_outputs.append(outputs_original.cpu().detach().numpy())
            all_train_labels.append(labels_original.cpu().numpy())
            train_samples += labels.size(0) # 累加当前 batch 的样本数


        # 计算 epoch 平均损失和指标
        train_epoch_outputs = np.concatenate(all_train_outputs, axis=0)
        train_epoch_labels = np.concatenate(all_train_labels, axis=0)
        train_r2 = r2_score(train_epoch_labels, train_epoch_outputs)
        # 注意：这里的损失是基于增强后的总样本数计算的平均值
        train_total_loss = epoch_train_total_loss / train_samples if train_samples > 0 else 0
        train_task_loss = epoch_train_task_loss / train_samples if train_samples > 0 else 0
        train_phys_loss = epoch_train_phys_loss / train_samples if train_samples > 0 else 0
        train_mae = epoch_train_mae / train_samples if train_samples > 0 else 0

        train_loss_all.append(train_total_loss)
        train_task_loss_all.append(train_task_loss)
        train_phys_loss_all.append(train_phys_loss)
        train_mae_all.append(train_mae)
        train_r2_all.append(train_r2)
        train_outputs_all_epochs.append(train_epoch_outputs)
        train_labels_all_epochs.append(train_epoch_labels)

        # 验证阶段
        model.eval()
        epoch_val_total_loss = 0.0
        epoch_val_task_loss = 0.0
        epoch_val_phys_loss = 0.0
        epoch_val_mae = 0.0
        val_samples = 0
        all_val_outputs = []
        all_val_labels = []

        with torch.no_grad():
            # 修改：DataLoader 返回 4 个值
            for images, tabular_data, labels, raw_v_deltas in val_dataloader:
                images = images.to(device).float()
                tabular_data = tabular_data.to(device).float()
                labels = labels.to(device).float().squeeze(-1)
                raw_v_deltas = raw_v_deltas.to(device).float()

                # 修改：调用模型时传入物理公式参数和标签归一化参数
                L_model_normalized, L_phys_normalized, _, _, _ = model(
                     images, tabular_data, raw_v_deltas,
                     L_max=L_max, alpha=alpha, v_delta_opt=v_delta_opt,
                     label_min=label_min, label_max=label_max
                ) # L_model_normalized shape: (B, 1), L_phys_normalized shape: (B, 1)

                labels_normalized = labels.unsqueeze(-1) # shape: (B, 1)

                # 计算任务损失 (模型预测 vs 真实标签)
                task_loss = criterion_task(L_model_normalized, labels_normalized)

                # 计算物理损失 (模型预测 vs 物理估计)
                phys_loss = criterion_phys(L_model_normalized, L_phys_normalized)

                # 计算总损失
                total_loss = task_loss + lambda_reg * phys_loss


                epoch_val_total_loss += total_loss.item() * labels.size(0)
                epoch_val_task_loss += task_loss.item() * labels.size(0)
                epoch_val_phys_loss += phys_loss.item() * labels.size(0)


                # 计算原始尺度的 MAE 和 R² (需要反归一化模型输出和真实标签)
                outputs_original = L_model_normalized.squeeze(-1) * (label_max - label_min) + label_min
                labels_original = labels * (label_max - label_min) + label_min

                epoch_val_mae += mean_absolute_error(labels_original.cpu().numpy(), outputs_original.cpu().detach().numpy()) * labels.size(0)
                all_val_outputs.append(outputs_original.cpu().detach().numpy())
                all_val_labels.append(labels_original.cpu().numpy())
                val_samples += labels.size(0)

        # 计算 epoch 平均损失和指标
        val_epoch_outputs = np.concatenate(all_val_outputs, axis=0)
        val_epoch_labels = np.concatenate(all_val_labels, axis=0)
        val_r2 = r2_score(val_epoch_labels, val_epoch_outputs)
        val_total_loss = epoch_val_total_loss / val_samples if val_samples > 0 else 0
        val_task_loss = epoch_val_task_loss / val_samples if val_samples > 0 else 0
        val_phys_loss = epoch_val_phys_loss / val_samples if val_samples > 0 else 0

        if val_samples > 0:
            val_mae = epoch_val_mae / val_samples
        else:
            val_mae = np.nan

        val_mae = np.nan_to_num(val_mae, nan=0.0)

        val_loss_all.append(val_total_loss)
        val_task_loss_all.append(val_task_loss)
        val_phys_loss_all.append(val_phys_loss)
        val_mae_all.append(val_mae)
        val_r2_all.append(val_r2)
        val_outputs_all_epochs.append(val_epoch_outputs)
        val_labels_all_epochs.append(val_epoch_labels)

        train_relative_error = (train_mae / avg_true) * 40
        val_relative_error = (val_mae / avg_true) * 40

        # 打印更详细的损失信息
        print(f'Train Total Loss: {train_total_loss:.4f}, Task Loss: {train_task_loss:.4f}, Phys Loss: {train_phys_loss:.4f}, MAE: {train_mae:.4f} ({train_relative_error:.2f}%), R²: {train_r2:.4f}')
        print(f'Val Total Loss: {val_total_loss:.4f}, Task Loss: {val_task_loss:.4f}, Phys Loss: {val_phys_loss:.4f}, MAE: {val_mae:.4f} ({val_relative_error:.2f}%), R²: {val_r2:.4f}')


        # Scheduler 根据总验证损失调整学习率
        scheduler.step(val_total_loss)

        # 保存最佳模型权重基于总验证损失
        if val_total_loss < best_loss:
            best_loss = val_total_loss
            best_model_wts = copy.deepcopy(model.state_dict())


    # 加载最佳模型权重 (基于总验证损失)
    model.load_state_dict(best_model_wts)

    # 测试阶段 (和原来逻辑类似，只需要确保使用传入的 label_min/max)
    model.eval()
    epoch_test_total_loss = 0.0  # 同样记录测试集总损失 (尽管不用于训练或早停)
    test_samples = 0
    all_test_outputs = []
    all_test_labels = []

    with torch.no_grad():
        # 修改：DataLoader 返回 4 个值
        for images, tabular_data, labels, raw_v_deltas in test_dataloader:
            images = images.to(device).float()
            tabular_data = tabular_data.to(device).float()
            labels = labels.to(device).float().squeeze(-1)
            raw_v_deltas = raw_v_deltas.to(device).float()


            # 修改：调用模型时传入物理公式参数和标签归一化参数
            L_model_normalized, L_phys_normalized, _, _, _ = model(
                 images, tabular_data, raw_v_deltas,
                 L_max=L_max, alpha=alpha, v_delta_opt=v_delta_opt,
                 label_min=label_min, label_max=label_max
            )

            labels_normalized = labels.unsqueeze(-1) # shape: (B, 1)

            # 计算测试集总损失 (任务损失 + 物理损失)
            task_loss = criterion_task(L_model_normalized, labels_normalized)
            phys_loss = criterion_phys(L_model_normalized, L_phys_normalized)
            total_loss = task_loss + lambda_reg * phys_loss
            epoch_test_total_loss += total_loss.item() * labels.size(0)


            # 计算原始尺度的 MAE 和 R² (使用模型预测 L_model_normalized)
            outputs_original = L_model_normalized.squeeze(-1) * (label_max - label_min) + label_min
            labels_original = labels * (label_max - label_min) + label_min

            all_test_outputs.append(outputs_original.cpu().detach().numpy())
            all_test_labels.append(labels_original.cpu().numpy())
            test_samples += labels.size(0)


    # 计算测试集平均损失和指标
    test_outputs = np.concatenate(all_test_outputs, axis=0)
    test_labels = np.concatenate(all_test_labels, axis=0)
    test_r2 = r2_score(test_labels, test_outputs)
    test_mae = mean_absolute_error(test_labels, test_outputs)
    test_total_loss = epoch_test_total_loss / test_samples if test_samples > 0 else 0


    # 修改返回值，增加训练和验证集的任务损失和物理损失列表
    return {
        'train_loss': train_loss_all, # 总损失
        'train_task_loss': train_task_loss_all, # 任务损失
        'train_phys_loss': train_phys_loss_all, # 物理损失
        'val_loss': val_loss_all, # 总损失
        'val_task_loss': val_task_loss_all, # 任务损失
        'val_phys_loss': val_phys_loss_all, # 物理损失
        'train_mae': train_mae_all,
        'val_mae': val_mae_all,
        'train_r2': train_r2_all,
        'val_r2': val_r2_all,
        'test_loss': test_total_loss, # 测试集总损失
        'test_mae': test_mae,
        'test_r2': test_r2,
        'train_outputs_all_epochs': train_outputs_all_epochs,
        'train_labels_all_epochs': train_labels_all_epochs,
        'val_outputs_all_epochs': val_outputs_all_epochs,
        'val_labels_all_epochs': val_labels_all_epochs,
        'test_outputs': test_outputs,
        'test_labels': test_labels
    }

def matplo_loss(results, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(results['train_loss'], label=f'{model_name} Train Loss')
    plt.plot(results['val_loss'], label=f'{model_name} Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    # 注意：保存和显示的调用在 main 函数中

def plot_r2_scatter(true_labels, predictions, title, r2, filename):
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=true_labels, y=predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{title}\nR² = {r2:.4f}')
    # Add a diagonal line for perfect prediction
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.grid(True)
# 修改 main 函数，调整划分比例，获取 label_min/max，并修改散点图文件名
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = r'D:\桌面文件\GH4169resnet5\GHdata'
    csv_file = r'D:\桌面文件\GH4169resnet5\data.csv'
    batch_size = 12
    num_epoch = 20

    # --- 新增：定义物理公式参数和正则化权重 ---
    # 你需要根据你的具体数据和物理知识来确定这些参数的值
    # 这里的数值只是示例，请替换为你实际的值
    L_max_physics = 19 # 示例值：最大可能寿命
    alpha_physics = 0.001 # 示例值：衰减系数
    v_delta_opt_physics = 0.05 # 示例值：最优 V_delta 值
    lambda_regularization = 0.1# 示例值：物理正则项的权重，这是一个重要的超参数，需要调优

    # 确定 V_delta 对应的列名 (请根据你的 CSV 文件实际列名修改)
    v_delta_column_name = 'δcontent(after)' # <-- !!请修改为你 CSV 文件中 V_delta 对应的列名!!

    # 获取数据加载器和标签的 min/max 值, 以及 V_delta 列索引
    train_loader, val_loader, test_loader, label_min, label_max, v_delta_col_index = train_val_test_data_process(
        image_dir, csv_file, v_delta_col_name=v_delta_column_name, batch_size=batch_size)

    # 计算真实值平均值 (使用 label_min 和 label_max 恢复原始尺度范围)
    df_full = pd.read_csv(csv_file)
    avg_true = np.mean(df_full.iloc[:, -1].values.astype('float32'))
    print(f"真实值平均值: {avg_true:.4f}")

    # 1. 创建 BatchNorm 模型
    model_bn = build_resnet_bn(tabular_input_dim=30).to(device) # 确保 tabular_input_dim 正确
    models = {'BatchNorm_PhysicsReg': model_bn} # 修改模型名称以区分


    model_results = {}

    # 2. 循环训练和评估不同的模型
    for model_name, model in models.items():
        print(f"\n--- Training and Evaluating {model_name} Model ---")

        # 训练模型，传入物理公式参数和正则化权重
        train_process = train_model_process(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            label_min=label_min, # 传入 label_min
            label_max=label_max, # 传入 label_max
            v_delta_column_index=v_delta_col_index, # 传入 V_delta 列索引
            L_max=L_max_physics, # 传入物理参数
            alpha=alpha_physics,
            v_delta_opt=v_delta_opt_physics,
            lambda_reg=lambda_regularization, # 传入正则化权重
            num_epoch=num_epoch,
            avg_true=avg_true
        )
        model_results[model_name] = train_process

        # 3. 绘制 Loss 曲线图 (现在有总损失、任务损失、物理损失)
        plt.figure(figsize=(12, 8))
        plt.plot(train_process['train_loss'], label=f'{model_name} Train Total Loss')
        plt.plot(train_process['val_loss'], label=f'{model_name} Val Total Loss')
        plt.plot(train_process['train_task_loss'], label=f'{model_name} Train Task Loss', linestyle='--')
        plt.plot(train_process['val_task_loss'], label=f'{model_name} Val Task Loss', linestyle='--')
        plt.plot(train_process['train_phys_loss'], label=f'{model_name} Train Physics Loss', linestyle=':')
        plt.plot(train_process['val_phys_loss'], label=f'{model_name} Val Physics Loss', linestyle=':')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Training and Validation Losses (Total, Task, Physics)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'losses_curve_{model_name}.png') # 修改文件名
        plt.show()

        # 4. 打印测试集性能 (输出信息包含模型名称)
        test_loss = train_process['test_loss'] # 这里的 test_loss 是总损失
        test_mae = train_process['test_mae']
        test_r2 = train_process['test_r2']
        print(f"{model_name} Test Total Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}")

        # 5. 找到验证集损失最小的 epoch (基于总验证损失)
        best_val_loss_epoch = np.argmin(train_process['val_loss']) # 基于总损失寻找最佳 epoch
        best_val_total_loss = train_process['val_loss'][best_val_loss_epoch]
        best_val_task_loss = train_process['val_task_loss'][best_val_loss_epoch]
        best_val_phys_loss = train_process['val_phys_loss'][best_val_loss_epoch]

        print(f"{model_name} Best Epoch (based on Val Total Loss): {best_val_loss_epoch + 1}")
        print(f"  Best Val Total Loss: {best_val_total_loss:.4f}")
        print(f"  Corresponding Val Task Loss: {best_val_task_loss:.4f}")
        print(f"  Corresponding Val Physics Loss: {best_val_phys_loss:.4f}")


    # 6.  绘制 R² 散点图 (为所有模型输出 R² 散点图， 文件名包含模型名称)
    for model_name, train_process in model_results.items():
        print(f"\n--- Plotting R² Scatter Plots for {model_name} Model ---")
        # 注意：散点图是基于模型预测 L_model 和真实标签 L_true 绘制的，
        # 物理估计 L_phys 仅在训练中用于正则化，不直接体现在这个散点图上。
        best_val_epoch_idx = np.argmin(train_process['val_loss']) # 仍然基于总损失选择 epoch

        # 1. 最佳验证 Total Loss Epoch 的 训练集 R² 散点图
        best_train_outputs = train_process['train_outputs_all_epochs'][best_val_epoch_idx]
        best_train_labels = train_process['train_labels_all_epochs'][best_val_epoch_idx]
        best_train_r2 = r2_score(best_train_labels, best_train_outputs)
        plot_r2_scatter(best_train_labels, best_train_outputs,
                        f"{model_name} Train Dataset (Epoch {best_val_epoch_idx + 1})",
                        best_train_r2,
                        f"{model_name}_train_r2_scatter_best_val_epoch.png")
        plt.show()

        # 2. 最佳验证 Total Loss Epoch 的 验证集 R² 散点图
        best_val_outputs = train_process['val_outputs_all_epochs'][best_val_epoch_idx]
        best_val_labels = train_process['val_labels_all_epochs'][best_val_epoch_idx]
        best_val_r2 = r2_score(best_val_labels, best_val_outputs)
        plot_r2_scatter(best_val_labels, best_val_outputs,
                        f"{model_name} Validation Dataset (Epoch {best_val_epoch_idx + 1})",
                        best_val_r2,
                        f"{model_name}_val_r2_scatter_best_val_epoch.png")
        plt.show()

        # 3. 最后 Epoch 的 测试集 R² 散点图 (测试结果是基于最佳验证模型)
        test_outputs = train_process['test_outputs']
        test_labels = train_process['test_labels']
        test_r2 = train_process['test_r2']
        plot_r2_scatter(test_labels, test_outputs,
                        f"{model_name} Test Dataset",
                        test_r2,
                        f"{model_name}_test_r2_scatter_test_set.png")
        plt.show()


if __name__ == '__main__':
    main()