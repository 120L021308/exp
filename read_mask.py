import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

# 设置matplotlib使用不需要GUI的后端，避免显示问题
plt.switch_backend('Agg')

# 处理中文显示问题
# 尝试设置中文字体，如果可用
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except:
    # 如果无法设置中文字体，则使用英文标签
    use_chinese = False
else:
    use_chinese = True

# 掩码文件夹路径 - 根据您的实际路径修改
mask_folder = './results/adversarial_classification_Chinatown_TimesNet_UEA_ftM_sl24_ll24_pl0_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_test/masks/'


# 修改可视化代码，使用英文标签
def visualize_final_mask(aggregate_mask, max_epoch):
    plt.figure(figsize=(12, 8))
    plt.imshow(aggregate_mask, cmap='viridis', interpolation='none')
    plt.colorbar(label='Mask Value (1=Keep, 0=Mask)')
    plt.title(f'Final Aggregated Mask (Epoch {max_epoch})')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Time Steps')
    plt.savefig('final_mask.png')

    # 创建热力图以突出显示哪些时间点/特征被频繁掩盖
    mask_importance = 1 - aggregate_mask  # 转换为"重要性"

    plt.figure(figsize=(12, 8))
    plt.imshow(mask_importance, cmap='hot', interpolation='none')
    plt.colorbar(label='Masking Frequency (0=Never, 1=Always)')
    plt.title(f'Important Features Heatmap (Based on Epoch {max_epoch})')
    plt.xlabel('Feature Dimension')
    plt.ylabel('Time Steps')
    plt.savefig('importance_heatmap.png')

    # 计算每个特征和时间步的平均重要性
    feature_importance = np.mean(mask_importance, axis=0)
    time_importance = np.mean(mask_importance, axis=1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')

    plt.subplot(1, 2, 2)
    plt.bar(range(len(time_importance)), time_importance)
    plt.title('Time Step Importance')
    plt.xlabel('Time Step Index')
    plt.ylabel('Importance Score')

    plt.tight_layout()
    plt.savefig('importance_bars.png')

    return feature_importance, time_importance


# 找出最后一个轮次的掩码文件
def get_final_epoch_masks(mask_folder):
    # 提取所有掩码文件名中的epoch信息
    epochs = []
    for file in os.listdir(mask_folder):
        if file.startswith('mask_e') and file.endswith('.npz'):
            try:
                epoch = int(file.split('_e')[1].split('_')[0])
                epochs.append(epoch)
            except:
                continue

    if not epochs:
        print("无法找到掩码文件或提取epoch信息")
        return None, None

    max_epoch = max(epochs)
    print(f"最终轮次: {max_epoch}")

    # 获取最终轮次的所有掩码文件
    final_files = []
    for file in os.listdir(mask_folder):
        if file.startswith(f'mask_e{max_epoch}_') and file.endswith('.npz'):
            final_files.append(os.path.join(mask_folder, file))

    return final_files, max_epoch


# 合并掩码
def create_aggregate_mask(mask_files):
    all_masks = []
    for file in mask_files:
        data = np.load(file)
        if 'mask' in data:
            mask = data['mask']
            # 处理不同形状的掩码
            if len(mask.shape) == 3:  # [batch, seq_len, feature]
                for batch_mask in mask:
                    all_masks.append(batch_mask)
            else:  # [seq_len, feature]
                all_masks.append(mask)

    if not all_masks:
        print("没有找到有效的掩码数据")
        return None

    # 确保所有掩码具有相同的形状
    shapes = [m.shape for m in all_masks]
    if len(set(shapes)) > 1:
        print(f"警告: 发现不同形状的掩码: {set(shapes)}")
        # 尝试找出最常见的形状
        from collections import Counter
        common_shape = Counter(shapes).most_common(1)[0][0]
        print(f"使用最常见的形状: {common_shape}")
        all_masks = [m for m in all_masks if m.shape == common_shape]

    aggregate_mask = np.mean(all_masks, axis=0)
    return aggregate_mask


# 主处理流程
final_files, max_epoch = get_final_epoch_masks(mask_folder)

if final_files:
    print(f"找到 {len(final_files)} 个最终轮次的掩码文件")
    aggregate_mask = create_aggregate_mask(final_files)

    if aggregate_mask is not None:
        feature_importance, time_importance = visualize_final_mask(aggregate_mask, max_epoch)

        # 输出最重要的特征和时间步
        top_features = np.argsort(feature_importance)[::-1][:5]
        top_times = np.argsort(time_importance)[::-1][:5]

        print(f"\n最重要的5个特征 (索引): {top_features}")
        print(f"最重要的5个时间步 (索引): {top_times}")

        # 保存最终掩码到单个文件
        np.savez('final_aggregate_mask.npz',
                 mask=aggregate_mask,
                 feature_importance=feature_importance,
                 time_importance=time_importance,
                 top_features=top_features,
                 top_times=top_times)

        print("\n已生成以下文件:")
        print("- final_mask.png - 最终聚合掩码可视化")
        print("- importance_heatmap.png - 特征重要性热力图")
        print("- importance_bars.png - 特征和时间步重要性条形图")
        print("- final_aggregate_mask.npz - 掩码和重要性数据")
    else:
        print("无法创建聚合掩码")
else:
    print("无法找到最终轮次的掩码文件")