# analyze_masks.py

import numpy as np
import matplotlib.pyplot as plt
import os
import glob  # 用于查找文件
from collections import defaultdict
import pandas as pd  # 用于更方便的数据处理和统计


# --- 中文字体设置 ---
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 或者微软雅黑
# plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei'] # 或者文泉驿正黑 (Linux下常见)
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac/Windows下有时可用
# 请确保您的系统上安装了这些字体中的某一个，或者替换为您已安装的中文字体名称

plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
# --- 中文字体设置结束 ---


# --- 配置参数 ---
# 根据你的实验设置名称修改 (这个需要和你运行run.py时生成的setting字符串完全一致，不包括末尾的_train/_test或迭代号_ii)
# 例如: 'adversarial_classification_Chinatown_AdvTest_TimesNet_UEA_ftM_sl96_ll0_pl0_dm16_nh_el2_dl_df32_expand_dc_fc_ebtimeF_dtTrue_Exp_0'
# 最好的方法是查看运行时的控制台输出，找到那个长长的 setting 字符串。
SETTING_NAME = 'adversarial_classification_Chinatown_AdvTest_TimesNet_UEA_ftM_sl24_ll24_pl0_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0'  # 【请务必修改为您的实际SETTING_NAME】

# 选择要分析的掩码类型: '_train' (对抗性掩码) 或 '_test' (随机掩码)
# 对于研究“最致命缺失模式”，通常选择 '_train'
ANALYSIS_TYPE = '_train'  # 【重要配置：'_train' 或 '_test'】

# 是否为每个样本的每个特征都生成并保存单独的可视化图表（如果样本/特征很多，会生成大量图片）
SAVE_INDIVIDUAL_SAMPLE_PLOTS = True  # 【配置：True 或 False】
MAX_SAMPLES_TO_PLOT_INDIVIDUALLY = 5  # 如果 SAVE_INDIVIDUAL_SAMPLE_PLOTS 为 True，最多绘制多少个样本的图
MAX_FEATURES_TO_PLOT_INDIVIDUALLY = 3  # 每个样本最多绘制多少个特征的图

RESULTS_BASE_DIR = './results'
# 完整的setting字符串，例如 'adversarial_classification_..._Exp_0_train' (假设迭代号ii为0)
# 您需要根据实际情况调整，特别是如果 itr > 1，setting字符串末尾的迭代号会变化。
# 假设迭代号为0 (itr=1时，ii为0)
FULL_SETTING_STRING = SETTING_NAME  # 通常 setting 字符串的末尾会有一个迭代号，例如 "_0"
MASK_SUBFOLDER = FULL_SETTING_STRING + ANALYSIS_TYPE
MASKS_DIR = os.path.join(RESULTS_BASE_DIR, MASK_SUBFOLDER, 'masks')
ANALYSIS_OUTPUT_DIR = os.path.join(MASKS_DIR, 'analysis_plots')  # 保存图表的目录

# --- 创建输出目录 ---
if not os.path.exists(ANALYSIS_OUTPUT_DIR):
    os.makedirs(ANALYSIS_OUTPUT_DIR)
    print(f"创建分析图表输出目录: {ANALYSIS_OUTPUT_DIR}")


# --- 工具函数 ---
def load_mask_data(file_path):
    """加载单个.npz文件中的掩码和原始数据"""
    try:
        data = np.load(file_path, allow_pickle=True)
        mask_batch = data['mask']  # 形状: [batch_size, seq_len, feature_dim]

        original_data_dict = data.get('original_data')
        if original_data_dict is None:
            original_batch_x = data.get('original_batch_x')
            original_labels_batch = data.get('original_labels')
        else:
            # .item() 用于从0维数组中取出字典, 确保它是一个实际的字典对象
            actual_dict = original_data_dict.item() if original_data_dict.ndim == 0 else original_data_dict
            original_batch_x = actual_dict.get('batch_x')
            original_labels_batch = actual_dict.get('label')  # 注意键名可能是 'label'

        zero_ratio_batch = data.get('zero_ratio')  # 这可能是整个batch的平均缺失率
        batch_idx_file = data.get('batch_idx')
        epoch_file = data.get('epoch')
        method_file = data.get('method')  # 如果是_test的掩码，可能有填充方法

        # 确保数据是 NumPy 数组
        if not isinstance(mask_batch, np.ndarray): mask_batch = np.array(mask_batch)
        if original_batch_x is not None and not isinstance(original_batch_x, np.ndarray):
            original_batch_x = np.array(original_batch_x)
        if original_labels_batch is not None and not isinstance(original_labels_batch, np.ndarray):
            original_labels_batch = np.array(original_labels_batch)

        # 检查并修正 original_batch_x 和 mask_batch 的 batch_size 是否一致
        if original_batch_x is not None and mask_batch.shape[0] != original_batch_x.shape[0]:
            print(
                f"警告: 文件 {os.path.basename(file_path)} 中掩码和原始数据的批次大小不匹配。掩码: {mask_batch.shape[0]}, 数据: {original_batch_x.shape[0]}. 跳过此文件。")
            return None
        if original_labels_batch is not None and mask_batch.shape[0] != original_labels_batch.shape[0]:
            print(
                f"警告: 文件 {os.path.basename(file_path)} 中掩码和标签的批次大小不匹配。掩码: {mask_batch.shape[0]}, 标签: {original_labels_batch.shape[0]}. 可能影响按类别分析。")
            # 可以选择不跳过，但标签可能不完全对应

        return {
            'mask_batch': mask_batch,
            'original_batch_x': original_batch_x,
            'original_labels_batch': original_labels_batch,
            'zero_ratio_batch_avg': zero_ratio_batch,  # 文件中存储的可能是批次平均
            'batch_idx_file': batch_idx_file,
            'epoch_file': epoch_file,
            'method_file': method_file,
            'file_path': file_path
        }
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_single_feature_with_mask(original_series, mask_series, title="Time Series with Mask", save_path=None):
    """绘制单个特征的原始时间序列和其上的掩码，并可选择保存"""
    plt.figure(figsize=(18, 6))
    time_steps = np.arange(len(original_series))

    plt.plot(time_steps, original_series, label='Original Data', color='dodgerblue', linewidth=1.5, zorder=1)

    missing_indices = np.where(mask_series == 0)[0]
    if len(missing_indices) > 0:
        plt.scatter(time_steps[missing_indices], original_series[missing_indices],
                    label='Masked (Missing) Points', color='red', marker='x', s=80, zorder=2)

    plt.title(title, fontsize=16)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Feature Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存到: {save_path}")
        plt.close()  # 保存后关闭图像，避免过多窗口
    else:
        plt.show()


# --- 1. 加载所有掩码文件信息 ---
print(f"在以下路径中查找 .npz 文件: {MASKS_DIR}")
# 实验代码保存掩码文件名格式为 'final_mask_b{batch_idx}.npz'
# 或者如果来自 test 方法且包含 method 信息，文件名可能不同，但通常也有 'mask'
# 我们使用 'final_mask_b*.npz' 来匹配训练时保存的对抗性掩码
# 如果是分析 _test 目录，可能需要更通用的匹配模式，如 '*.npz'，并后续过滤
if ANALYSIS_TYPE == '_train':
    npz_files = glob.glob(os.path.join(MASKS_DIR, 'final_mask_b*.npz'))
else:  # _test 目录下的掩码文件名可能包含填充方法，或者也是 final_mask_b*
    npz_files = glob.glob(os.path.join(MASKS_DIR, '*.npz'))
    # 过滤掉汇总文件
    npz_files = [f for f in npz_files if 'summary' not in os.path.basename(f).lower()]

if not npz_files:
    print(f"在 {MASKS_DIR} 中没有找到 .npz 文件。请检查 SETTING_NAME, ANALYSIS_TYPE 和实验是否已正确生成掩码文件。")
    exit()
else:
    print(f"找到 {len(npz_files)} 个 .npz 文件.")

all_loaded_batch_data = []
for f_path in npz_files:
    data_item = load_mask_data(f_path)
    if data_item and data_item['original_batch_x'] is not None:  # 确保有原始数据才能进行很多分析
        all_loaded_batch_data.append(data_item)

if not all_loaded_batch_data:
    print("未能加载任何有效的掩码数据（可能缺少原始数据或文件损坏）。")
    exit()
else:
    print(f"成功从 {len(all_loaded_batch_data)} 个文件中加载了包含原始数据的掩码信息。")

# --- 提取所有样本级别的数据 ---
all_samples_mask = []
all_samples_original_x = []
all_samples_labels = []
# 假设 seq_len 和 feature_dim 对于所有批次都是一致的
# 从第一个加载的批次获取维度信息
first_batch_mask = all_loaded_batch_data[0]['mask_batch']
_, seq_len, num_features = first_batch_mask.shape

for batch_data in all_loaded_batch_data:
    mask_batch = batch_data['mask_batch']
    original_x_batch = batch_data['original_batch_x']
    labels_batch = batch_data['original_labels_batch']

    for i in range(mask_batch.shape[0]):  # 遍历批次中的每个样本
        all_samples_mask.append(mask_batch[i])
        all_samples_original_x.append(original_x_batch[i])
        if labels_batch is not None and i < len(labels_batch):
            all_samples_labels.append(labels_batch[i])
        else:
            all_samples_labels.append(None)  # 如果没有标签或不匹配

all_samples_mask = np.array(all_samples_mask)  # Shape: [total_samples, seq_len, num_features]
all_samples_original_x = np.array(all_samples_original_x)  # Shape: [total_samples, seq_len, num_features]
all_samples_labels = np.array(all_samples_labels)  # Shape: [total_samples] or [total_samples, ...]
# 【新增】对加载的掩码进行二值化处理，确保是严格的0和1
print(f"原始 all_samples_mask 的唯一值: {np.unique(all_samples_mask[:5])}") # 查看前几个样本的掩码原始值


print(f"原始 all_samples_mask 的形状: {all_samples_mask.shape}")
print(f"原始 all_samples_mask 的最小值: {np.min(all_samples_mask)}")
print(f"原始 all_samples_mask 的最大值: {np.max(all_samples_mask)}")
print(f"原始 all_samples_mask 的均值: {np.mean(all_samples_mask)}")
# 查看前几个样本掩码的一些值
if all_samples_mask.size > 0:
    print(f"原始 all_samples_mask 的前几个值示例: {all_samples_mask.flat[:10]}")
# 绘制一个值的直方图，了解其分布
import matplotlib.pyplot as plt
plt.hist(all_samples_mask.flatten(), bins=50)
plt.title("原始掩码值分布 (二值化前)")
plt.xlabel("掩码值 (保留概率)")
plt.ylabel("频率")
plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "debug_raw_mask_distribution.png"))
plt.close()
print(f"原始掩码值分布图已保存到: {os.path.join(ANALYSIS_OUTPUT_DIR, 'debug_raw_mask_distribution.png')}")
all_samples_mask = (all_samples_mask > 0.5).astype(np.int8) # 大于0.5的为1（保留），否则为0（缺失）
print(f"二值化后 all_samples_mask 的唯一值: {np.unique(all_samples_mask)}")

total_num_samples = all_samples_mask.shape[0]
print(f"总共提取了 {total_num_samples} 个样本进行分析。")
print(f"数据维度: 序列长度={seq_len}, 特征数={num_features}")

# --- 2. 可视化分析：单个样本的掩码和原始数据 (按配置选择是否执行和数量) ---
if SAVE_INDIVIDUAL_SAMPLE_PLOTS and total_num_samples > 0:
    print(f"\n--- 为前 {min(total_num_samples, MAX_SAMPLES_TO_PLOT_INDIVIDUALLY)} 个样本生成单独的特征可视化图表 ---")
    samples_plotted = 0
    for sample_idx in range(min(total_num_samples, MAX_SAMPLES_TO_PLOT_INDIVIDUALLY)):
        sample_mask = all_samples_mask[sample_idx]
        sample_original_x = all_samples_original_x[sample_idx]
        sample_label = all_samples_labels[sample_idx] if all_samples_labels[sample_idx] is not None else "N/A"

        # 从文件名中获取一些信息，例如原始的batch_idx (如果需要)
        # 这里我们简单地使用全局样本索引

        print(f"  绘制样本 {sample_idx} (标签: {sample_label})...")
        for feature_idx in range(min(num_features, MAX_FEATURES_TO_PLOT_INDIVIDUALLY)):
            original_series = sample_original_x[:, feature_idx]
            mask_series = sample_mask[:, feature_idx]

            plot_title = (f"Sample {sample_idx}, Feature {feature_idx} (Label: {sample_label})\n"
                          f"Mask Type: {ANALYSIS_TYPE.strip('_')}")
            save_filename = f"sample_{sample_idx}_feature_{feature_idx}_label_{str(sample_label).replace('/', '_')}.png"
            save_path = os.path.join(ANALYSIS_OUTPUT_DIR, save_filename)
            plot_single_feature_with_mask(original_series, mask_series, title=plot_title, save_path=save_path)
        samples_plotted += 1
    print(f"为 {samples_plotted} 个样本的选定特征生成了单独图表。")

# --- 3. 全局聚合统计分析 ---
if total_num_samples > 0:
    print("\n--- 全局聚合统计分析 ---")

    # 3.1 每个时间步的平均掩码频率
    print("  计算每个时间步的平均掩码频率...")
    # all_samples_mask 的形状是 [total_samples, seq_len, num_features]
    # 我们关心的是在某个时间步t，所有样本和所有特征中，被掩码的比例
    # mask中0表示缺失，所以计算 (1-mask)
    masked_counts_per_timestep = np.sum(1 - all_samples_mask, axis=(0, 2))  # Sum over samples and features
    total_points_per_timestep = total_num_samples * num_features
    masking_freq_per_timestep = masked_counts_per_timestep / total_points_per_timestep

    plt.figure(figsize=(15, 7))
    plt.plot(np.arange(seq_len), masking_freq_per_timestep, marker='o', linestyle='-')
    plt.title(f'Average Masking Frequency per Time Step (Overall)\nMask Type: {ANALYSIS_TYPE.strip("_")}', fontsize=16)
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Average Masking Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path_ts_freq = os.path.join(ANALYSIS_OUTPUT_DIR, f'agg_mask_freq_vs_timestep_{ANALYSIS_TYPE.strip("_")}.png')
    plt.savefig(save_path_ts_freq)
    print(f"图表已保存到: {save_path_ts_freq}")
    plt.close()

    # 3.2 每个特征的平均掩码频率 (与之前脚本类似，但基于所有样本)
    print("  计算每个特征的平均掩码频率...")
    masked_counts_per_feature = np.sum(1 - all_samples_mask, axis=(0, 1))  # Sum over samples and time steps
    total_points_per_feature = total_num_samples * seq_len
    masking_freq_per_feature = masked_counts_per_feature / total_points_per_feature

    plt.figure(figsize=(max(10, num_features * 0.5), 6))  # 动态调整宽度
    plt.bar(np.arange(num_features), masking_freq_per_feature, color='skyblue')
    plt.title(f'Average Masking Frequency per Feature (Overall)\nMask Type: {ANALYSIS_TYPE.strip("_")}', fontsize=16)
    plt.xlabel('Feature Index', fontsize=12)
    plt.ylabel('Average Masking Frequency', fontsize=12)
    plt.xticks(np.arange(num_features))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_path_feat_freq = os.path.join(ANALYSIS_OUTPUT_DIR, f'agg_mask_freq_vs_feature_{ANALYSIS_TYPE.strip("_")}.png')
    plt.savefig(save_path_feat_freq)
    print(f"图表已保存到: {save_path_feat_freq}")
    plt.close()

    # 3.3 被掩码位置 vs. 未被掩码位置的原始数据特性 (所有样本聚合)
    print("  计算被掩码位置 vs. 未被掩码位置的原始数据特性...")
    # all_samples_mask: [N, S, F], 0 for masked
    # all_samples_original_x: [N, S, F]

    all_masked_points_original_values = all_samples_original_x[all_samples_mask == 0]
    all_unmasked_points_original_values = all_samples_original_x[all_samples_mask == 1]

    # 对于原始数据值分布图
    if len(all_masked_points_original_values) > 0 and len(all_unmasked_points_original_values) > 0:
        print("调试: 进入原始值分布图的if条件块。")  # 新增
        plt.figure(figsize=(12, 6))
        if len(all_masked_points_original_values) > 0:  # 再次检查，虽然外层if已保证
            print("调试: 绘制被掩码点原始值直方图...")  # 新增
            plt.hist(all_masked_points_original_values, bins=50, alpha=0.7, label='Original Values at Masked Points',
                     color='red', density=True)
        if len(all_unmasked_points_original_values) > 0:  # 再次检查
            print("调试: 绘制未被掩码点原始值直方图...")  # 新增
            plt.hist(all_unmasked_points_original_values, bins=50, alpha=0.7,
                     label='Original Values at Unmasked Points', color='dodgerblue', density=True)

        print("调试: 设置图表标题和标签...")  # 新增
        plt.title(f'Distribution of Original Values (All Samples, All Features)\nMask Type: {ANALYSIS_TYPE.strip("_")}',
                  fontsize=16)
        plt.xlabel('Original Value', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        save_path_val_dist = os.path.join(ANALYSIS_OUTPUT_DIR,
                                          f'agg_original_value_distribution_{ANALYSIS_TYPE.strip("_")}.png')
        print(f"调试: 准备保存图表到: {save_path_val_dist}")  # 新增
        try:
            plt.savefig(save_path_val_dist)
            print(f"图表已保存到: {save_path_val_dist}")  # 这是您期望看到的
        except Exception as e_savefig:
            print(f"调试: 保存图表 {save_path_val_dist} 时发生错误: {e_savefig}")  # 新增错误捕获

        plt.close()
        print("调试: 原始值分布图处理完毕。")  # 新增

        print("\n  Overall Statistics for Original Values:")
        # ... (打印统计信息) ...


        print("\n  Overall Statistics for Original Values:")
        print(
            f"    Masked Points: Count={len(all_masked_points_original_values)}, Mean={np.mean(all_masked_points_original_values):.4f}, Median={np.median(all_masked_points_original_values):.4f}, Std={np.std(all_masked_points_original_values):.4f}")
        print(
            f"  Unmasked Points: Count={len(all_unmasked_points_original_values)}, Mean={np.mean(all_unmasked_points_original_values):.4f}, Median={np.median(all_unmasked_points_original_values):.4f}, Std={np.std(all_unmasked_points_original_values):.4f}")
    else:
        print("调试: 未满足绘制原始值分布图的条件 (被掩码点或未被掩码点列表为空)。")  # 新增
        print(f"  len(all_masked_points_original_values) = {len(all_masked_points_original_values)}")
        print(f"  len(all_unmasked_points_original_values) = {len(all_unmasked_points_original_values)}")


    # 3.4 被掩码位置 vs. 未被掩码位置的原始数据变化量特性 (所有样本聚合)
    print("  计算被掩码位置 vs. 未被掩码位置的原始数据变化量特性...")
    # 计算每个样本每个特征的一阶差分
    all_samples_original_x_diff = np.diff(all_samples_original_x, axis=1,
                                          prepend=all_samples_original_x[:, [0], :])  # 在时间轴(axis=1)上差分

    # 提取对应位置的差分值 (注意掩码也需要对应调整，因为差分后序列长度不变，但掩码是针对原始长度的)
    # all_samples_mask 仍然是 [N, S, F]
    all_masked_points_original_diff_abs = np.abs(all_samples_original_x_diff[all_samples_mask == 0])
    all_unmasked_points_original_diff_abs = np.abs(all_samples_original_x_diff[all_samples_mask == 1])

    if len(all_masked_points_original_diff_abs) > 0 and len(all_unmasked_points_original_diff_abs) > 0:
        plt.figure(figsize=(12, 6))
        plt.hist(all_masked_points_original_diff_abs, bins=50, alpha=0.7, label='Abs. Change at Masked Points',
                 color='red', density=True)
        plt.hist(all_unmasked_points_original_diff_abs, bins=50, alpha=0.7, label='Abs. Change at Unmasked Points',
                 color='dodgerblue', density=True)
        plt.title(
            f'Distribution of Absolute Value Changes (All Samples, All Features)\nMask Type: {ANALYSIS_TYPE.strip("_")}',
            fontsize=16)
        plt.xlabel('Absolute Change in Value (1st Diff)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        # 对于变化量，x轴可能需要对数尺度或限制范围以获得更好的可视化效果
        # plt.xscale('log') # 如果变化量差异很大，可以尝试
        plt.tight_layout()
        save_path_diff_dist = os.path.join(ANALYSIS_OUTPUT_DIR,
                                           f'agg_original_diff_abs_distribution_{ANALYSIS_TYPE.strip("_")}.png')
        plt.savefig(save_path_diff_dist)
        print(f"图表已保存到: {save_path_diff_dist}")
        plt.close()

        print("\n  Overall Statistics for Absolute Value Changes (1st Diff):")
        print(
            f"    Masked Points: Count={len(all_masked_points_original_diff_abs)}, Mean Abs Diff={np.mean(all_masked_points_original_diff_abs):.4f}")
        print(
            f"  Unmasked Points: Count={len(all_unmasked_points_original_diff_abs)}, Mean Abs Diff={np.mean(all_unmasked_points_original_diff_abs):.4f}")

    # 3.5 (可选) 按类别进行聚合分析的框架
    # 需要确保 all_samples_labels 是有效的并且可以用于分组
    unique_labels = np.unique(all_samples_labels[all_samples_labels != np.array(None)])  # 获取有效标签
    if len(unique_labels) > 1 and len(unique_labels) < total_num_samples / 2:  # 确保有多个类别且不是每个样本一个类别
        print(f"\n--- 按类别进行聚合统计分析 (发现 {len(unique_labels)} 个类别) ---")
        # 示例：按类别统计各特征的掩码频率
        # 您可以扩展此部分以进行更复杂的按类别分析

        # 确保 all_samples_labels 是一维的，如果不是，需要适当处理
        # 例如，如果标签是 [val,], 取 val
        processed_labels = []
        valid_indices_for_labels = []
        for idx, lbl in enumerate(all_samples_labels):
            if lbl is not None:
                # 假设标签是单个值或者可以转换成单个hashable值
                try:
                    processed_labels.append(
                        lbl.item() if hasattr(lbl, 'item') else lbl[0] if isinstance(lbl, (list, np.ndarray)) and len(
                            lbl) > 0 else lbl)
                    valid_indices_for_labels.append(idx)
                except:  # pylint: disable=bare-except
                    processed_labels.append(str(lbl))  # fallback to string
                    valid_indices_for_labels.append(idx)

        if valid_indices_for_labels:
            df_data = {
                'label': np.array(processed_labels),
                # 添加每个特征的掩码情况 (例如，每个样本每个特征的掩码点数)
            }
            # 为了简化，我们这里只演示按类别统计特征掩码频率的思路
            # masks_for_labeled_samples = all_samples_mask[valid_indices_for_labels]
            # original_x_for_labeled_samples = all_samples_original_x[valid_indices_for_labels]

            # # 此处可以循环每个类别，提取该类别下的掩码和原始数据，然后进行类似3.1-3.4的分析
            # for class_label in unique_labels:
            #     class_indices = [i for i, lbl_val in enumerate(processed_labels) if lbl_val == class_label]
            #     if not class_indices:
            #         continue

            #     class_masks = masks_for_labeled_samples[class_indices] # [num_class_samples, seq_len, num_features]
            #     # class_original_x = original_x_for_labeled_samples[class_indices]

            #     print(f"\n  分析类别: {class_label} (共 {len(class_masks)} 个样本)")

            #     # 对 class_masks 进行类似上述的频率分析、值分布分析等，并保存图表
            #     # 例如：计算该类别下每个特征的掩码频率
            #     masked_counts_per_feature_class = np.sum(1 - class_masks, axis=(0, 1))
            #     total_points_per_feature_class = class_masks.shape[0] * seq_len
            #     masking_freq_per_feature_class = masked_counts_per_feature_class / total_points_per_feature_class

            #     plt.figure(figsize=(max(10, num_features * 0.5), 6))
            #     plt.bar(np.arange(num_features), masking_freq_per_feature_class, color='coral')
            #     plt.title(f'Avg Masking Freq per Feature (Class: {class_label})\nMask Type: {ANALYSIS_TYPE.strip("_")}', fontsize=16)
            #     plt.xlabel('Feature Index', fontsize=12)
            #     plt.ylabel('Average Masking Frequency', fontsize=12)
            #     plt.xticks(np.arange(num_features))
            #     plt.grid(axis='y', linestyle='--', alpha=0.7)
            #     plt.tight_layout()
            #     save_path_class_feat_freq = os.path.join(ANALYSIS_OUTPUT_DIR, f'class_{str(class_label).replace("/","_")}_mask_freq_vs_feature_{ANALYSIS_TYPE.strip("_")}.png')
            #     plt.savefig(save_path_class_feat_freq)
            #     print(f"图表已保存到: {save_path_class_feat_freq}")
            #     plt.close()
            print("  按类别分析的框架已设置。您可以根据需要填充更详细的分析逻辑。")
        else:
            print("  没有找到有效的标签用于按类别分析。")


else:
    print("没有加载到有效数据，跳过聚合分析。")

print(f"\n--- 分析完成 ---")
print(f"所有生成的聚合图表已保存到目录: {ANALYSIS_OUTPUT_DIR}")
if SAVE_INDIVIDUAL_SAMPLE_PLOTS and total_num_samples > 0:
    print(f"单独样本的图表也保存在该目录下（如果启用了该选项）。")