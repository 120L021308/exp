import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import traceback
import matplotlib.pyplot as plt

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
        plt.rcParams['axes.unicode_minus'] = False
    except:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 回退方案
        plt.rcParams['axes.unicode_minus'] = False

def visualize_mask(mask, feature_names=None, class_label=None, save_path=None):
    """可视化单个掩码矩阵"""
    plt.figure(figsize=(12, 8))

    # 检查掩码的维度，并进行必要的转换
    original_shape = mask.shape
    print(f"原始掩码形状: {original_shape}")

    # 处理三维掩码 (batch_size, seq_len, features)
    if len(mask.shape) == 3:
        # 如果第三维是1，可以直接压缩掉这个维度
        if mask.shape[2] == 1:
            mask = mask.squeeze(2)  # 从 (16, 24, 1) 变成 (16, 24)
            print(f"压缩掩码形状为: {mask.shape}")
        else:
            # 如果特征维度大于1，将其展平为 (batch_size, seq_len*features)
            batch_size, seq_len, n_features = mask.shape
            mask = mask.reshape(batch_size, seq_len * n_features)
            print(f"展平后的掩码形状: {mask.shape}")

            # 更新特征名称
            if feature_names is None:
                feature_names = []
                for t in range(seq_len):
                    for f in range(n_features):
                        feature_names.append(f"T{t + 1}_F{f + 1}")

    # 创建蓝白二分色图
    colors = ["white", "blue"]
    cmap = LinearSegmentedColormap.from_list("blue_white", colors)

    # 绘制掩码热图 (1为保留值，0为缺失值)
    ax = sns.heatmap(1 - mask, cmap=cmap, cbar_kws={'label': '缺失状态 (1=缺失)'})

    # 设置坐标轴标签和刻度
    if len(original_shape) == 3 and original_shape[2] > 1:
        # 为展平的特征设置适当的标签
        plt.xlabel('时间步×特征')
        # 在合适的位置添加主要时间步分隔线
        for t in range(1, original_shape[1]):
            plt.axvline(x=t * original_shape[2], color='black', linestyle='--', linewidth=0.5)
    else:
        plt.xlabel('时间步' if mask.shape[1] > mask.shape[0] else '特征')

    plt.ylabel('样本' if mask.shape[0] > 1 else '特征')

    # 如果样本数太多，限制显示的y轴刻度
    if mask.shape[0] > 20:
        plt.yticks(np.arange(0, mask.shape[0], 5))

    # 添加标题
    title = "数据掩码矩阵"
    if class_label is not None:
        title += f" (类别: {class_label})"
    plt.title(title)

    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


# 统计类别级缺失模式
def analyze_masks_by_class(masks_dir, output_dir=None):
    """分析不同类别的缺失模式，处理掩码形状不一致的情况"""
    if output_dir is None:
        output_dir = os.path.join(masks_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    # 收集所有掩码
    all_masks = []
    class_labels = []
    mask_shapes = []  # 用于记录所有掩码形状

    for filename in os.listdir(masks_dir):
        if filename.startswith('final_mask_') and filename.endswith('.npz'):
            try:
                data = np.load(os.path.join(masks_dir, filename), allow_pickle=True)
                mask = data['mask']
                print(f"文件: {filename}, 掩码形状: {mask.shape}")
                mask_shapes.append(mask.shape)

                # 从原始数据获取类别标签
                if 'original_data' in data:
                    original_data = data['original_data'].item()
                    if 'label' in original_data:
                        try:
                            label = original_data['label'].squeeze()[0]  # 获取第一个样本的标签
                            class_labels.append(label)
                            all_masks.append(mask)
                        except Exception as e:
                            print(f"处理标签时出错: {e}")
                            # 尝试不同的数据结构
                            if isinstance(original_data['label'], (int, float)):
                                label = original_data['label']
                                class_labels.append(label)
                                all_masks.append(mask)
                    else:
                        print(f"警告: {filename} 不包含标签信息，使用默认标签0")
                        class_labels.append(0)
                        all_masks.append(mask)
                else:
                    print(f"警告: {filename} 不包含原始数据，使用默认标签0")
                    class_labels.append(0)
                    all_masks.append(mask)
            except Exception as e:
                print(f"加载文件 {filename} 出错: {e}")
                traceback.print_exc()

    if not all_masks:
        print("未找到有效的掩码文件!")
        return {}

    # 打印收集到的掩码数量
    print(f"收集到 {len(all_masks)} 个掩码，类别标签: {set(class_labels)}")

    # 找出最常见的掩码形状
    from collections import Counter
    shape_counter = Counter([str(shape) for shape in mask_shapes])
    most_common_shape_str = shape_counter.most_common(1)[0][0]
    print(f"最常见的掩码形状: {most_common_shape_str}")

    # 按类别分组掩码
    unique_classes = np.unique(class_labels)
    class_masks = {cls: [] for cls in unique_classes}

    for mask, label in zip(all_masks, class_labels):
        class_masks[label].append(mask)

    # 为每个类别计算平均掩码
    for cls in unique_classes:
        if class_masks[cls]:
            try:
                # 标准化掩码形状
                cls_masks = class_masks[cls]
                standardized_masks = []

                # 找出该类别最常见的掩码形状
                cls_shapes = [m.shape for m in cls_masks]
                cls_shape_counter = Counter([str(shape) for shape in cls_shapes])
                cls_common_shape_str = cls_shape_counter.most_common(1)[0][0]

                # 将字符串形状转换回元组
                import ast
                cls_common_shape = ast.literal_eval(cls_common_shape_str)

                print(f"类别 {cls} 的最常见掩码形状: {cls_common_shape}")

                # 只处理形状匹配的掩码
                for m in cls_masks:
                    if m.shape == cls_common_shape:
                        # 处理三维掩码
                        if len(m.shape) == 3:
                            if m.shape[2] == 1:
                                standardized_masks.append(m.squeeze(2))
                            else:
                                standardized_masks.append(m)
                        else:
                            standardized_masks.append(m)

                if not standardized_masks:
                    print(f"警告: 类别 {cls} 没有标准形状的掩码")
                    continue

                print(f"类别 {cls} 使用 {len(standardized_masks)}/{len(cls_masks)} 个形状一致的掩码")

                # 计算平均掩码
                avg_mask = np.mean(standardized_masks, axis=0)

                # 处理三维掩码的特征名称
                feature_names = None
                if len(avg_mask.shape) == 3 and avg_mask.shape[2] > 1:
                    batch_size, seq_len, n_features = avg_mask.shape
                    # 创建展平后的特征名称
                    feature_names = []
                    for t in range(seq_len):
                        for f in range(n_features):
                            feature_names.append(f"T{t + 1}_F{f + 1}")

                    # 重塑掩码以便可视化
                    avg_mask = avg_mask.reshape(batch_size, seq_len * n_features)
                elif avg_mask.shape[-1] == 1 and len(avg_mask.shape) == 3:
                    avg_mask = avg_mask.squeeze(-1)
                    feature_names = [f"特征{i + 1}" for i in range(avg_mask.shape[1])]
                else:
                    feature_names = [f"特征{i + 1}" for i in range(avg_mask.shape[1]
                                                                   if len(avg_mask.shape) > 1 else 1)]

                # 可视化
                visualize_mask(
                    avg_mask,
                    feature_names=feature_names,
                    class_label=cls,
                    save_path=os.path.join(output_dir, f'class_{cls}_avg_mask.png')
                )

            except Exception as e:
                print(f"处理类别 {cls} 的掩码时出错: {e}")
                traceback.print_exc()

    return class_masks


def analyze_missing_impact(masks_dir, results_file, output_dir=None):
    """分析缺失位置对性能的影响，处理掩码形状不一致问题"""
    if output_dir is None:
        output_dir = os.path.join(masks_dir, 'impact_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # 加载所有掩码
    all_masks = []  # 处理后的掩码
    mask_shapes = []  # 原始掩码形状
    raw_masks = []  # 原始掩码（不做处理）

    for filename in os.listdir(masks_dir):
        if filename.startswith('final_mask_') and filename.endswith('.npz'):
            try:
                data = np.load(os.path.join(masks_dir, filename), allow_pickle=True)
                mask = data['mask']
                raw_masks.append(mask)  # 保存原始掩码
                mask_shapes.append(mask.shape)

                # 处理三维掩码
                if len(mask.shape) == 3:
                    if mask.shape[2] == 1:
                        mask = mask.squeeze(2)
                    else:
                        # 处理多特征情况
                        batch_size, seq_len, n_features = mask.shape
                        mask = mask.reshape(batch_size, seq_len * n_features)

                all_masks.append(mask)
            except Exception as e:
                print(f"加载掩码文件 {filename} 时出错: {e}")

    # 计算平均掩码
    if all_masks:
        try:
            # 找出最常见的掩码形状
            from collections import Counter
            shape_counter = Counter([str(shape) for shape in mask_shapes])
            most_common_shape_str = shape_counter.most_common(1)[0][0]
            print(f"最常见的掩码形状: {most_common_shape_str}")

            # 过滤出相同原始形状的掩码的索引
            import ast
            most_common_shape = ast.literal_eval(most_common_shape_str)
            filtered_indices = []

            for i, shape in enumerate(mask_shapes):
                if shape == most_common_shape:
                    filtered_indices.append(i)

            # 选择对应的处理后掩码
            filtered_masks = [all_masks[i] for i in filtered_indices]

            print(f"使用 {len(filtered_masks)}/{len(all_masks)} 个形状一致的掩码进行分析")

            if not filtered_masks:
                print("过滤后没有形状一致的掩码，无法继续分析")
                return

            avg_mask = np.mean(filtered_masks, axis=0)

            # 分析时间维度的缺失影响
            temporal_missing = np.mean(1 - avg_mask, axis=1)  # 沿特征维度平均
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(temporal_missing)), temporal_missing)
            plt.xlabel('时间步')
            plt.ylabel('平均缺失率')
            plt.title('不同时间步的平均缺失率')
            plt.savefig(os.path.join(output_dir, 'temporal_missing_rate.png'))
            plt.close()

            # 分析特征维度的缺失影响
            feature_missing = np.mean(1 - avg_mask, axis=0)  # 沿时间维度平均
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(feature_missing)), feature_missing)
            plt.xlabel('特征')
            plt.ylabel('平均缺失率')
            plt.title('不同特征的平均缺失率')
            plt.savefig(os.path.join(output_dir, 'feature_missing_rate.png'))
            plt.close()
        except Exception as e:
            print(f"分析缺失位置影响时出错: {e}")
            traceback.print_exc()


def compare_filling_methods(results_file, output_dir):
    """比较不同填充方法的性能"""
    os.makedirs(output_dir, exist_ok=True)

    # 检查结果文件是否存在
    if not os.path.exists(results_file):
        print(f"结果文件 {results_file} 不存在，无法进行填充方法比较")

        # 创建一个空图表，表明数据不可用
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "无填充方法比较数据\n(结果文件不存在)",
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=16)
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'filling_methods_comparison_na.png'))
        plt.close()
        return

    methods = []
    accuracies = []
    drop_rates = []

    try:
        # 读取性能结果
        with open(results_file, 'r') as f:
            lines = f.readlines()

        original_acc = None
        for line in lines:
            if line.startswith('原始准确率:'):
                original_acc = float(line.split(':')[1])
            elif '填充准确率' in line:
                method = line.split('填充准确率')[0]
                acc = float(line.split(':')[1].split(',')[0])
                drop = float(line.split('下降率:')[1].strip('%\n')) / 100

                methods.append(method)
                accuracies.append(acc)
                drop_rates.append(drop)

        if not methods:
            print(f"在结果文件 {results_file} 中未找到填充方法数据")
            return

        # 绘制性能比较图
        plt.figure(figsize=(12, 6))

        x = np.arange(len(methods))
        width = 0.35

        ax1 = plt.subplot(1, 2, 1)
        ax1.bar(x, accuracies, width, label='准确率')
        if original_acc:
            ax1.axhline(y=original_acc, color='r', linestyle='-', label='原始准确率')
        ax1.set_ylabel('准确率')
        ax1.set_title('不同填充方法的准确率')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods)
        ax1.legend()

        ax2 = plt.subplot(1, 2, 2)
        ax2.bar(x, drop_rates, width, label='性能下降率')
        ax2.set_ylabel('性能下降率')
        ax2.set_title('不同填充方法的性能下降率')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'filling_methods_comparison.png'))
        plt.close()
    except Exception as e:
        print(f"比较填充方法时出错: {e}")
        traceback.print_exc()

def identify_critical_missing(masks_dir, output_dir=None):
    """识别对性能影响最大的缺失位置，处理掩码形状不一致问题"""
    if output_dir is None:
        output_dir = os.path.join(masks_dir, 'critical_analysis')
    os.makedirs(output_dir, exist_ok=True)

    # 加载所有成功生成的对抗性掩码
    all_masks = []
    zero_ratios = []
    mask_shapes = []
    raw_masks = []  # 保存原始掩码，不做任何处理

    for filename in os.listdir(masks_dir):
        if filename.startswith('final_mask_') and filename.endswith('.npz'):
            try:
                data = np.load(os.path.join(masks_dir, filename), allow_pickle=True)
                mask = data['mask']
                raw_masks.append(mask)  # 保存原始掩码
                mask_shapes.append(mask.shape)

                # 处理三维掩码
                if len(mask.shape) == 3:
                    if mask.shape[2] == 1:
                        mask = mask.squeeze(2)
                    else:
                        # 处理多特征情况
                        batch_size, seq_len, n_features = mask.shape
                        mask = mask.reshape(batch_size, seq_len * n_features)

                # 尝试获取zero_ratio
                if 'zero_ratio' in data:
                    zero_ratio = data['zero_ratio']
                else:
                    # 如果没有保存zero_ratio，自己计算
                    zero_ratio = 1.0 - np.mean(mask)

                all_masks.append(mask)
                zero_ratios.append(zero_ratio)
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                traceback.print_exc()

    if not all_masks:
        print("没有找到有效的掩码文件")
        return

    try:
        # 找出最常见的掩码形状
        from collections import Counter
        shape_counter = Counter([str(shape) for shape in mask_shapes])
        most_common_shape_str = shape_counter.most_common(1)[0][0]
        print(f"最常见的掩码形状: {most_common_shape_str}")

        # 过滤出具有相同原始形状的掩码索引
        import ast
        most_common_shape = ast.literal_eval(most_common_shape_str)
        filtered_indices = []

        for i, shape in enumerate(mask_shapes):
            if shape == most_common_shape:
                filtered_indices.append(i)

        # 选择对应的处理后掩码
        filtered_masks = [all_masks[i] for i in filtered_indices]

        print(f"使用 {len(filtered_masks)}/{len(all_masks)} 个形状一致的掩码进行分析")

        if not filtered_masks:
            print("过滤后没有形状一致的掩码，无法继续分析")
            return

        # 计算每个位置的缺失频率
        stacked_masks = np.stack(filtered_masks)
        missing_freq = 1 - np.mean(stacked_masks, axis=0)

        # 可视化高频缺失位置
        plt.figure(figsize=(15, 10))
        cmap = plt.cm.Reds
        plt.imshow(missing_freq, cmap=cmap, aspect='auto')
        plt.colorbar(label='缺失频率')
        plt.xlabel('特征')
        plt.ylabel('时间步')
        plt.title('关键缺失位置热图')
        plt.savefig(os.path.join(output_dir, 'critical_missing_heatmap.png'))
        plt.close()

        # 找出前N个最关键的位置
        top_n = min(10, missing_freq.size)  # 确保top_n不超过矩阵大小
        flat_indices = np.argsort(missing_freq.flatten())[-top_n:]
        critical_positions = np.unravel_index(flat_indices, missing_freq.shape)

        with open(os.path.join(output_dir, 'critical_positions.txt'), 'w') as f:
            f.write(f"Top {top_n} 关键缺失位置:\n")
            for i in range(top_n):
                t, feat = critical_positions[0][i], critical_positions[1][i]
                f.write(f"位置 {i + 1}: 时间步={t}, 特征={feat}, 缺失频率={missing_freq[t, feat]:.4f}\n")
    except Exception as e:
        print(f"识别关键缺失位置时出错: {e}")
        traceback.print_exc()


def analyze_class_specific_impact(masks_dir, output_dir=None):
    """分析缺失对不同类别的影响，处理掩码形状不一致问题"""
    if output_dir is None:
        output_dir = os.path.join(masks_dir, 'class_impact')
    os.makedirs(output_dir, exist_ok=True)

    # 收集不同类别的掩码
    class_masks = {}
    class_shapes = {}  # 记录每个类别的掩码形状

    for filename in os.listdir(masks_dir):
        if filename.startswith('final_mask_') and filename.endswith('.npz'):
            try:
                data = np.load(os.path.join(masks_dir, filename), allow_pickle=True)

                if 'original_data' in data:
                    mask = data['mask']

                    # 记录掩码形状
                    mask_shape = mask.shape

                    # 处理三维掩码
                    if len(mask.shape) == 3:
                        if mask.shape[2] == 1:
                            mask = mask.squeeze(2)
                        else:
                            # 处理多特征情况
                            batch_size, seq_len, n_features = mask.shape
                            mask = mask.reshape(batch_size, seq_len * n_features)

                    original_data = data['original_data'].item()
                    if 'label' in original_data:
                        try:
                            label = original_data['label'].squeeze()[0]  # 获取样本的标签
                            if label not in class_masks:
                                class_masks[label] = []
                                class_shapes[label] = []
                            class_masks[label].append(mask)
                            class_shapes[label].append(mask_shape)
                        except Exception as e:
                            print(f"处理标签时出错: {e}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    # 计算每个类别的缺失频率并比较
    if len(class_masks) >= 2:
        try:
            plt.figure(figsize=(15, 10))

            # 计算最常见的形状
            from collections import Counter

            for i, (label, masks) in enumerate(class_masks.items()):
                if masks:
                    # 找出该类别最常见的掩码形状
                    shapes = class_shapes[label]
                    shape_counter = Counter([str(shape) for shape in shapes])
                    most_common_shape_str = shape_counter.most_common(1)[0][0]

                    # 将字符串形状转换回元组
                    import ast
                    most_common_shape = ast.literal_eval(most_common_shape_str)

                    print(f"类别 {label} 的最常见掩码形状: {most_common_shape}")

                    # 过滤出相同形状的掩码
                    filtered_masks = []
                    for j, mask in enumerate(masks):
                        if str(class_shapes[label][j]) == most_common_shape_str:
                            filtered_masks.append(mask)

                    print(f"类别 {label} 使用 {len(filtered_masks)}/{len(masks)} 个形状一致的掩码")

                    if not filtered_masks:
                        print(f"类别 {label} 没有足够的形状一致的掩码，跳过")
                        continue

                    # 计算该类别的平均缺失频率
                    avg_mask = np.mean(filtered_masks, axis=0)
                    missing_freq = 1 - avg_mask

                    plt.subplot(len(class_masks), 1, i + 1)
                    plt.imshow(missing_freq, cmap='Reds', aspect='auto')
                    plt.colorbar(label='缺失频率')
                    plt.title(f'类别 {label} 的缺失频率')
                    plt.ylabel('时间步')

            plt.xlabel('特征')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'class_specific_missing.png'))
            plt.close()

            # 计算类别间的缺失差异
            if len(class_masks) == 2:
                labels = list(class_masks.keys())

                # 确保两个类别都有掩码
                if class_masks[labels[0]] and class_masks[labels[1]]:
                    # 找出两个类别共同的形状
                    shape0_counter = Counter([str(s) for s in class_shapes[labels[0]]])
                    shape1_counter = Counter([str(s) for s in class_shapes[labels[1]]])

                    common_shapes = set(shape0_counter.keys()) & set(shape1_counter.keys())

                    if not common_shapes:
                        print("两个类别没有共同的掩码形状，无法比较")
                        return

                    # 使用最常见的共同形状
                    common_shape_str = max(common_shapes, key=lambda s: shape0_counter[s] + shape1_counter[s])
                    import ast
                    common_shape = ast.literal_eval(common_shape_str)

                    print(f"使用共同形状 {common_shape} 进行类别间比较")

                    # 过滤两个类别中符合共同形状的掩码
                    masks1 = []
                    masks2 = []

                    for j, mask in enumerate(class_masks[labels[0]]):
                        if str(class_shapes[labels[0]][j]) == common_shape_str:
                            masks1.append(mask)

                    for j, mask in enumerate(class_masks[labels[1]]):
                        if str(class_shapes[labels[1]][j]) == common_shape_str:
                            masks2.append(mask)

                    if not masks1 or not masks2:
                        print("过滤后没有足够的掩码进行比较")
                        return

                    print(f"类别 {labels[0]} 有 {len(masks1)} 个符合条件的掩码")
                    print(f"类别 {labels[1]} 有 {len(masks2)} 个符合条件的掩码")

                    avg_mask1 = np.mean(masks1, axis=0)
                    avg_mask2 = np.mean(masks2, axis=0)

                    diff = (1 - avg_mask1) - (1 - avg_mask2)  # 缺失频率的差异

                    plt.figure(figsize=(12, 8))
                    cmap = plt.cm.coolwarm
                    plt.imshow(diff, cmap=cmap, aspect='auto')
                    plt.colorbar(label=f'缺失频率差异 (类别{labels[0]}-类别{labels[1]})')
                    plt.xlabel('特征')
                    plt.ylabel('时间步')
                    plt.title('类别间缺失模式差异')
                    plt.savefig(os.path.join(output_dir, 'class_missing_difference.png'))
                    plt.close()
        except Exception as e:
            print(f"分析类别特定影响时出错: {e}")
            traceback.print_exc()


def main_analysis(experiment_name, search=True):
    """运行完整的分析流程

    Args:
        experiment_name: 实验名称，可以是完整路径或部分匹配名称
        search: 是否搜索匹配的目录名(当直接路径不存在时)
    """
    # 首先尝试直接使用提供的名称
    base_dir = f'./results/{experiment_name}'

    # 如果直接路径不存在，并且允许搜索
    if not os.path.exists(base_dir) and search:
        print(f"未找到直接路径: {base_dir}")
        print("搜索包含实验名称的目录...")

        # 获取results目录下的所有子目录
        results_root = './results'
        if os.path.exists(results_root):
            subdirs = [d for d in os.listdir(results_root)
                       if os.path.isdir(os.path.join(results_root, d))]

            # 查找包含实验名称的目录
            matching_dirs = [d for d in subdirs if experiment_name in d]

            if matching_dirs:
                # 选择第一个匹配的目录（或者可以让用户选择）
                base_dir = os.path.join(results_root, matching_dirs[0])
                print(f"找到匹配目录: {base_dir}")
            else:
                print(f"未找到包含 '{experiment_name}' 的目录!")
                return

    # 检查找到的目录中是否有masks子目录
    masks_dir = os.path.join(base_dir, 'masks')
    if not os.path.exists(masks_dir):
        print(f"在 {base_dir} 中未找到掩码目录!")

        # 递归搜索掩码目录
        print("递归搜索掩码目录...")
        found_masks_dirs = []

        for root, dirs, files in os.walk(base_dir):
            if 'masks' in dirs:
                found_masks_dirs.append(os.path.join(root, 'masks'))

        if found_masks_dirs:
            masks_dir = found_masks_dirs[0]  # 使用第一个找到的掩码目录
            print(f"找到掩码目录: {masks_dir}")
        else:
            print("未找到任何掩码目录!")
            return

    # 检查masks目录中是否有掩码文件
    mask_files = [f for f in os.listdir(masks_dir)
                  if f.startswith('final_mask_') and f.endswith('.npz')]

    if not mask_files:
        print(f"在 {masks_dir} 中未找到掩码文件!")
        return

    results_file = os.path.join(base_dir, 'result_adversarial_classification.txt')
    if not os.path.exists(results_file):
        print(f"警告: 未找到结果文件 {results_file}")
        # 尝试在不同位置查找结果文件
        alternative_files = []
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.startswith('result_') and file.endswith('.txt'):
                    alternative_files.append(os.path.join(root, file))

        if alternative_files:
            results_file = alternative_files[0]
            print(f"使用替代结果文件: {results_file}")

    # 设置输出目录
    output_dir = os.path.join(base_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    # 执行分析流程...
    try:
        print(f"使用掩码目录: {masks_dir}")
        print(f"使用结果文件: {results_file}")

        # 1. 分析缺失分布
        print("分析缺失分布...")
        class_masks = analyze_masks_by_class(masks_dir, os.path.join(output_dir, 'class_masks'))

        # 2. 分析不同填充方法的性能
        print("比较填充方法性能...")
        if os.path.exists(results_file):
            compare_filling_methods(results_file, os.path.join(output_dir, 'filling_methods'))
        else:
            print("跳过填充方法比较 - 结果文件不存在")

        # 3. 识别关键缺失位置
        print("识别关键缺失位置...")
        identify_critical_missing(masks_dir, os.path.join(output_dir, 'critical_missing'))

        # 4. 分析缺失对不同类别的影响
        print("分析类别特定影响...")
        analyze_class_specific_impact(masks_dir, os.path.join(output_dir, 'class_impact'))

        # 5. 分析缺失位置对性能的影响
        print("分析缺失位置影响...")
        analyze_missing_impact(masks_dir, results_file, os.path.join(output_dir, 'missing_impact'))

        print(f"分析完成，结果保存至 {output_dir}")

    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        traceback.print_exc()


# 使用示例
if __name__ == "__main__":
    # 使用完整的目录名
    experiment_name = "adversarial_classification_Chinatown_AdvTest_TimesNet_UEA_ftM_sl24_ll24_pl0_dm16_nh8_el2_dl1_df32_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0_train"
    main_analysis(experiment_name)