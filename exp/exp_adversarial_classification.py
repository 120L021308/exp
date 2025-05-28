# 文件: exp/exp_adversarial_classification.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from torch.autograd import Variable

from exp.exp_basic import Exp_Basic
from exp.exp_classification import Exp_Classification
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy


class AdversarialMasking:
    """对抗性缺失掩码生成器类"""

    def __init__(self, args):
        self.args = args
        self.mask_ratio = args.max_missing  # 最大缺失率，默认0.2
        self.temperature = args.tau  # Gumbel-Softmax温度参数，默认0.5
        self.lambda_sparsity = args.lambda_sparsity  # 稀疏性惩罚系数
        self.device = args.device

    # [AdversarialMasking类的其他方法保持不变...]
    def initialize_mask(self, batch_size, seq_len, feature_dim):
        """初始化掩码矩阵的logits参数"""
        # 初始化为较小的负值，使得初始掩码主要为1（保留）
        mask_logits = 0.5 * torch.ones(batch_size, seq_len, feature_dim, device=self.device)
        return mask_logits

    def gumbel_softmax(self, logits, temperature=None, hard=False):
        """Gumbel-Softmax技巧使二元掩码可微"""
        if temperature is None:
            temperature = self.temperature

        # 为二元选择准备logits：[保留(1), 缺失(0)]
        binary_logits = torch.stack([logits, torch.zeros_like(logits)], dim=-1)

        # 应用Gumbel-Softmax
        gumbel_dist = F.gumbel_softmax(binary_logits, tau=temperature, hard=hard, dim=-1)

        # 返回第一个通道的概率，这是保留的概率（即掩码值）
        return gumbel_dist[..., 0]

    def apply_mask(self, x, mask):
        """应用掩码到输入数据，mask形状为[batch_size, seq_len, feature_dim]"""
        return x * mask

    def calculate_sparsity_penalty(self, mask):
        """计算稀疏性惩罚，确保缺失率不超过设定值"""
        # 计算当前掩码中0的比例
        zero_ratio = 1.0 - mask.mean()

        # 如果超过最大缺失率，施加惩罚
        if zero_ratio > self.mask_ratio:
            return self.lambda_sparsity * (zero_ratio - self.mask_ratio) ** 2
        return 0.0

    def project_mask(self, mask_logits):
        """投影掩码以满足稀疏性约束"""
        with torch.no_grad():
            # 计算当前掩码
            mask_probs = torch.sigmoid(mask_logits)
            mask_hard = (mask_probs > 0.5).float()
            zero_ratio = 1.0 - mask_hard.mean()

            # 如果超过最大缺失率
            if zero_ratio > self.mask_ratio:
                # 找出最应该保留的位置
                k = int((1.0 - self.mask_ratio) * mask_logits.numel())
                flat_logits = mask_logits.flatten()
                threshold = torch.sort(flat_logits)[0][k]

                # 调整logits
                mask_logits = torch.where(
                    mask_logits >= threshold,
                    torch.ones_like(mask_logits) * 5.0,  # 强制为1的掩码
                    torch.ones_like(mask_logits) * -5.0  # 强制为0的掩码
                )

        return mask_logits

    # 实现不同的填充方法
    def fill_zero(self, x, mask):
        """零值填充(默认)"""
        return x * mask  # 缺失位置为0

    def fill_mean(self, x, mask):
        masked_x = x * mask
        sum_x = torch.sum(masked_x, dim=1, keepdim=True)
        count_x = torch.sum(mask, dim=1, keepdim=True)
        count_x = torch.where(count_x == 0, torch.ones_like(count_x), count_x)  # Avoid division by zero
        feature_means = sum_x / count_x
        filled_x = x * mask + feature_means * (1 - mask)
        return filled_x

    def fill_knn(self, x, mask, k=5):
        """KNN填充 - 使用k个最相似的样本均值填充"""
        batch_size, seq_len, feature_dim = x.shape
        filled_x = x.clone()

        # 对每个样本进行KNN填充
        for b in range(batch_size):
            # 找出非缺失位置
            valid_mask = mask[b] > 0.5  # [seq_len, feature_dim]

            # 对每个时间步
            for t in range(seq_len):
                # 对每个特征
                for f in range(feature_dim):
                    if mask[b, t, f] < 0.5:  # 缺失位置
                        # 计算当前时间步与所有其他时间步的相似度
                        similarities = []
                        for other_t in range(seq_len):
                            if t == other_t:
                                continue

                            # 计算两个时间步之间的相似度
                            # 只考虑两个时间步中都有值的特征
                            common_features = valid_mask[other_t] & valid_mask[t]
                            if common_features.sum() > 0:
                                sim = torch.norm(
                                    x[b, t, common_features] - x[b, other_t, common_features]
                                ).item()
                                similarities.append((other_t, sim))

                        if similarities:
                            # 按相似度排序，选取k个最相似的
                            similarities.sort(key=lambda x: x[1])
                            top_k = similarities[:min(k, len(similarities))]

                            # 使用这k个时间步中该特征的均值填充
                            values = [x[b, t_idx, f].item() for t_idx, _ in top_k if mask[b, t_idx, f] > 0.5]
                            if values:
                                filled_x[b, t, f] = sum(values) / len(values)
                            else:
                                # 如果找不到有效值，使用全局均值
                                valid_values = x[b, valid_mask[:, f], f]
                                if len(valid_values) > 0:
                                    filled_x[b, t, f] = valid_values.mean()

        return filled_x

    def fill_interpolation(self, x, mask):
        """线性插值填充 (per feature, per sample)"""
        batch_size, seq_len, feature_dim = x.shape
        filled_x = x.clone()

        for b in range(batch_size):
            for f in range(feature_dim):
                # Get the values and their indices where data is not missing
                valid_mask_f = mask[b, :, f] > 0.5
                indices = torch.arange(seq_len, device=x.device)

                valid_indices = indices[valid_mask_f]
                valid_values = x[b, valid_mask_f, f]

                if len(valid_indices) == 0:  # No valid points to interpolate from
                    continue
                if len(valid_indices) == 1:  # Only one valid point, fill with it
                    filled_x[b, ~valid_mask_f, f] = valid_values[0]
                    continue

                # Perform interpolation for missing points
                missing_indices = indices[~valid_mask_f]
                if len(missing_indices) > 0:
                    # np.interp requires numpy arrays
                    interp_values = np.interp(missing_indices.cpu().numpy(),
                                              valid_indices.cpu().numpy(),
                                              valid_values.cpu().numpy())
                    filled_x[b, missing_indices, f] = torch.tensor(interp_values, dtype=x.dtype, device=x.device)
        return filled_x

    def fill_with_method(self, x, mask, method='zero'):
        """使用指定方法填充缺失值"""
        if method == 'zero':
            return self.fill_zero(x, mask)
        elif method == 'mean':
            return self.fill_mean(x, mask)
        elif method == 'knn':
            return self.fill_knn(x, mask)
        elif method == 'interpolation':
            return self.fill_interpolation(x, mask)
        else:
            raise ValueError(f"未知的填充方法: {method}")




class Exp_AdversarialClassification(Exp_Basic):
    """对抗性缺失分类实验类"""

    def __init__(self, args):
        # 首先，添加默认参数
        if not hasattr(args, 'target_performance_drop'):
            args.target_performance_drop = 0.3  # 默认目标性能下降30%
        if not hasattr(args, 'performance_threshold'):
            args.performance_threshold = 0.05  # 默认性能差异阈值
        if not hasattr(args, 'mask_learning_rate'):
            args.mask_learning_rate = 0.01  # 默认掩码学习率
        if not hasattr(args, 'filling_method'):
            args.filling_method = 'mean'  # 默认填充方法

        # 调用基类初始化，这会调用_build_model方法
        super(Exp_AdversarialClassification, self).__init__(args)

        # 创建对抗掩码工具
        self.adv_masking = AdversarialMasking(args)
        self.original_accuracy = None  # 存储原始模型准确率
        self.target_accuracy = None  # 目标准确率

    def save_masks(self, masks, setting, original_data=None):
        """保存最终的对抗性掩码矩阵到文件

        Args:
            masks: 包含掩码信息的列表，每个元素是一个字典
            setting: 实验设置名称，用于文件路径
            original_data: 原始数据列表，与掩码对应
        """
        # 创建结果目录
        folder_path = f'./results/{setting}/masks/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 只保存最终epoch的掩码
        final_masks = []
        if masks:
            # 获取最大的epoch值，即最后一个epoch
            max_epoch = max([m.get('epoch', 0) for m in masks])
            final_masks = [m for m in masks if m.get('epoch', 0) == max_epoch]

            print(f"过滤后只保留最终epoch {max_epoch}的掩码，共 {len(final_masks)} 个")

        # 保存最终掩码为 numpy 数组
        saved_count = 0
        for i, mask_info in enumerate(final_masks):
            if 'mask' in mask_info:
                batch_idx = mask_info.get('batch_idx', i)

                # 创建保存数据
                save_data = {
                    'mask': mask_info['mask'],
                    'zero_ratio': mask_info.get('zero_ratio', 0),
                    'batch_idx': batch_idx,
                    'epoch': mask_info.get('epoch', 0)
                }

                # 如果有原始数据，也保存对应的原始数据
                if original_data is not None and batch_idx < len(original_data):
                    save_data['original_data'] = original_data[batch_idx]

                # 保存为 numpy 文件
                file_name = f'final_mask_b{batch_idx}.npz'
                np.savez(os.path.join(folder_path, file_name), **save_data)
                saved_count += 1

        # 另外保存一个汇总文件，包含所有掩码的统计信息
        if final_masks:
            summary_data = {
                'zero_ratios': [m.get('zero_ratio', 0) for m in final_masks if 'mask' in m],
                'batch_indices': [m.get('batch_idx', i) for i, m in enumerate(final_masks) if 'mask' in m]
            }
            np.savez(os.path.join(folder_path, 'final_masks_summary.npz'), **summary_data)

        print(f"已保存 {saved_count} 个最终掩码矩阵到 {folder_path}")


    def _build_model(self):
        """构建模型 - 必须实现，被Exp_Basic.__init__调用"""
        # 获取数据集，用于确定模型输入维度
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')

        # 设置模型参数
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)

        # 初始化模型
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """获取数据 - 必须实现基类方法"""
        from data_provider.data_factory import data_provider
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """选择优化器"""
        model_optim = torch.optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """选择损失函数"""
        criterion = nn.CrossEntropyLoss()
        return criterion

    def _get_original_performance(self):
        """获取原始模型性能作为基准"""
        if self.original_accuracy is None:
            print("评估原始模型性能...")
            test_data, test_loader = self._get_data(flag='TEST')
            criterion = self._select_criterion()

            # 加载模型
            if os.path.exists(os.path.join(self.args.checkpoints, 'original_model.pth')):
                print("加载预训练的原始模型...")
                self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, 'original_model.pth')))

            # 评估性能
            _, self.original_accuracy = self.vali(test_data, test_loader, criterion, apply_mask=False)
            print(f"原始模型准确率: {self.original_accuracy:.4f}")

            # 保存原始模型
            if not os.path.exists(os.path.join(self.args.checkpoints)):
                os.makedirs(os.path.join(self.args.checkpoints))
            torch.save(self.model.state_dict(), os.path.join(self.args.checkpoints, 'original_model.pth'))

        return self.original_accuracy

    def _calculate_target_accuracy(self):
        """计算目标准确率"""
        if self.target_accuracy is None:
            original_acc = self._get_original_performance()
            self.target_accuracy = original_acc * (1.0 - self.args.target_performance_drop)
            print(
                f"目标准确率: {self.target_accuracy:.4f} (原始准确率的{(1.0 - self.args.target_performance_drop):.1%})")
        return self.target_accuracy

    def train_adversarial_mask(self, batch_x, label, padding_mask):
        """训练对抗性掩码"""
        target_accuracy = self._calculate_target_accuracy()

        # 初始化掩码logits
        batch_size, seq_len, feature_dim = batch_x.shape
        mask_logits = self.adv_masking.initialize_mask(batch_size, seq_len, feature_dim)
        mask_logits = Variable(mask_logits, requires_grad=True)

        # 使用Adam优化器优化掩码
        mask_optimizer = torch.optim.Adam([mask_logits], lr=self.args.mask_learning_rate)

        # 将模型设为评估模式
        self.model.eval()
        criterion = self._select_criterion()

        max_iterations = 50  # 最大迭代次数
        best_mask = None
        best_accuracy_diff = float('inf')

        for iter_idx in range(max_iterations):
            # 使用Gumbel-Softmax生成可微分掩码
            hard = (iter_idx > 5)  # 前几次迭代使用软掩码，之后使用硬掩码
            # 在硬掩码模式下降低温度，使分布更加尖锐
            temperature = self.adv_masking.temperature if not hard else self.adv_masking.temperature * 0.5

            mask = self.adv_masking.gumbel_softmax(mask_logits, temperature=temperature, hard=hard)

            # 计算稀疏性惩罚
            sparsity_penalty = self.adv_masking.calculate_sparsity_penalty(mask)

            # 应用掩码到输入
            masked_x = self.adv_masking.apply_mask(batch_x, mask)

            # 前向传播计算损失
            outputs = self.model(masked_x, padding_mask, None, None)
            classification_loss = criterion(outputs, label.long().squeeze(-1))

            # 总损失 = 分类损失 + 稀疏性惩罚
            loss = classification_loss + sparsity_penalty

            # 计算当前准确率
            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probs, dim=1)
                correct = (predictions == label.squeeze(-1)).float().mean()
                current_accuracy = correct.item()

                # 计算与目标准确率的差异
                accuracy_diff = abs(current_accuracy - target_accuracy)

                # 记录最佳掩码
                if accuracy_diff < best_accuracy_diff:
                    best_accuracy_diff = accuracy_diff
                    best_mask = mask.detach().clone()

                # 检查是否达到目标准确率
                if accuracy_diff < self.args.performance_threshold:
                    print(f"达到目标准确率，迭代次数: {iter_idx}: {current_accuracy:.4f}")
                    break

            # 计算梯度并更新掩码
            mask_optimizer.zero_grad()

            # 如果当前准确率高于目标，则增加损失（使准确率降低）
            # 否则降低损失（避免准确率降低过多）
            loss_sign = 1.0 if current_accuracy > target_accuracy else -1.0
            loss = loss_sign * classification_loss + sparsity_penalty

            loss.backward()
            mask_optimizer.step()

            # 投影掩码以满足稀疏性约束
            with torch.no_grad():
                mask_logits.data = self.adv_masking.project_mask(mask_logits.data)

            if iter_idx % 5 == 0 or iter_idx == max_iterations - 1:
                # 计算掩码中0的比例
                zero_ratio = 1.0 - mask.mean().item()
                print(f"迭代 {iter_idx}, 准确率: {current_accuracy:.4f}, "
                      f"目标: {target_accuracy:.4f}, 缺失率: {zero_ratio:.4f}")

        # 使用最佳掩码或最终掩码
        final_mask = best_mask if best_mask is not None else mask.detach()

        # 应用最终掩码并使用指定的填充方法
        masked_x = batch_x.clone()
        masked_x = self.adv_masking.apply_mask(masked_x, final_mask)

        # 填充缺失值
        filled_x = self.adv_masking.fill_with_method(
            batch_x, final_mask, method=self.args.filling_method
        )

        # 将模型恢复为训练模式
        self.model.train()

        # 记录掩码统计信息
        mask_stats = {
            'zero_ratio': (1.0 - final_mask.mean().item()),
            'mask': final_mask.cpu().numpy()
        }

        return filled_x, final_mask, mask_stats

    def vali(self, vali_data, vali_loader, criterion, apply_mask=True, save_masks=False, return_original_data=False):
        """验证方法，可以选择是否应用对抗性掩码"""
        total_loss = []
        preds = []
        trues = []
        masks = []
        original_data_list = []  # 存储原始数据
        self.model.eval()

        print(f"验证集大小: {len(vali_loader)}")

        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # 存储原始数据
                if return_original_data:
                    original_data_list.append({
                        'batch_x': batch_x.cpu().numpy(),
                        'label': label.cpu().numpy(),
                        'batch_idx': i
                    })

                # 在验证时应用掩码和填充
                applied_mask = None
                if apply_mask and hasattr(self.args, 'apply_mask_in_validation') and self.args.apply_mask_in_validation:
                    batch_size, seq_len, feature_dim = batch_x.shape
                    # 生成随机掩码
                    mask_probs = torch.rand(batch_size, seq_len, feature_dim, device=self.device)
                    mask = (mask_probs > self.args.max_missing).float()  # 随机生成掩码
                    applied_mask = mask.cpu().numpy()
                    masks.append({
                        'mask': applied_mask,
                        'batch_idx': i,
                        'zero_ratio': 1.0 - mask.mean().item()
                    })

                    # 应用掩码和填充
                    batch_x = self.adv_masking.fill_with_method(
                        batch_x, mask, method=self.args.filling_method
                    )

                try:
                    outputs = self.model(batch_x, padding_mask, None, None)

                    if outputs is None:
                        print(f"  警告: 模型输出为None，跳过此批次")
                        continue

                    pred = outputs.detach().cpu()
                    loss = criterion(pred, label.long().squeeze().cpu())
                    total_loss.append(loss)

                    preds.append(outputs.detach())
                    trues.append(label)

                except Exception as e:
                    print(f"  处理批次 {i} 时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        print(f"处理完成，preds列表长度: {len(preds)}")

        if len(preds) == 0:
            print("警告: preds列表为空，无法计算准确率")
            return (0.0, 0.0, []) if return_original_data else (0.0, 0.0)

        total_loss = np.average(total_loss) if total_loss else 0.0

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()

        # 返回原始数据（如果需要）
        if return_original_data:
            return total_loss, accuracy, original_data_list

        # 返回掩码信息（如果应用了掩码）
        if apply_mask and hasattr(self.args,
                                  'apply_mask_in_validation') and self.args.apply_mask_in_validation and masks:
            return total_loss, accuracy, masks

        return total_loss, accuracy

    def train(self, setting):
        """训练方法以集成对抗性掩码和填充 (已修改以确保早停时保存掩码)"""
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')  # 通常验证集用 'VAL' 或 'VALID' flag，这里按原样保留 'TEST'
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 首先评估原始模型性能作为基准
        self._get_original_performance()
        print("训练开始...")

        # 计算目标准确率
        target_acc = self._calculate_target_accuracy()  # target_acc 变量未使用，可以移除或后续使用

        # 存储最终epoch的原始数据和掩码
        final_masks_info = []
        final_original_data = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            masks_info = []  # 存储当前epoch的掩码信息用于分析
            epoch_original_data = []  # 存储当前epoch的原始数据

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # 存储原始数据
                original_batch_data = {
                    'batch_x': batch_x.cpu().numpy(),
                    'label': label.cpu().numpy(),
                    'batch_idx': i  # 这是批次在当前 epoch 内的索引
                }
                epoch_original_data.append(original_batch_data)

                # 生成对抗性掩码并应用
                if hasattr(self.args, 'use_adversarial_mask') and self.args.use_adversarial_mask:
                    # 确保 self.adv_masking 已经初始化
                    if not hasattr(self, 'adv_masking'):
                        # 如果在 __init__ 中创建，这里应该总是存在
                        # 但作为安全检查，可以考虑在 __init__ 中确保它被创建
                        # 或者在这里按需创建，但这通常不是最佳实践
                        print("警告: AdversarialMasking 工具未初始化!")
                        # 可以选择跳过或引发错误
                    else:
                        masked_x, mask, mask_stats = self.train_adversarial_mask(batch_x, label, padding_mask)
                        # 记录掩码信息
                        mask_stats.update({
                            'batch_idx': i,  # 这是批次在当前 epoch 内的索引
                            'epoch': epoch
                        })
                        masks_info.append(mask_stats)
                        batch_x = masked_x  # 使用经过掩码和填充处理后的数据进行训练

                # 如果不使用对抗性掩码但需要随机掩码
                elif hasattr(self.args, 'use_random_mask') and self.args.use_random_mask:
                    # 确保 self.adv_masking 已经初始化 (因为它包含填充方法)
                    if not hasattr(self, 'adv_masking'):
                        print("警告: AdversarialMasking 工具未初始化 (用于随机掩码填充)!")
                    else:
                        batch_size_rand, seq_len_rand, feature_dim_rand = batch_x.shape
                        # 生成随机掩码
                        mask_probs_rand = torch.rand(batch_size_rand, seq_len_rand, feature_dim_rand,
                                                     device=self.device)
                        mask_rand = (mask_probs_rand > self.args.max_missing).float()
                        # 应用掩码和填充
                        batch_x = self.adv_masking.fill_with_method(
                            batch_x, mask_rand, method=self.args.filling_method
                        )
                        # 记录掩码信息
                        masks_info.append({
                            'mask': mask_rand.cpu().numpy(),
                            'zero_ratio': (1.0 - mask_rand.mean().item()),
                            'batch_idx': i,
                            'epoch': epoch
                        })

                outputs = self.model(batch_x, padding_mask, None, None)  # 这里的 batch_x 可能是原始的，也可能是处理过的
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:  # 每100个iter打印一次信息
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                # 梯度裁剪，防止梯度爆炸
                if hasattr(self.args,
                           'clip_grad_norm') and self.args.clip_grad_norm is not None and self.args.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad_norm)
                else:  # 使用原代码的默认值或一个通用值
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            # --- Epoch 结束 ---
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss_avg = np.average(train_loss) if train_loss else 0.0  # 处理 train_loss 为空的情况

            # 在验证/测试前，将模型设置为评估模式
            self.model.eval()  # 确保在vali和test中模型是eval模式，vali方法内部应该也有
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion, apply_mask=False)  # 验证时通常不用掩码
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion, apply_mask=False)  # 测试时通常不用掩码
            self.model.train()  # 恢复为训练模式，准备下一个epoch

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss_avg, vali_loss, val_accuracy, test_loss, test_accuracy))

            # 【修改点】始终用当前已完成的epoch的掩码和原始数据更新 final_masks_info 和 final_original_data
            # 这样即使发生早停，这些变量也会包含最后一个实际执行的完整epoch的数据。
            # save_masks 函数内部会处理只保存最后一个epoch（max_epoch）的掩码。
            if masks_info:  # 只有当 masks_info 不为空时才更新，避免覆盖成空列表
                final_masks_info = masks_info
                final_original_data = epoch_original_data

            early_stopping(-val_accuracy, self.model, path)  # Early stopping监视验证集准确率（负数，因为通常早停类监视损失）
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出 epoch 循环

        # --- 训练循环结束 ---
        print("加载最佳模型 (来自早停或最后一个epoch的模型)...")
        best_model_path = os.path.join(path, 'checkpoint.pth')  # 确保路径正确
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        else:
            print(f"警告: 未找到最佳模型路径 {best_model_path}。将使用最后一个epoch的模型状态。")

        # 保存最终的掩码和原始数据
        if final_masks_info:  # 如果 final_masks_info 有内容 (即至少一个epoch生成了掩码)
            print(f"准备保存 {len(final_masks_info)} 条掩码信息 (来自最后一个实际运行的epoch)...")
            self.save_masks(final_masks_info, setting + '_train', final_original_data)
        else:
            print("没有可供保存的训练掩码信息 (final_masks_info 为空)。")

        return self.model

    def test(self, setting, test=0):
        """测试方法，评估不同填充方法的性能并保存掩码"""
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('加载模型')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # 获取原始性能
        criterion = self._select_criterion()
        print("评估原始模型性能...")
        original_loss, original_accuracy = self.vali(test_data, test_loader, criterion, apply_mask=False)
        print(f"原始模型准确率: {original_accuracy:.4f}")

        # 评估各种填充方法的性能
        filling_methods = ['zero', 'mean', 'knn', 'interpolation']
        results = {'original': original_accuracy}

        # 用于收集所有测试掩码和原始数据
        test_masks_info = []
        test_original_data = []

        # 首先获取所有原始测试数据
        _, _, original_data_list = self.vali(
            test_data, test_loader, criterion, apply_mask=False, return_original_data=True
        )
        test_original_data = original_data_list

        for method in filling_methods:
            print(f"使用{method}填充方法评估模型性能...")
            self.args.filling_method = method
            self.args.apply_mask_in_validation = True

            # 评估性能并获取掩码
            method_loss, method_accuracy, method_masks = self.vali(
                test_data, test_loader, criterion, apply_mask=True
            )

            # 将掩码添加到总列表
            for mask_info in method_masks:
                mask_info['method'] = method
                mask_info['epoch'] = 0  # 测试阶段设置为0
                test_masks_info.append(mask_info)

            results[method] = method_accuracy

            print(f"使用{method}填充的准确率: {method_accuracy:.4f}, "
                  f"相比原始下降: {(original_accuracy - method_accuracy) / original_accuracy:.2%}")

        # 保存测试期间收集的掩码和原始数据
        if test_masks_info:
            self.save_masks(test_masks_info, setting + '_test', test_original_data)

        # 结果保存
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = 'result_adversarial_classification.txt'
        with open(os.path.join(folder_path, file_name), 'a') as f:
            f.write(setting + "  \n")
            f.write(f'原始准确率:{original_accuracy:.4f}\n')

            for method, acc in results.items():
                if method != 'original':
                    drop_rate = (original_accuracy - acc) / original_accuracy
                    f.write(f'{method}填充准确率:{acc:.4f}, 下降率:{drop_rate:.2%}\n')

            f.write('\n')

        return results