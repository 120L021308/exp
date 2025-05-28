# 文件: exp/exp_adversarial_classification.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from torch.autograd import Variable

from exp.exp_basic import Exp_Basic
# from exp.exp_classification import Exp_Classification # 如果此类未使用，可以注释掉
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy


class AdversarialMasking:
    """对抗性缺失掩码生成器类"""

    def __init__(self, args):
        self.args = args
        self.mask_ratio = args.max_missing  # 最大缺失率，默认0.2
        self.temperature = args.tau  # Gumbel-Softmax温度参数，默认0.5
        # 确保 lambda_sparsity 在 args 中存在
        self.lambda_sparsity = getattr(args, 'lambda_sparsity', 1.0)  # 稀疏性惩罚系数, 增加默认值
        self.device = args.device

    def initialize_mask(self, batch_size, seq_len, feature_dim):
        """初始化掩码矩阵的logits参数"""
        mask_logits = -3.0 * torch.ones(batch_size, seq_len, feature_dim, device=self.device)
        return mask_logits

    def gumbel_softmax(self, logits, temperature=None, hard=False):
        """Gumbel-Softmax技巧使二元掩码可微"""
        if temperature is None:
            temperature = self.temperature
        binary_logits = torch.stack([logits, torch.zeros_like(logits)], dim=-1)
        gumbel_dist = F.gumbel_softmax(binary_logits, tau=temperature, hard=hard, dim=-1)
        return gumbel_dist[..., 0]

    def apply_mask(self, x, mask):
        """应用掩码到输入数据，mask形状为[batch_size, seq_len, feature_dim]"""
        return x * mask

    def calculate_sparsity_penalty(self, mask):
        """计算稀疏性惩罚，确保缺失率不超过设定值"""
        zero_ratio = 1.0 - mask.mean()
        if zero_ratio > self.mask_ratio:
            return self.lambda_sparsity * (zero_ratio - self.mask_ratio) ** 2
        return 0.0

    def project_mask(self, mask_logits):
        """投影掩码以满足稀疏性约束"""
        with torch.no_grad():
            mask_probs = torch.sigmoid(mask_logits)  # More numerically stable than softmax for binary
            mask_hard = (mask_probs > 0.5).float()
            current_zero_ratio = 1.0 - mask_hard.mean()

            if current_zero_ratio > self.mask_ratio:
                # Find the k-th largest logit to keep (1 - mask_ratio) proportion of ones
                num_elements_to_keep = int((1.0 - self.mask_ratio) * mask_logits.numel())
                if num_elements_to_keep < mask_logits.numel():  # Avoid error if mask_ratio is 0
                    # We want to find the threshold for the smallest logits to set to 0
                    # So, find (1-mask_ratio)*N largest logits
                    threshold = torch.kthvalue(mask_logits.flatten(), mask_logits.numel() - num_elements_to_keep).values
                    # For logits >= threshold, make them positive (mask = 1)
                    # For logits < threshold, make them negative (mask = 0)
                    mask_logits_projected = torch.where(mask_logits >= threshold,
                                                        torch.ones_like(mask_logits) * 5.0,
                                                        torch.ones_like(mask_logits) * -5.0)
                    return mask_logits_projected
        return mask_logits

    # 实现不同的填充方法
    def fill_zero(self, x, mask):
        """零值填充(默认)"""
        return x * mask

    def fill_mean(self, x, mask):
        """均值填充"""
        # 计算每个特征在非缺失位置的均值
        masked_x = x * mask
        sum_x = torch.sum(masked_x, dim=1, keepdim=True)
        count_x = torch.sum(mask, dim=1, keepdim=True)
        count_x = torch.where(count_x == 0, torch.ones_like(count_x), count_x)  # Avoid division by zero
        feature_means = sum_x / count_x
        filled_x = x * mask + feature_means * (1 - mask)
        return filled_x

    def fill_knn(self, x, mask, k=5):
        """KNN填充 - 简单的实现，可能较慢，仅用于演示"""
        batch_size, seq_len, feature_dim = x.shape
        filled_x = x.clone()

        for b in range(batch_size):
            for t_miss in range(seq_len):
                for f_miss in range(feature_dim):
                    if mask[b, t_miss, f_miss] < 0.5:  # If missing
                        distances = []
                        valid_neighbors_values = []
                        # Find k nearest temporal neighbors for this feature
                        for t_neighbor in range(seq_len):
                            if t_neighbor == t_miss:
                                continue
                            # Consider only neighbors where the feature f_miss is NOT missing
                            if mask[b, t_neighbor, f_miss] > 0.5:
                                # Simple distance based on other features at these time steps (if needed)
                                # For simplicity here, we just collect values of the same feature from neighbors
                                valid_neighbors_values.append(x[b, t_neighbor, f_miss].item())
                                # A more sophisticated distance would compare feature vectors at t_miss and t_neighbor
                                # For now, let's just use temporal proximity (less ideal) or average

                        if valid_neighbors_values:
                            # Simplistic: use mean of all valid neighbors of this feature
                            # A true KNN would sort by distance and take k closest
                            # This part needs a proper distance metric and selection for KNN
                            if len(valid_neighbors_values) >= k:  # If enough neighbors
                                # This is not true KNN, just an example.
                                # A real KNN would require a distance metric between time points.
                                # For now, let's use the mean of available values for that feature.
                                filled_x[b, t_miss, f_miss] = torch.tensor(valid_neighbors_values).mean()
                            elif valid_neighbors_values:  # use mean if less than k
                                filled_x[b, t_miss, f_miss] = torch.tensor(valid_neighbors_values).mean()
                            # else: remains as is (or could be filled with global mean as fallback)
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

    def fill_with_method(self, x_original, mask, method='zero'):
        """使用指定方法填充缺失值"""
        x_masked = x_original * mask  # Start with masked data
        if method == 'zero':
            return x_masked
        elif method == 'mean':
            return self.fill_mean(x_original.clone(), mask)  # Pass original for mean calculation
        elif method == 'knn':
            return self.fill_knn(x_original.clone(), mask)
        elif method == 'interpolation':
            return self.fill_interpolation(x_original.clone(), mask)
        else:
            raise ValueError(f"未知的填充方法: {method}")


class Exp_AdversarialClassification(Exp_Basic):
    """对抗性缺失分类实验类"""

    def __init__(self, args):
        # 添加实验特定默认参数
        default_args = {
            'target_performance_drop': 0.3,
            'performance_threshold': 0.05,
            'mask_learning_rate': 0.01,
            'filling_method': 'zero',
            'lambda_sparsity': 1.0,  # 确保稀疏性惩罚系数存在
            'max_missing': 0.2,  # 确保最大缺失率存在
            'tau': 0.5,  # 确保Gumbel温度存在
            'use_adversarial_mask': False,  # 训练时是否用对抗掩码
            'use_random_mask': False,  # 训练时是否用随机掩码 (如果不用对抗)
            'apply_mask_in_validation': False,  # 验证/测试时是否应用掩码
            'test_mask_type': 'random',  # 测试时使用的掩码类型: 'random' 或 'adversarial'
        }
        for k, v in default_args.items():
            if not hasattr(args, k):
                setattr(args, k, v)

        super(Exp_AdversarialClassification, self).__init__(args)
        self.adv_masking = AdversarialMasking(args)
        self.original_accuracy = None
        self.target_accuracy = None

    def save_masks(self, masks_info_list, setting_str, original_data_list=None, mask_type="unknown"):
        """保存掩码矩阵到文件
        Args:
            masks_info_list: 包含掩码信息的列表，每个元素是一个字典
                             {'mask': ndarray, 'zero_ratio': float, 'batch_idx': int, 'epoch': int, 'method'(optional):str}
            setting_str: 实验设置名称，用于文件路径 (例如: YourSetting_train_adv_masks, YourSetting_test_rand_masks)
            original_data_list: 原始数据列表，与掩码对应 [{'batch_x': ndarray, 'label': ndarray, 'batch_idx': int}]
            mask_type: 掩码类型 (e.g., 'adversarial_train', 'random_test', 'adversarial_test')
        """
        folder_path = f'./results/{self.args.model}_{self.args.data_path.split("/")[-1].split(".")[0]}/{setting_str}/masks_{mask_type}/'  # 更详细的路径
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        processed_masks = []
        if masks_info_list:
            # 如果是训练掩码，通常只保存最后一个epoch的
            if 'train' in mask_type:
                max_epoch = max([m.get('epoch', -1) for m in masks_info_list], default=-1)
                if max_epoch != -1:
                    processed_masks = [m for m in masks_info_list if m.get('epoch', -1) == max_epoch]
                    print(f"保存来自 epoch {max_epoch} 的 {len(processed_masks)} 个训练掩码批次。")
                else:
                    processed_masks = masks_info_list  # Fallback
            else:  # 测试掩码，全部保存
                processed_masks = masks_info_list
                print(f"保存 {len(processed_masks)} 个测试掩码批次。")

        saved_count = 0
        for i, mask_info in enumerate(processed_masks):
            if 'mask' in mask_info and isinstance(mask_info['mask'], np.ndarray):
                batch_idx = mask_info.get('batch_idx', i)
                epoch_num = mask_info.get('epoch', 0)  # 默认为0 (例如测试时)

                save_data = {
                    'mask': mask_info['mask'],  # Should be (batch_size, seq_len, feature_dim)
                    'zero_ratio': mask_info.get('zero_ratio', 0),
                    'batch_idx': batch_idx,
                    'epoch': epoch_num,
                    'mask_type': mask_type,
                    'filling_method_at_generation': mask_info.get('filling_method',
                                                                  self.args.filling_method if 'train' in mask_type else 'N/A')
                }
                if 'method' in mask_info:  # For test masks, this indicates the filling method being evaluated
                    save_data['evaluated_filling_method'] = mask_info['method']

                # 匹配并保存原始数据
                if original_data_list:
                    # Find corresponding original data by batch_idx
                    # This assumes original_data_list is a list of dicts, each for a batch
                    original_batch = next((od for od in original_data_list if od.get('batch_idx') == batch_idx), None)
                    if original_batch and 'batch_x' in original_batch and 'label' in original_batch:
                        # Ensure mask and data batch sizes align, or save sample-wise if they don't match perfectly.
                        # Current mask is per batch. Original data is also per batch.
                        save_data['original_batch_x'] = original_batch['batch_x']
                        save_data['original_label'] = original_batch['label']
                    else:
                        print(f"警告: 无法为掩码批次 {batch_idx} 找到或匹配原始数据。")

                file_name = f'mask_epoch{epoch_num}_b{batch_idx}.npz'
                try:
                    np.savez_compressed(os.path.join(folder_path, file_name), **save_data)
                    saved_count += 1
                except Exception as e:
                    print(f"保存掩码文件 {file_name} 失败: {e}")
            else:
                print(f"警告: 掩码信息不完整或掩码不是ndarray, 跳过批次 {mask_info.get('batch_idx', i)}")

        if processed_masks:
            summary_data = {
                'zero_ratios': [m.get('zero_ratio', 0) for m in processed_masks if 'mask' in m],
                'batch_indices': [m.get('batch_idx', i) for i, m in enumerate(processed_masks) if 'mask' in m],
                'epochs': [m.get('epoch', 0) for m in processed_masks if 'mask' in m],
                'mask_type': mask_type,
            }
            np.savez_compressed(os.path.join(folder_path, f'summary_{mask_type}.npz'), **summary_data)

        print(f"已保存 {saved_count} 个掩码矩阵 ({mask_type}) 到 {folder_path}")

    def _build_model(self):
        train_data, _ = self._get_data(flag='TRAIN')
        test_data, _ = self._get_data(flag='TEST')  # To get overall max_seq_len
        self.args.seq_len = train_data.max_seq_len  # Use train_data for model structure usually
        if hasattr(test_data, 'max_seq_len'):
            self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        from data_provider.data_factory import data_provider
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = torch.optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def _get_original_performance(self, force_recalc=False):
        if self.original_accuracy is None or force_recalc:
            print("评估原始模型性能...")
            model_path = os.path.join(self.args.checkpoints, 'original_model.pth')
            # Prefer loading a saved original model if it exists and no retraining is implied
            if os.path.exists(
                    model_path) and not force_recalc:  # and not self.args.train_only_adversarial_mask: ( hypothetical arg)
                print(f"加载预训练的原始模型从 {model_path}...")
                try:
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                except Exception as e:
                    print(f"加载原始模型失败: {e}. 将重新训练/评估。")
                    self._train_original_model_if_needed()  # A helper to train if not exists
            else:
                self._train_original_model_if_needed()

            test_data, test_loader = self._get_data(flag='TEST')
            criterion = self._select_criterion()
            _, self.original_accuracy = self._evaluate(test_data, test_loader, criterion,
                                                       apply_mask_config=None)  # No mask for original
            print(f"原始模型准确率: {self.original_accuracy:.4f}")
        return self.original_accuracy

    def _train_original_model_if_needed(self):
        """ Helper to train and save an original model if it doesn't exist or needs retraining """
        original_model_path = os.path.join(self.args.checkpoints, 'original_model.pth')
        # Simplified: if not exists, run a brief training.
        # In a real scenario, you might have separate flags or training for the original model.
        if not os.path.exists(original_model_path):
            print("原始模型未找到或需要重新训练。现在训练原始模型...")
            # Placeholder: A minimal training loop for the original model.
            # This should ideally be a full training run saved separately.
            # For this example, we'll do a mock training or just save the current state
            # if this method is called before any other training.
            # Or, ensure your main script runs a phase to pretrain and save this.
            # For now, let's assume the model is initialized and we save it.
            # A proper original model training would involve its own loop.
            print("警告: 执行简化的原始模型'训练'(仅保存当前状态或需要完整训练流程)")
            # Simulate training the base model for a few epochs if needed
            # For this example, we assume Exp_Basic or a prior step handles base model training.
            # If not, you need to implement a training loop here for the original model.
            # This simplified version just saves the current model state if no file.
            # It's better to train and save 'original_model.pth' from a standard training run first.
            if not os.path.exists(self.args.checkpoints):
                os.makedirs(self.args.checkpoints)
            # Fallback: if we are here means we need it, so let's do a quick train if this is the first time.
            # This is a conceptual placeholder. Proper training of the original model is assumed.
            # A simple loop:
            # optim = self._select_optimizer()
            # crit = self._select_criterion()
            # train_data, train_loader = self._get_data(flag='TRAIN')
            # for _ in range(min(5, self.args.train_epochs // 2)): # Few epochs
            #     for batch_x, label, padding_mask in train_loader:
            #         optim.zero_grad()
            #         outputs = self.model(batch_x.float().to(self.device), padding_mask.float().to(self.device), None, None)
            #         loss = crit(outputs, label.to(self.device).long().squeeze(-1))
            #         loss.backward()
            #         optim.step()
            torch.save(self.model.state_dict(), original_model_path)
            print(f"原始模型已保存到 {original_model_path}")

    def _calculate_target_accuracy(self):
        if self.target_accuracy is None:
            original_acc = self._get_original_performance()
            self.target_accuracy = original_acc * (1.0 - self.args.target_performance_drop)
            print(
                f"目标准确率 (基于对抗攻击): {self.target_accuracy:.4f} (原始准确率的{(1.0 - self.args.target_performance_drop):.1%})")
        return self.target_accuracy

    def _generate_adversarial_mask_batch(self, batch_x_orig, label, padding_mask, filling_method_for_attack):
        """生成对抗性掩码 (不修改模型,仅生成掩码)"""
        # Ensure model is loaded and in eval mode for mask generation
        # self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, 'original_model.pth'), map_location=self.device))
        self.model.eval()  # Ensure model is in eval mode for consistent mask generation

        target_accuracy_for_mask = self._calculate_target_accuracy()
        batch_size, seq_len, feature_dim = batch_x_orig.shape
        mask_logits = self.adv_masking.initialize_mask(batch_size, seq_len, feature_dim)
        mask_logits = Variable(mask_logits, requires_grad=True)
        mask_optimizer = torch.optim.Adam([mask_logits], lr=self.args.mask_learning_rate)
        criterion_mask_gen = self._select_criterion()  # Use the same criterion for evaluating effect of mask

        best_mask_tensor = None
        best_accuracy_diff = float('inf')
        final_mask_for_batch = None
        achieved_accuracy_at_best_mask = 0.0

        # Temporarily store current model learning rate if it's part of self.args to restore later
        # current_model_lr = self.args.learning_rate

        # It's crucial that the model parameters are frozen during mask optimization
        for param in self.model.parameters():
            param.requires_grad = False

        for iter_idx in range(self.args.mask_opt_iterations if hasattr(self.args,
                                                                       'mask_opt_iterations') else 50):  # Configurable iterations
            hard = (iter_idx > (
                self.args.mask_opt_soft_iterations if hasattr(self.args, 'mask_opt_soft_iterations') else 5))
            temperature = self.adv_masking.temperature if not hard else self.adv_masking.temperature * 0.5
            current_mask_tensor = self.adv_masking.gumbel_softmax(mask_logits, temperature=temperature, hard=hard)

            # Apply mask, then fill
            data_after_masking = self.adv_masking.apply_mask(batch_x_orig.clone(),
                                                             current_mask_tensor)  # Apply to a clone
            data_after_filling = self.adv_masking.fill_with_method(batch_x_orig.clone(), current_mask_tensor,
                                                                   method=filling_method_for_attack)

            outputs = self.model(data_after_filling, padding_mask, None, None)
            classification_loss = criterion_mask_gen(outputs, label.long().squeeze(-1))
            sparsity_penalty = self.adv_masking.calculate_sparsity_penalty(current_mask_tensor)

            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probs, dim=1)
                current_accuracy_val = (predictions == label.squeeze(-1)).float().mean().item()
                accuracy_diff = abs(current_accuracy_val - target_accuracy_for_mask)

                if accuracy_diff < best_accuracy_diff:
                    best_accuracy_diff = accuracy_diff
                    best_mask_tensor = current_mask_tensor.detach().clone()
                    achieved_accuracy_at_best_mask = current_accuracy_val

                if accuracy_diff < self.args.performance_threshold and hard:  # Stop if close enough and hard mask
                    print(
                        f"  AdvMask Gen: Iter {iter_idx}, Acc {current_accuracy_val:.4f} (Target {target_accuracy_for_mask:.4f}). Reached target.")
                    final_mask_for_batch = best_mask_tensor
                    break

            loss_sign = 1.0 if current_accuracy_val > target_accuracy_for_mask else -1.0
            # We want to MINIMIZE this objective: (sign * classification_loss) to drive accuracy towards target
            # If current_accuracy > target, loss_sign is 1.0. We want to MAXIMIZE classification_loss (i.e., MINIMIZE -classification_loss)
            # But optimizer minimizes. So if current_acc > target, we want to make model predict worse.
            # So loss for optimizer should be -classification_loss if current_acc > target_acc
            # and classification_loss if current_acc < target_acc
            # This is equivalent to: loss_sign * (-classification_loss) to make it a minimization problem for Adam.
            # Wait, the original logic was: if acc > target, increase loss. if acc < target, decrease loss.
            # Optimizer minimizes loss.
            # If current_accuracy > target_accuracy: we want to *increase* the model's error, so the mask should make the loss higher.
            #    So the mask's "loss" is -classification_loss (we want to maximize classification_loss).
            # If current_accuracy < target_accuracy: we want to *decrease* the model's error (make mask less harmful).
            #    So the mask's "loss" is classification_loss.
            # This is correct: loss_sign = 1.0 if current_accuracy > target_accuracy else -1.0
            # Then the loss for the mask optimizer is: loss_sign * classification_loss + sparsity_penalty
            # This means if acc > target, optimizer tries to find mask that increases class_loss. Correct.
            # If acc < target, optimizer tries to find mask that decreases class_loss. Correct.

            adv_loss = loss_sign * classification_loss + sparsity_penalty

            mask_optimizer.zero_grad()
            adv_loss.backward()
            mask_optimizer.step()

            with torch.no_grad():
                mask_logits.data = self.adv_masking.project_mask(mask_logits.data)

            if iter_idx % 10 == 0 or iter_idx == (
            self.args.mask_opt_iterations if hasattr(self.args, 'mask_opt_iterations') else 50) - 1:
                zero_ratio_val = 1.0 - current_mask_tensor.mean().item()
                print(
                    f"  AdvMask Gen: Iter {iter_idx}, Acc: {current_accuracy_val:.4f}, Target: {target_accuracy_for_mask:.4f}, ZeroRatio: {zero_ratio_val:.4f}, SparsityPen: {sparsity_penalty:.4f}, ClassLoss: {classification_loss:.4f}")

        # Restore model grads
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()  # Set model back to train mode if it was for training overall

        final_mask_for_batch = best_mask_tensor if best_mask_tensor is not None else current_mask_tensor.detach()
        zero_ratio_final = 1.0 - final_mask_for_batch.mean().item()

        mask_stats = {
            'zero_ratio': zero_ratio_final,
            'mask': final_mask_for_batch.cpu().numpy(),  # Store as numpy
            'achieved_accuracy': achieved_accuracy_at_best_mask,
            'target_accuracy': target_accuracy_for_mask,
            'filling_method_at_generation': filling_method_for_attack
        }

        # Return the data that has been masked and then filled using the specified method for this attack generation
        data_final_filled = self.adv_masking.fill_with_method(batch_x_orig.clone(), final_mask_for_batch,
                                                              method=filling_method_for_attack)

        return data_final_filled, final_mask_for_batch, mask_stats

    def _evaluate(self, data_set, data_loader, criterion,
                  apply_mask_config=None,  # Dict: {'type': 'random'/'adversarial', 'filling_method': 'zero', ...}
                  return_original_and_masks=False):
        """ Unified evaluation method.
            apply_mask_config:
                None: No mask applied (original performance).
                {'type': 'random', 'filling_method': str, 'max_missing': float}: Applies random mask then fills.
                {'type': 'adversarial', 'filling_method_for_attack': str, 'filling_method_for_eval': str}:
                    Generates adversarial mask (using filling_method_for_attack),
                    then applies it to original data and fills with filling_method_for_eval for model input.
        """
        self.model.eval()
        total_loss_val = []
        all_preds = []
        all_trues = []

        generated_masks_info = []  # List of dicts
        original_data_batches = []  # List of dicts {'batch_x': np, 'label': np, 'batch_idx': int}

        # Determine if we need to load the original model for adversarial mask generation
        # This should be done outside the loop if generating adversarial masks for multiple batches
        original_model_path = os.path.join(self.args.checkpoints, 'original_model.pth')
        if apply_mask_config and apply_mask_config['type'] == 'adversarial':
            if not os.path.exists(original_model_path):
                print(f"错误: 评估对抗性掩码需要预训练的原始模型 '{original_model_path}'。请先训练并保存。")
                # Fallback to random or raise error
                # For now, let's try to proceed but results might be inconsistent if model state is not 'original'
                # A better approach is to ensure 'original_model.pth' is always available for this path.
                # self._get_original_performance(force_recalc=True) # Try to create it
                print("警告: 原始模型未找到，对抗性掩码生成可能使用当前模型状态。")
            else:
                # Create a temporary model instance or load state for mask generation to avoid interference
                # For simplicity, assume self.model can be temporarily set to original weights
                # A cleaner way: temp_model = self._build_model(); temp_model.load_state_dict(...)
                # For now: current_model_state = self.model.state_dict()
                # self.model.load_state_dict(torch.load(original_model_path, map_location=self.device))
                # print("  _evaluate: Loaded original model for adversarial mask generation.")
                pass  # Mask generation function will handle model state

        with torch.no_grad():  # No gradients during evaluation itself
            for i, (batch_x_orig, label, padding_mask) in enumerate(data_loader):
                batch_x_orig = batch_x_orig.float().to(self.device)
                label = label.to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                if return_original_and_masks:
                    original_data_batches.append({
                        'batch_x': batch_x_orig.cpu().numpy(),
                        'label': label.cpu().numpy(),
                        'batch_idx': i
                    })

                current_batch_for_model = batch_x_orig.clone()
                batch_mask_info = {'batch_idx': i, 'epoch': 0}  # epoch 0 for test/vali

                if apply_mask_config:
                    mask_type = apply_mask_config['type']

                    if mask_type == 'random':
                        filling_method_eval = apply_mask_config['filling_method']
                        max_missing_eval = apply_mask_config.get('max_missing', self.args.max_missing)

                        bs, sl, fd = batch_x_orig.shape
                        # Generate random mask
                        # Ensure mask is generated with the correct max_missing for this evaluation
                        # The self.adv_masking.mask_ratio might be from general args.
                        # It's better to pass max_missing directly or set it on adv_masking if it's fixed per call.
                        # For simplicity, assume self.args.max_missing is the one to use if not specified.
                        # temp_mask_ratio = self.adv_masking.mask_ratio
                        # self.adv_masking.mask_ratio = max_missing_eval
                        # This is problematic if mask_ratio is used elsewhere.
                        # A better way for random mask:
                        random_mask_tensor = (torch.rand(bs, sl, fd, device=self.device) > max_missing_eval).float()
                        # self.adv_masking.mask_ratio = temp_mask_ratio # Restore

                        current_batch_for_model = self.adv_masking.fill_with_method(
                            batch_x_orig.clone(), random_mask_tensor, method=filling_method_eval
                        )
                        batch_mask_info.update({
                            'mask': random_mask_tensor.cpu().numpy(),
                            'zero_ratio': 1.0 - random_mask_tensor.mean().item(),
                            'mask_type': 'random_eval',
                            'evaluated_filling_method': filling_method_eval,
                            'max_missing_setting': max_missing_eval
                        })

                    elif mask_type == 'adversarial':
                        # This part uses the original model to generate the mask
                        # The mask is generated to attack the 'original_model.pth'
                        # The filling method used DURING MASK GENERATION
                        filling_method_for_attack = apply_mask_config['filling_method_for_attack']
                        # The filling method used TO PREPARE DATA FOR THE CURRENT MODEL (after mask is applied)
                        filling_method_for_eval = apply_mask_config['filling_method_for_eval']

                        # Load original model state for mask generation
                        # current_model_state_dict_eval = self.model.state_dict() # Save current state
                        # try:
                        #     self.model.load_state_dict(torch.load(original_model_path, map_location=self.device))
                        # except Exception as e:
                        #     print(f"  Error loading original model for adv mask gen in _evaluate: {e}. Using current model state.")

                        # Generate adversarial mask for this batch_x_orig
                        # The _generate_adversarial_mask_batch itself will put model in eval and freeze params
                        _, adv_mask_tensor, adv_mask_stats_dict = \
                            self._generate_adversarial_mask_batch(batch_x_orig.clone(), label, padding_mask,
                                                                  filling_method_for_attack)
                        # Restore current model state if it was changed
                        # self.model.load_state_dict(current_model_state_dict_eval)
                        # self.model.eval() # Ensure eval mode for this part of _evaluate

                        # Now, apply this generated adv_mask_tensor to the original batch_x_orig
                        # And then fill it using filling_method_for_eval for the *current* model's input
                        current_batch_for_model = self.adv_masking.fill_with_method(
                            batch_x_orig.clone(), adv_mask_tensor, method=filling_method_for_eval
                        )
                        batch_mask_info.update({
                            'mask': adv_mask_tensor.cpu().numpy(),  # Already numpy from adv_mask_stats_dict['mask']
                            'zero_ratio': adv_mask_stats_dict['zero_ratio'],
                            'mask_type': 'adversarial_eval',
                            'achieved_accuracy_during_gen': adv_mask_stats_dict['achieved_accuracy'],
                            'target_accuracy_for_gen': adv_mask_stats_dict['target_accuracy'],
                            'filling_method_at_generation': adv_mask_stats_dict['filling_method_at_generation'],
                            'evaluated_filling_method': filling_method_for_eval
                        })

                    if 'mask' in batch_mask_info:  # Only append if a mask was actually generated and added
                        generated_masks_info.append(batch_mask_info)

                # ----- Model Prediction -----
                outputs = self.model(current_batch_for_model, padding_mask, None, None)
                loss_val = criterion(outputs, label.long().squeeze(-1))
                total_loss_val.append(loss_val.item())
                all_preds.append(outputs.detach())
                all_trues.append(label)

        # Restore original model state if it was temporarily changed for adv mask generation within the loop
        # (Better to handle this inside _generate_adversarial_mask_batch or pass a dedicated model instance)
        # if apply_mask_config and apply_mask_config['type'] == 'adversarial' and 'current_model_state' in locals():
        #    self.model.load_state_dict(current_model_state) # Restore main model state
        #    print("  _evaluate: Restored main model state after adversarial mask generation.")

        self.model.train()  # Set back to train mode after evaluation

        if not all_preds:
            print("警告: _evaluate did not produce any predictions.")
            if return_original_and_masks:
                return 0.0, 0.0, [], []
            return 0.0, 0.0

        avg_loss = np.average(total_loss_val)
        all_preds = torch.cat(all_preds, 0)
        all_trues = torch.cat(all_trues, 0)

        probs = torch.nn.functional.softmax(all_preds, dim=1)
        predictions_np = torch.argmax(probs, dim=1).cpu().numpy()
        trues_np = all_trues.flatten().cpu().numpy()
        accuracy_val = cal_accuracy(predictions_np, trues_np)

        if return_original_and_masks:
            return avg_loss, accuracy_val, generated_masks_info, original_data_batches
        return avg_loss, accuracy_val

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        # Use a distinct validation set if available, otherwise use test set for validation
        vali_data, vali_loader = self._get_data(
            flag='VAL' if 'VAL' in self.args.data_flags else 'TEST')  # Assuming data_flags in args

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path): os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # Get original performance using the clean, original model
        # This also ensures 'original_model.pth' is created/loaded if needed for adv mask generation
        self._get_original_performance()  # This will be the baseline
        if self.args.use_adversarial_mask:
            self._calculate_target_accuracy()  # Calculate target based on original_accuracy

        # Store masks and original data from the epoch that performs best on validation, or last epoch
        # For simplicity, we'll save from the last epoch if early stopping doesn't occur earlier.
        # A more robust way is to track the best epoch and save corresponding masks.
        # For now, we collect masks per epoch and decide at the end.
        all_epochs_masks_info = []  # List of lists (outer for epoch, inner for batch masks)
        all_epochs_original_data = []

        for epoch in range(self.args.train_epochs):
            epoch_train_loss = []
            epoch_masks_info_train = []  # Masks generated during this training epoch
            epoch_original_data_train = []  # Original data for this training epoch

            self.model.train()  # Ensure model is in training mode
            epoch_time_start = time.time()

            for i, (batch_x_orig, label, padding_mask) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x_orig = batch_x_orig.float().to(self.device)  # This is the clean data
                label = label.to(self.device)
                padding_mask = padding_mask.float().to(self.device)

                # Store original data for this batch (before any masking)
                current_original_batch_data = {
                    'batch_x': batch_x_orig.cpu().numpy(),
                    'label': label.cpu().numpy(),
                    'batch_idx': i
                }
                epoch_original_data_train.append(current_original_batch_data)

                batch_x_for_model = batch_x_orig.clone()  # Start with original data for this batch

                if self.args.use_adversarial_mask:
                    # Generate adversarial mask and get the data_for_model (masked and filled)
                    # The filling method here is self.args.filling_method (the one used with the attack)
                    # Pass the original batch_x_orig for mask generation
                    batch_x_for_model, _, adv_mask_stats = \
                        self._generate_adversarial_mask_batch(batch_x_orig, label, padding_mask,
                                                              filling_method_for_attack=self.args.filling_method)

                    adv_mask_stats.update({'batch_idx': i, 'epoch': epoch})
                    epoch_masks_info_train.append(adv_mask_stats)

                elif self.args.use_random_mask:
                    bs, sl, fd = batch_x_orig.shape
                    random_mask_tensor = (torch.rand(bs, sl, fd, device=self.device) > self.args.max_missing).float()
                    batch_x_for_model = self.adv_masking.fill_with_method(
                        batch_x_orig.clone(), random_mask_tensor, method=self.args.filling_method
                    )
                    epoch_masks_info_train.append({
                        'mask': random_mask_tensor.cpu().numpy(),
                        'zero_ratio': 1.0 - random_mask_tensor.mean().item(),
                        'batch_idx': i, 'epoch': epoch, 'mask_type': 'random_train',
                        'filling_method_at_generation': self.args.filling_method
                    })

                # --- Model Training Step ---
                # batch_x_for_model is now either original, adversarially perturbed, or randomly masked
                outputs = self.model(batch_x_for_model, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                epoch_train_loss.append(loss.item())

                loss.backward()
                # Consider gradient clipping if you face instability, e.g., nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()

                if (i + 1) % 100 == 0:  # Logging
                    current_lr = model_optim.param_groups[0]['lr']
                    print(
                        f"Epoch: {epoch + 1}/{self.args.train_epochs} | Batch: {i + 1}/{train_steps} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

            # --- End of Epoch ---
            avg_epoch_train_loss = np.average(epoch_train_loss) if epoch_train_loss else 0

            # Validation
            # During validation, typically no adversarial/random masks are applied unless specifically testing robustness
            # Use self.args.apply_mask_in_validation to control this.
            # For standard validation to check model convergence, apply_mask_config=None
            vali_mask_config = None
            if self.args.apply_mask_in_validation:  # e.g. validate with random masks
                vali_mask_config = {'type': 'random', 'filling_method': self.args.filling_method,
                                    'max_missing': self.args.max_missing}

            vali_loss, vali_accuracy = self._evaluate(vali_data, vali_loader, criterion,
                                                      apply_mask_config=vali_mask_config)

            print(
                f"Epoch: {epoch + 1} | Train Loss: {avg_epoch_train_loss:.4f} | Vali Loss: {vali_loss:.4f} | Vali Acc: {vali_accuracy:.4f} | Time: {time.time() - epoch_time_start:.2f}s")

            all_epochs_masks_info.append(epoch_masks_info_train)
            all_epochs_original_data.append(epoch_original_data_train)

            early_stopping(-vali_accuracy, self.model, path)  # Stop based on validation accuracy
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            if hasattr(self.args, 'lr_adj') and self.args.lr_adj != 'none':
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        print("Training finished.")
        best_model_path = os.path.join(path, 'checkpoint.pth')
        print(f"Loading best model from: {best_model_path}")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        # Save masks from the best epoch (or last epoch if no early stopping / specific logic)
        # For simplicity, saving masks from the epoch of the 'checkpoint.pth'
        # EarlyStopping saves the model when val_score improves. Need to know which epoch that was.
        # If early_stopping.best_score is available, and you log epoch number with it.
        # For now, let's assume the last epoch before stopping (or self.args.train_epochs-1)
        # is the one whose masks we want to save if use_adversarial_mask or use_random_mask was on.

        # Find the epoch that was saved by early stopping
        # early_stopping.counter tells how many epochs since last improvement.
        # So, best epoch was current_epoch - early_stopping.counter
        # However, masks are collected per epoch.
        # If using early stopping, it saves the *model*. We need to save the *masks* from that epoch.
        # A robust way: store masks with epoch number, then retrieve masks for `early_stopping.best_epoch_num`

        # Simplified: Save masks from the epoch that corresponds to the best_model_path.
        # This requires knowing which epoch generated 'checkpoint.pth'.
        # If early_stopping.best_epoch is an attribute storing the epoch number:
        # best_epoch_idx = early_stopping.best_epoch if hasattr(early_stopping, 'best_epoch') else epoch
        best_epoch_idx = epoch - early_stopping.counter if early_stopping.early_stop else epoch

        if best_epoch_idx < len(all_epochs_masks_info) and (
                self.args.use_adversarial_mask or self.args.use_random_mask):
            masks_to_save = all_epochs_masks_info[best_epoch_idx]
            original_data_for_masks = all_epochs_original_data[best_epoch_idx]
            mask_type_str = "adversarial_train" if self.args.use_adversarial_mask else "random_train"
            self.save_masks(masks_to_save, setting + "_train_masks", original_data_for_masks, mask_type=mask_type_str)
        else:
            print("No training masks to save or best epoch index out of bounds.")

        return self.model

    def test(self, setting, test_run_num=0):  # test_run_num to make setting unique if run multiple times
        test_data, test_loader = self._get_data(flag='TEST')

        # Load the best model saved during training
        model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(model_path):
            print(f'Loading best trained model for testing from: {model_path}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(
                f"警告: No trained model found at {model_path}. Testing with current model state (might be uninitialized or from a previous run).")
            # Attempt to load the 'original_model.pth' as a fallback if the specific setting checkpoint is missing.
            original_model_path_fallback = os.path.join(self.args.checkpoints, 'original_model.pth')
            if os.path.exists(original_model_path_fallback):
                print(f"  Fallback: Loading 'original_model.pth' for testing.")
                self.model.load_state_dict(torch.load(original_model_path_fallback, map_location=self.device))
            else:
                print("  Fallback 'original_model.pth' also not found. Testing with potentially untrained model.")

        criterion = self._select_criterion()

        # 1. Evaluate on clean data (Original Performance with the final trained model)
        # This is important to see if the model learned well.
        # The self.original_accuracy was based on the 'original_model.pth' (potentially a different base model).
        # We need the performance of *this setting's trained model* on clean data.
        print("评估已训练模型在干净测试数据上的性能...")
        clean_loss, clean_accuracy = self._evaluate(test_data, test_loader, criterion, apply_mask_config=None)
        print(f"已训练模型在干净测试数据上的准确率: {clean_accuracy:.4f}")

        results_log = {
            'setting': setting,
            'trained_model_clean_accuracy': clean_accuracy,
            'original_model_base_accuracy': self.original_accuracy if self.original_accuracy is not None else "N/A",
            # From _get_original_performance
            'tests': []
        }

        # 2. Evaluate with different mask types and filling methods
        # Mask types to test: 'random', 'adversarial' (if original_model.pth exists)
        # Filling methods to test: ['zero', 'mean', 'knn', 'interpolation']

        filling_methods_to_test = ['zero', 'mean', 'knn', 'interpolation']
        test_mask_configurations = []

        # A. Random Mask Configurations
        for fill_m in filling_methods_to_test:
            test_mask_configurations.append({
                'eval_name': f'RandomMask_Fill-{fill_m}',
                'mask_config': {'type': 'random', 'filling_method': fill_m, 'max_missing': self.args.max_missing},
                'save_mask_type': f'random_test_fill_{fill_m}'
            })

        # B. Adversarial Mask Configurations (Requires original_model.pth to be robustly available)
        # The adversarial mask is generated to attack 'original_model.pth'.
        # We then see how the *current trained model* performs against this mask when data is filled using various methods.
        original_model_path = os.path.join(self.args.checkpoints, 'original_model.pth')
        if os.path.exists(original_model_path):  # Only add adversarial tests if original model is present
            # Adversarial masks can be generated with a specific filling method in mind during attack.
            # Let's assume we generate adversarial masks using 'zero' fill during attack,
            # and then evaluate the current model with various filling methods.
            # Or, generate masks for each filling method used in attack. Simpler: one attack fill method.
            adv_attack_fill_method = self.args.filling_method  # Use the same filling method as in training for attack gen.

            for eval_fill_m in filling_methods_to_test:
                test_mask_configurations.append({
                    'eval_name': f'AdvMask(atk_fill:{adv_attack_fill_method})_EvalFill-{eval_fill_m}',
                    'mask_config': {
                        'type': 'adversarial',
                        'filling_method_for_attack': adv_attack_fill_method,  # Method used when optimizing the mask
                        'filling_method_for_eval': eval_fill_m  # Method used to fill for this specific test
                    },
                    'save_mask_type': f'adversarial_test_atkfill_{adv_attack_fill_method}_evalfill_{eval_fill_m}'
                })
        else:
            print("警告: 'original_model.pth' 未找到, 跳过对抗性掩码测试评估。")

        # --- Execute Test Evaluations ---
        for config in test_mask_configurations:
            print(f"\n--- 测试: {config['eval_name']} ---")

            # The _evaluate function now returns masks and original data if requested
            loss, acc, masks_generated_during_eval, corresponding_original_data = \
                self._evaluate(test_data, test_loader, criterion,
                               apply_mask_config=config['mask_config'],
                               return_original_and_masks=True)  # Crucial to get masks and data

            print(
                f"  准确率 ({config['eval_name']}): {acc:.4f}, 下降率 (vs trained clean): {(clean_accuracy - acc) / clean_accuracy:.2% if clean_accuracy > 0 else 0.0:.2%}")

            results_log['tests'].append({
                'eval_name': config['eval_name'],
                'accuracy': acc,
                'loss': loss,
                'performance_drop_vs_trained_clean': (
                                                                 clean_accuracy - acc) / clean_accuracy if clean_accuracy > 0 else 0.0,
                'masks_saved_to': f"{setting}_test_masks/masks_{config['save_mask_type']}/"
            })

            # Save the masks generated during this specific evaluation run
            if masks_generated_during_eval:
                # Pass the config['save_mask_type'] to uniquely identify these masks
                self.save_masks(masks_generated_during_eval, setting + "_test_masks",
                                corresponding_original_data, mask_type=config['save_mask_type'])
            else:
                print(f"  没有为 {config['eval_name']} 生成或收集到掩码。")

        # --- Save Results Log ---
        results_folder = f'./results/{self.args.model}_{self.args.data_path.split("/")[-1].split(".")[0]}/{setting}/'
        if not os.path.exists(results_folder): os.makedirs(results_folder)

        # Simple text log
        log_file_path = os.path.join(results_folder, f'test_results_log_{test_run_num}.txt')
        with open(log_file_path, 'w') as f:
            f.write(f"Test Results for Setting: {setting}\n")
            f.write(f"Trained Model Clean Accuracy (Test Set): {results_log['trained_model_clean_accuracy']:.4f}\n")
            f.write(f"Original Model Base Accuracy (Test Set): {results_log['original_model_base_accuracy']}\n\n")
            for test_entry in results_log['tests']:
                f.write(f"Evaluation: {test_entry['eval_name']}\n")
                f.write(f"  Accuracy: {test_entry['accuracy']:.4f}\n")
                f.write(f"  Loss: {test_entry['loss']:.4f}\n")
                f.write(f"  Drop vs Trained Clean: {test_entry['performance_drop_vs_trained_clean']:.2%}\n")
                f.write(f"  Masks saved to path ending with: .../{test_entry['masks_saved_to']}\n\n")
        print(f"测试结果已记录到: {log_file_path}")

        # Detailed JSON log (optional, but good for parsing)
        import json
        json_log_path = os.path.join(results_folder, f'test_results_detailed_{test_run_num}.json')
        with open(json_log_path, 'w') as f_json:
            json.dump(results_log, f_json, indent=4)
        print(f"详细测试结果 (JSON) 已保存到: {json_log_path}")

        return results_log


if __name__ == '__main__':
    # This is a placeholder for how you might run your experiment.
    # You'll need to set up your `args` object similar to how your main script does.

    class ArgsSimulator:  # Simulate the args object
        def __init__(self):
            # --- Essential paths and model/data args ---
            self.model = 'TimesNet'  # Or your specific model
            self.data_path = 'your_dataset.csv'  # e.g., 'ETTm1.csv'
            self.checkpoints = './checkpoints_adv_sim/'
            self.root_path = './dataset/'  # Path to your data directory
            self.data = 'custom'  # Or 'ETTh1', 'ETTm1', etc. if using predefined datasets

            # --- General training args ---
            self.train_epochs = 1  # Keep low for quick test
            self.batch_size = 16
            self.patience = 3
            self.learning_rate = 0.001
            self.lr_adj = 'type1'  # Learning rate adjustment strategy
            self.use_gpu = torch.cuda.is_available()
            self.gpu = 0
            self.device = torch.device('cuda:0') if self.use_gpu else torch.device('cpu')
            self.use_multi_gpu = False
            self.device_ids = [0]

            # --- TimesNet specific (or your model's specific) args ---
            self.num_kernels = 6  # Example for TimesNet
            self.top_k = 5  # Example for TimesNet
            self.d_model = 32  # Example for TimesNet
            self.d_ff = 32  # Example for TimesNet
            self.e_layers = 2  # Example for TimesNet
            self.dropout = 0.1  # Example for TimesNet
            # Add other model-specific args as needed by your TimesNet or other model implementation

            # --- Adversarial Masking and Filling Args ---
            self.max_missing = 0.3  # Max missing ratio for random masks and sparsity target for adv masks
            self.tau = 1.0  # Gumbel-softmax temperature
            self.lambda_sparsity = 1.0  # Sparsity penalty coefficient for adversarial mask generation

            self.target_performance_drop = 0.2  # Target 20% drop for adversarial masks
            self.performance_threshold = 0.02  # +/- 2% around target accuracy is acceptable for adv mask
            self.mask_learning_rate = 0.01  # LR for optimizing mask logits
            self.mask_opt_iterations = 30  # Iterations for optimizing mask per batch
            self.mask_opt_soft_iterations = 3  # Iterations with soft Gumbel before hard

            # Control training phase
            self.use_adversarial_mask = True  # True to train with adversarial masks
            self.use_random_mask = False  # True to train with random (if not adversarial)

            # Filling method used *during training* if masks are applied
            # Also used as the 'filling_method_for_attack' when generating adv masks in test phase
            self.filling_method = 'mean'

            # Control validation/testing phase masking (for _evaluate calls within train loop for vali)
            self.apply_mask_in_validation = False  # False: validate on clean data during training epochs.
            # True: validate with random masks during training epochs.

            # For data_provider, ensure these are set if your provider uses them
            self.features = 'M'  # M: multivariate, S: univariate, MS: mixed
            self.target = 'OT'  # Target feature for some datasets, not always used in classification
            self.freq = 'h'  # data frequency (e.g., h for hourly) - for Informer family, may not be used by TimesNet directly for classification

            # This is a flag for the data_provider, adjust as per your setup
            self.data_flags = ['TRAIN', 'TEST', 'VAL']  # Or whatever flags your data_provider expects

            # Ensure model_dict is populated correctly elsewhere or provide a mock one here if Exp_Basic needs it
            # self.model_dict = {'TimesNet': TimesNetClassPlaceholder}


    print("Simulating experiment run...")
    args = ArgsSimulator()


    # You need to have your model definition (e.g., TimesNet) available.
    # For this simulation, we'll assume it's imported or defined.
    # from models.TimesNet import Model as TimesNetModel # Example import
    # args.model_dict = { 'TimesNet': TimesNetModel }

    # --- Mock TimesNet Model for placeholder --
    class MockTimesNet(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.seq_len = config.seq_len
            self.enc_in = config.enc_in
            self.num_class = config.num_class
            # A very simple linear layer for mock classification
            self.projection = nn.Linear(self.seq_len * self.enc_in, self.num_class)
            print(f"MockTimesNet initialized: seq_len={self.seq_len}, enc_in={self.enc_in}, num_class={self.num_class}")

        def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
            # x_enc shape: (batch_size, seq_len, features)
            batch_size = x_enc.size(0)
            x_flat = x_enc.reshape(batch_size, -1)  # Flatten
            output = self.projection(x_flat)  # (batch_size, num_class)
            return output


    if not hasattr(args, 'model_dict'):
        args.model_dict = {'TimesNet': MockTimesNet}  # Use mock if real not available
    # --- End Mock ---

    # Initialize and run experiment
    exp = Exp_AdversarialClassification(args)

    setting_name = f"{args.model}_{args.data_path.split('/')[-1].split('.')[0]}_advTrainFill-{args.filling_method}_drop-{args.target_performance_drop}_miss-{args.max_missing}"
    if args.use_adversarial_mask:
        print(f"\n--- Training with Adversarial Masks (Setting: {setting_name}) ---")
        exp.train(setting_name)
        print(f"\n--- Testing (Setting: {setting_name}) ---")
        exp.test(setting_name, test_run_num=0)

    # Example: Train with random masks if not adversarial
    if not args.use_adversarial_mask and args.use_random_mask:
        args.use_adversarial_mask = False  # Ensure it's off
        args.use_random_mask = True
        setting_name_rand = f"{args.model}_{args.data_path.split('/')[-1].split('.')[0]}_randTrainFill-{args.filling_method}_miss-{args.max_missing}"
        print(f"\n--- Training with Random Masks (Setting: {setting_name_rand}) ---")
        exp.train(setting_name_rand)
        print(f"\n--- Testing (Setting: {setting_name_rand}) ---")
        exp.test(setting_name_rand, test_run_num=1)

    # Example: Train on clean data (no masks during training)
    args.use_adversarial_mask = False
    args.use_random_mask = False
    setting_name_clean = f"{args.model}_{args.data_path.split('/')[-1].split('.')[0]}_cleanTrain"
    print(f"\n--- Training on Clean Data (Setting: {setting_name_clean}) ---")
    # Re-initialize optimizer if needed, or ensure train resets it.
    # exp.model_optim = exp._select_optimizer() # If optimizer state needs reset
    exp.train(setting_name_clean)  # Train on clean
    print(f"\n--- Testing (Setting: {setting_name_clean}) ---")
    # Test this clean-trained model against random and adversarial masks
    exp.test(setting_name_clean, test_run_num=2)

    print("\n--- Adversarial Classification Experiment Simulation Complete ---")

    # Now, you would run the analysis script provided in the next section
    # on the generated ./results/ directory.