import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from aeon.classification.convolution_based import RocketClassifier
from aeon.datasets import load_classification
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class TSDataProcessor:
    @staticmethod
    def load_dataset(dataset_name, data_path):
        try:
            X_train, y_train = load_classification(dataset_name, split="train", extract_path=data_path)
            X_test, y_test = load_classification(dataset_name, split="test", extract_path=data_path)

            # 标签编码
            le = LabelEncoder()
            y_train = le.fit_transform(y_train.ravel())  # 展平并编码
            y_test = le.transform(y_test.ravel())

            # 统一维度为 [samples, channels, time]
            if X_train.ndim == 2:
                X_train = X_train.reshape(-1, 1, X_train.shape[1])
                X_test = X_test.reshape(-1, 1, X_test.shape[1])

            # 缺失值处理
            X_train = TSDataProcessor.handle_missing_values(X_train)
            X_test = TSDataProcessor.handle_missing_values(X_test)

            return X_train, y_train, X_test, y_test
        except Exception as e:
            print(f"Error loading {dataset_name}: {str(e)}")
            return None, None, None, None

    @staticmethod
    def handle_missing_values(data):
        data = torch.tensor(data).float()
        nan_mask = torch.isnan(data)

        for t in range(1, data.size(-1)):
            data[:, :, t] = torch.where(nan_mask[:, :, t], data[:, :, t - 1], data[:, :, t])

        for t in range(data.size(-2) - 2, -1, -1):
            data[:, :, t] = torch.where(nan_mask[:, :, t], data[:, :, t + 1], data[:, :, t])

        global_mean = torch.nanmean(data)
        return torch.where(nan_mask, global_mean, data).numpy()


class AdversarialAttacker:
    class ProbabilisticRocket(RocketClassifier):
        def __init__(self, n_jobs=-1):
            super().__init__(
                n_jobs=n_jobs,
                estimator=LogisticRegression(
                    multi_class="multinomial",
                    max_iter=1000,
                    solver="lbfgs"
                )
            )

        def predict_proba(self, X):
            """智能处理输入输出类型"""
            # 转换输入为 NumPy 并保持三维结构
            if isinstance(X, torch.Tensor):
                X_np = X.detach().cpu().numpy()
                if X_np.ndim == 2:
                    X_np = X_np.reshape(-1, 1, X_np.shape[1])
            else:
                X_np = np.asarray(X)
                if X_np.ndim == 2:
                    X_np = X_np.reshape(-1, 1, X_np.shape[1])

            # 获取概率并转换为可微分张量
            proba = super().predict_proba(X_np)
            return torch.tensor(proba,
                                device=X.device if isinstance(X, torch.Tensor) else 'cpu',
                                requires_grad=True)

        def torch_predict(self, X_tensor):
            """专用方法处理张量输入"""
            with torch.no_grad():
                X_np = X_tensor.detach().cpu().numpy()
                if X_np.ndim == 2:
                    X_np = X_np.reshape(-1, 1, X_np.shape[1])
                self.fit(X_np, y_train)  # 假设 y_train 已定义
            return self.predict_proba(X_tensor)

    def __init__(self, target_f1_ratio=0.3, lr=0.05, eta=0.2, max_epochs=100):
        self.target_ratio = target_f1_ratio
        self.lr = lr
        self.eta = eta
        self.max_epochs = max_epochs
        self.model = self.ProbabilisticRocket(n_jobs=-1)
        self.results = pd.DataFrame(columns=['Dataset', 'Original_F1', 'Attacked_F1', 'Delta'])

    class MaskGenerator(nn.Module):
        def __init__(self, seq_len, n_channels, temp=0.5):
            super().__init__()
            self.temp = temp
            self.logits = nn.Parameter(torch.randn(seq_len, n_channels))

        def forward(self, batch_size):
            # Gumbel-Softmax 采样
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits)))
            logits_with_noise = (self.logits + gumbel_noise) / self.temp
            probs = torch.sigmoid(logits_with_noise)

            # 伯努利采样 + 直通估计
            with torch.no_grad():
                mask_hard = torch.bernoulli(probs)  # 前向传播用采样值
            mask_soft = probs  # 反向传播用概率值

            # 梯度桥接
            mask = mask_hard - mask_soft.detach() + mask_soft

            return mask.unsqueeze(0).expand(batch_size, -1, -1)

    def attack_dataset(self, dataset_name, X_train, y_train, save_path):
        X_tensor = torch.tensor(X_train).float()
        y_tensor = torch.tensor(y_train).long()
        seq_len, n_channels = X_tensor.shape[1], X_tensor.shape[2]

        mask_gen = self.MaskGenerator(seq_len, n_channels)
        optimizer = torch.optim.AdamW([mask_gen.logits], lr=self.lr, weight_decay=1e-4)
        # 初始化关键变量
        X_filled = None
        best_mask = None
        best_ce = 0

        # 基准评估
        with torch.no_grad():
            # 转换为numpy数组进行模型训练
            # self.model.fit(
            #     X_filled.detach().cpu().numpy(),  # 必须转换为numpy
            #     y_train
            # )
            self.model.fit(X_train, y_train)
            orig_probs = self.model.predict_proba(X_train)
            orig_ce = F.cross_entropy(orig_probs, y_tensor).item()
            target_ce = orig_ce * (1 + self.target_ratio)

        progress_bar = tqdm(range(self.max_epochs), desc=f"Attacking {dataset_name}")
        # for epoch in progress_bar:
        #     mask_probs = mask_gen(X_tensor.size(0))
        #     mask = torch.bernoulli(mask_probs)
        #
        #     # 应用掩码并保持梯度
        #     X_corrupted = X_tensor.clone()
        #     X_corrupted[mask.bool()] = torch.nan
        #     X_filled = self._fill_missing(X_corrupted)
        #
        #     # 模型训练
        #     # with torch.no_grad():
        #     #     self.model.fit(X_filled.detach().cpu().numpy(), y_train)
        #     # 训练模型（转换为NumPy）
        #     # with torch.no_grad():
        #     #     X_filled_np = X_filled.detach().cpu().numpy()
        #     #     if X_filled_np.ndim == 2:
        #     #         X_filled_np = X_filled_np.reshape(-1, 1, X_filled_np.shape[1])
        #     #     self.model.fit(X_filled_np, y_train)
        #     # 分离训练过程
        #     with torch.no_grad():
        #         self.model.torch_predict(X_filled)  # 内部处理训练
        for epoch in progress_bar:
            # 生成可微分掩码
            mask = mask_gen(X_tensor.size(0))

            # 应用掩码 (保持梯度)
            X_corrupted = X_tensor * (1 - mask)
            X_filled = self._fill_missing(X_corrupted)  # 确保填充操作可微

            # 计算损失
            logits = self.model.predict_proba(X_filled)
            ce_loss = F.cross_entropy(logits, y_tensor)
            loss = (ce_loss - target_ce).pow(2)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度存在性检查
            if mask_gen.logits.grad is None:
                raise RuntimeError("梯度未正确传播!")

            # 参数更新
            with torch.no_grad():
                mask_gen.logits.data += self.lr * mask_gen.logits.grad
                mask_gen.logits.data = torch.clamp(mask_gen.logits.data, -self.eta, self.eta)
            # 计算可微分损失
            logits = self.model.predict_proba(X_filled)
            ce_loss = F.cross_entropy(logits, y_tensor)
            loss = (ce_loss - target_ce).pow(2)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(mask_gen.parameters(), max_norm=1.0)

            # PGD更新
            with torch.no_grad():
                mask_gen.logits.data += self.lr * mask_gen.logits.grad
                mask_gen.logits.data = torch.clamp(mask_gen.logits.data, -self.eta, self.eta)

            # 监控指标
            with torch.no_grad():
                current_f1 = f1_score(y_train, logits.argmax(dim=1).numpy(), average='macro')

            progress_bar.set_postfix({
                'Current CE': f'{ce_loss.item():.3f}',
                'Target CE': f'{target_ce:.3f}',
                'Best CE': f'{best_ce:.3f}'
            })

            if ce_loss.item() > best_ce:
                best_ce = ce_loss.item()
                best_mask = mask.detach().clone()

            if abs(ce_loss.item() - target_ce) < 0.05 * target_ce:
                break

        # 最终评估
        final_mask = best_mask.numpy()
        X_attacked = X_train * (1 - final_mask)
        X_attacked = self._fill_missing(torch.tensor(X_attacked)).numpy()

        self.model.fit(X_attacked, y_train)
        attacked_preds = self.model.predict(X_attacked)
        attacked_f1 = f1_score(y_train, attacked_preds, average='macro')

        self._save_results(dataset_name, orig_ce, attacked_f1, save_path)
        self._save_mask(final_mask, dataset_name, save_path)

        return final_mask

    def _fill_missing(self, data):
        channel_means = torch.nanmean(data, dim=(0, 2), keepdim=True)
        return torch.where(torch.isnan(data), channel_means, data)

    def _save_results(self, dataset_name, orig_f1, attacked_f1, save_path):
        new_row = pd.DataFrame({
            'Dataset': [dataset_name],
            'Original_F1': [orig_f1],
            'Attacked_F1': [attacked_f1],
            'Delta': [orig_f1 - attacked_f1]
        })
        self.results = pd.concat([self.results, new_row], ignore_index=True)
        self.results.to_csv(os.path.join(save_path, 'attack_results.csv'), index=False)

    @staticmethod
    def _save_mask(mask, dataset_name, save_path):
        np.save(os.path.join(save_path, f'{dataset_name}_mask.npy'), mask)


if __name__ == "__main__":
    DATA_PATH = r'E:\72fd7dbf\shiyan\my_work\chosen'
    SAVE_PATH = r'E:\72fd7dbf\shiyan\my_work\save'
    TARGET_F1_RATIO = 0.7

    processor = TSDataProcessor()
    attacker = AdversarialAttacker(target_f1_ratio=TARGET_F1_RATIO)

    for dataset in os.listdir(DATA_PATH):
        dataset_path = os.path.join(DATA_PATH, dataset)
        if os.path.isdir(dataset_path):
            print(f"\nProcessing dataset: {dataset}")
            X_train, y_train, X_test, y_test = processor.load_dataset(dataset, dataset_path)

            if X_train is not None:
                final_mask = attacker.attack_dataset(dataset, X_train, y_train, SAVE_PATH)
                X_test_attacked = X_test * (1 - final_mask[:len(X_test)])
                X_test_filled = attacker._fill_missing(torch.tensor(X_test_attacked)).numpy()
                test_preds = attacker.model.predict(X_test_filled)
                test_f1 = f1_score(y_test, test_preds, average='macro')
                print(f"Test F1 after attack: {test_f1:.4f}")
