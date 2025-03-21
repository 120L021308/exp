import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from aeon.classification.convolution_based import RocketClassifier
from aeon.datasets import load_classification
from tqdm import tqdm
import os


# --------------------------
# 数据预处理模块
# --------------------------
class TSDataProcessor:
    @staticmethod
    def load_dataset(dataset_name, data_path):
        try:
            X_train, y_train = load_classification(dataset_name, split="train", extract_path=data_path)
            X_test, y_test = load_classification(dataset_name, split="test", extract_path=data_path)

            # 标签编码
            le = LabelEncoder()
            y_train = le.fit_transform(y_train.ravel())
            y_test = le.transform(y_test.ravel())

            # 统一维度为 [samples, channels, time]
            if X_train.ndim == 2:
                X_train = X_train.reshape(-1, 1, X_train.shape[1])
                X_test = X_test.reshape(-1, 1, X_test.shape[1])

            return X_train, y_train, X_test, y_test
        except Exception as e:
            print(f"Error loading {dataset_name}: {str(e)}")
            return None, None, None, None


# --------------------------
# 可微分Rocket模块
# --------------------------
class DifferentiableRocket(nn.Module):
    """将预训练的Rocket模型转换为可微分PyTorch模块"""

    def __init__(self, rocket_model):
        super().__init__()
        # 提取卷积核参数
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.tensor(kernel, dtype=torch.float32), requires_grad=False)
            for kernel in rocket_model._get_kernels()
        ])

        # 提取分类器参数
        self.classifier = nn.Linear(
            len(self.kernels) * 2,  # 每个核生成2个特征（max和mean）
            len(rocket_model.classes_)
        )
        self._init_classifier(rocket_model.estimator_)

    def _init_classifier(self, sklearn_model):
        """转换sklearn分类器参数到PyTorch"""
        if hasattr(sklearn_model, 'coef_'):
            self.classifier.weight.data = torch.tensor(sklearn_model.coef_, dtype=torch.float32)
            self.classifier.bias.data = torch.tensor(sklearn_model.intercept_, dtype=torch.float32)
        else:
            nn.init.xavier_normal_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        # 输入x形状: [batch, channels, seq_len]
        features = []
        for kernel in self.kernels:
            # 卷积操作
            conv_out = F.conv1d(x, kernel.unsqueeze(0))

            # 提取特征
            max_val, _ = torch.max(conv_out, dim=-1)
            mean_val = torch.mean(conv_out, dim=-1)
            features.extend([max_val, mean_val])

        # 拼接特征
        features = torch.cat(features, dim=1)
        return self.classifier(features)


# --------------------------
# 对抗攻击模块
# --------------------------
class AdversarialAttacker:
    def __init__(self, rocket_model, lr=0.1, eta=0.3, temp=0.5):
        """
        :param rocket_model: 预训练的RocketClassifier实例
        :param lr: 学习率
        :param eta: 参数更新幅度限制
        :param temp: Gumbel-Softmax温度系数
        """
        self.rocket = DifferentiableRocket(rocket_model)
        self.temp = temp
        self.eta = eta

        # 冻结Rocket参数
        for param in self.rocket.parameters():
            param.requires_grad_(False)

    class MaskGenerator(nn.Module):
        """可微分掩码生成器"""

        def __init__(self, seq_len, n_channels):
            super().__init__()
            self.logits = nn.Parameter(torch.randn(seq_len, n_channels))

        def forward(self, batch_size):
            # Gumbel-Softmax采样
            gumbel = -torch.log(-torch.log(torch.rand_like(self.logits)))
            sampled = (self.logits + gumbel) / self.temp
            probs = torch.sigmoid(sampled)

            # 直通估计器
            hard = (probs > 0.5).float()
            return hard.unsqueeze(0).expand(batch_size, -1, -1) - probs.detach() + probs

    def attack(self, X_train, y_train, epochs=100, target_acc=0.7):
        """
        :param X_train: 训练数据 [n_samples, channels, seq_len]
        :param y_train: 训练标签
        :param epochs: 训练轮次
        :param target_acc: 目标攻击准确率
        """
        # 数据准备
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)

        # 初始化掩码生成器
        mask_gen = self.MaskGenerator(X_tensor.shape[2], X_tensor.shape[1])
        optimizer = torch.optim.Adam([mask_gen.logits], lr=self.lr)

        # 训练循环
        best_mask = None
        best_acc = 1.0
        progress = tqdm(range(epochs), desc="Adversarial Training")

        for epoch in progress:
            # 生成掩码
            mask = mask_gen(X_tensor.shape[0])

            # 应用掩码并填充
            X_masked = X_tensor * (1 - mask)
            X_filled = self._fill_missing(X_masked)

            # 前向传播
            logits = self.rocket(X_filled)
            loss = F.cross_entropy(logits, y_tensor)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度投影
            with torch.no_grad():
                grad = mask_gen.logits.grad
                if grad is None:
                    continue

                # 梯度裁剪
                grad = torch.clamp(grad, -self.eta, self.eta)
                mask_gen.logits.data -= self.lr * grad

            # 评估当前掩码
            with torch.no_grad():
                current_acc = (logits.argmax(1) == y_tensor).float().mean().item()
                if current_acc < best_acc:
                    best_acc = current_acc
                    best_mask = mask.detach().clone()

                progress.set_postfix({
                    "Current Acc": f"{current_acc:.3f}",
                    "Best Acc": f"{best_acc:.3f}"
                })

                if current_acc <= target_acc:
                    print(f"\nEarly stopping at epoch {epoch}, reached accuracy {current_acc:.3f}")
                    break

        return best_mask.numpy()

    def _fill_missing(self, data):
        """可微分缺失值填充"""
        # 计算通道均值
        channel_means = torch.nanmean(data, dim=(0, 2), keepdim=True)
        # 填充缺失值
        return torch.where(torch.isnan(data), channel_means, data)


# --------------------------
# 主执行流程
# --------------------------
if __name__ == "__main__":
    # 配置参数
    DATA_PATH = r'E:\72fd7dbf\shiyan\my_work\chosen'
    SAVE_PATH = r'E:\72fd7dbf\shiyan\my_work\save'
    TARGET_ACC = 0.7  # 目标攻击后的准确率

    # 数据预处理
    processor = TSDataProcessor()
    X_train, y_train, X_test, y_test = processor.load_dataset("YourDataset", DATA_PATH)

    # 预训练Rocket模型
    rocket = RocketClassifier()
    rocket.fit(X_train, y_train)
    print(f"Original Accuracy: {rocket.score(X_test, y_test):.3f}")

    # 初始化攻击器
    attacker = AdversarialAttacker(rocket, lr=0.1, eta=0.5)

    # 执行对抗攻击
    final_mask = attacker.attack(X_train, y_train, epochs=200, target_acc=TARGET_ACC)

    # 评估攻击效果
    X_attacked = X_train * (1 - final_mask)
    attacked_score = rocket.score(X_attacked, y_train)
    print(f"Attacked Train Accuracy: {attacked_score:.3f}")

    # 保存结果
    np.save(os.path.join(SAVE_PATH, "adversarial_mask.npy"), final_mask)
    pd.DataFrame({"Original Acc": [rocket.score(X_test, y_test)],
                  "Attacked Acc": [attacked_score]}).to_csv(
        os.path.join(SAVE_PATH, "results.csv"), index=False)