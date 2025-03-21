import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from typing import List, Optional


# 定义一个稀疏掩码生成层
class SparseMaskGenerator(nn.Module):
    def __init__(self, input_shape, missing_rate: float = 0.1, temperature: float = 1.0):
        """
        参数：
            input_shape: 输入数据的形状 (batch_size, seq_len, features)
            missing_rate: 缺失率（缺失值的比例）
            temperature: Gumbel-Softmax温度参数
        """
        super().__init__()
        self.missing_rate = missing_rate
        self.temperature = temperature
        # 初始化可学习的logits参数
        self.logits = nn.Parameter(torch.randn(input_shape))  # [batch, seq, features]

    def forward(self):
        """
        生成稀疏掩码矩阵M，使用Gumbel-Softmax
        返回：
            mask: 稀疏二值掩码矩阵（0表示缺失，1表示保留）
        """
        # 生成Gumbel-Softmax采样
        mask = F.gumbel_softmax(self.logits, tau=self.temperature, hard=True, dim=-1)  # [batch, seq, features]

        # 应用缺失率约束：每个样本的缺失值比例为missing_rate
        batch_size, seq_len, features = mask.shape
        total_elements = seq_len * features
        num_missing = int(total_elements * self.missing_rate)  # 每个样本的缺失值数量

        # 对每个样本，选择缺失值的位置
        for b in range(batch_size):
            flat_mask = mask[b].view(-1)  # 展平
            _, indices = torch.topk(-flat_mask, k=num_missing)  # 选择最小的num_missing个值
            flat_mask[indices] = 0  # 将这些位置置为缺失
            mask[b] = flat_mask.view(seq_len, features)  # 恢复形状

        return mask


# 缺失值填充方法模块
class MissingValueFiller:
    @staticmethod
    def fill(data: torch.Tensor, mask: torch.Tensor, method: str = 'mean', model: Optional[nn.Module] = None):
        """
        填充缺失值（根据不同的方法）
        参数：
            data: 原始数据 [batch, seq, features]
            mask: 掩码矩阵（0表示缺失）
            method: 填充方法（'zero', 'mean', 'interp', 'model'）
            model: 用于基于模型填充的预训练模型
        返回：
            filled_data: 填充后的数据
        """
        masked_data = data * mask  # 应用掩码

        if method == 'zero':
            return masked_data  # 直接返回（缺失部分为0）

        elif method == 'mean':
            # 按特征维度计算均值填充
            feature_means = torch.mean(masked_data, dim=(0, 1), keepdim=True)  # [1, 1, features]
            return masked_data + (1 - mask) * feature_means

        elif method == 'interp':
            # 时间序列线性插值（沿时间维度）
            filled = masked_data.clone()
            for b in range(filled.shape[0]):
                for f in range(filled.shape[2]):
                    # 找到缺失位置并进行插值
                    valid_idx = torch.where(mask[b, :, f] == 1)[0]
                    if len(valid_idx) < 2:
                        continue
                    filled[b, :, f] = torch.interp(
                        torch.arange(filled.shape[1]).float(),
                        valid_idx.float(),
                        masked_data[b, valid_idx, f]
                    )
            return filled

        elif method == 'model' and model is not None:
            # 使用预训练模型预测缺失值
            with torch.no_grad():
                predictions = model(masked_data)
            return masked_data + (1 - mask) * predictions

        else:
            raise ValueError(f"Unsupported fill method: {method}")


# 缺失值攻击模块
class MissingValueAttack:
    def __init__(self, model: nn.Module, params: dict, input_shape: tuple):
        """
        参数：
            model: 目标分类模型
            params: 包含超参数的字典
                - missing_rate: 缺失率（缺失值的比例）
                - temperature: Gumbel-Softmax温度
                - lr: 学习率
                - n_iters: 优化迭代次数
            input_shape: 输入数据形状 (batch, seq, features)
        """
        self.model = model
        self.params = params
        self.device = next(model.parameters()).device

        # 初始化掩码生成器
        self.mask_generator = SparseMaskGenerator(
            input_shape,
            missing_rate=params['missing_rate'],
            temperature=params['temperature']
        ).to(self.device)

        # 优化器设置
        self.optimizer = optim.Adam(
            self.mask_generator.parameters(),
            lr=params['lr']
        )

    def generate_attack(self, data: torch.Tensor, t_adv: torch.Tensor, fill_method: str = 'mean'):
        """
        生成对抗性缺失模式
        参数：
            data: 原始输入数据 [batch, seq, features]
            t_adv: 目标分类结果（希望模型预测的结果）
            fill_method: 使用的填充方法
        返回：
            mask: 生成的掩码矩阵
        """
        self.model.eval()  # 固定目标模型参数

        # 将数据移动到设备
        data = data.to(self.device)
        t_adv = t_adv.to(self.device)

        # 训练掩码生成器
        for _ in tqdm(range(self.params['n_iters']), desc="Training Mask"):
            self.optimizer.zero_grad()

            # 生成掩码
            mask = self.mask_generator()

            # 应用填充
            filled_data = MissingValueFiller.fill(
                data, mask, method=fill_method
            )

            # 获取模型预测
            preds = self.model(filled_data)

            # 计算损失（分类损失 + 正则化）
            cls_loss = F.cross_entropy(preds, t_adv)  # 分类损失
            reg_loss = torch.mean(1 - mask)  # 正则化损失（鼓励更多缺失）
            total_loss = cls_loss + 0.1 * reg_loss  # 可调整权重

            # 反向传播
            total_loss.backward()
            self.optimizer.step()

        # 生成最终掩码（应用阈值）
        with torch.no_grad():
            final_mask = (self.mask_generator() > 0.5).float()

        return final_mask.detach().cpu()


# 示例使用代码
if __name__ == "__main__":
    # 假设参数配置
    params = {
        'missing_rate': 0.1,  # 缺失率为10%
        'temperature': 0.1,
        'lr': 0.01,
        'n_iters': 100
    }

    # 假设输入数据（batch=32, seq_len=10, features=20）
    input_shape = (32, 10, 20)
    dummy_data = torch.randn(input_shape)
    dummy_target = torch.randint(0, 5, (32,))  # 假设5分类任务


    # 假设分类模型（简单示例）
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(20, 64, batch_first=True)
            self.fc = nn.Linear(64, 5)

        def forward(self, x):
            _, h = self.rnn(x)
            return self.fc(h.squeeze(0))


    model = DummyModel()

    # 初始化攻击模块
    attacker = MissingValueAttack(model, params, input_shape)

    # 生成对抗性掩码（希望模型预测错误）
    adversarial_mask = attacker.generate_attack(
        dummy_data,
        t_adv=(dummy_target + 1) % 5,  # 目标标签设为真实标签+1（模拟攻击）
        fill_method='mean'
    )

    # 应用填充并评估
    filled_data = MissingValueFiller.fill(dummy_data, adversarial_mask, 'mean')
    with torch.no_grad():
        preds = model(filled_data)
        acc = (preds.argmax(dim=1) == dummy_target).float().mean()

    print(f"原始准确率：{(model(dummy_data).argmax(1) == dummy_target).float().mean():.2f}")
    print(f"攻击后准确率：{acc:.2f}")