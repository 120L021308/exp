import torch
import torch.nn as nn


class GumbelMaskGenerator(nn.Module):
    def __init__(self, seq_len, feat_dim, max_missing=0.2, tau=0.5):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(seq_len, feat_dim))  # 可学习参数
        self.max_missing = max_missing
        self.tau = tau

    def forward(self, x, hard=True):
        # 生成Gumbel噪声
        u = torch.rand_like(self.logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-10))

        # 计算连续掩码
        m_soft = torch.sigmoid((self.logits + gumbel_noise) / self.tau)

        # 训练时保持连续，推理时二值化
        if not self.training and hard:
            # 投影到稀疏性约束：保留top (1-max_missing)的元素
            k = int((1 - self.max_missing) * m_soft.numel())
            threshold = torch.topk(m_soft.flatten(), k, sorted=False).values.min()
            m_hard = (m_soft >= threshold).float()
            return m_hard
        return m_soft


class LinearImputer(nn.Module):
    @staticmethod
    def forward(x_masked):
        # 沿时间维度线性插值
        x_filled = x_masked.clone()
        mask = (x_masked != 0).float()
        for d in range(x_masked.size(-1)):
            for b in range(x_masked.size(0)):
                t = 0
                while t < x_masked.size(1):
                    if mask[b, t, d] == 0:
                        start = t - 1 if t > 0 else 0
                        while t < x_masked.size(1) and mask[b, t, d] == 0:
                            t += 1
                        end = t if t < x_masked.size(1) else x_masked.size(1) - 1
                        if start < end:
                            x_filled[b, start:end, d] = torch.linspace(
                                x_masked[b, start, d], x_masked[b, end, d], steps=end - start
                            )
        return x_filled