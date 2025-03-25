# run_experiment.py
import os
import subprocess
import torch
import torch.nn as nn


class MaskGenerator(nn.Module):
    def __init__(self, seq_len, feat_dim, tau=0.1, max_missing=0.2):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(seq_len, feat_dim))  # 伯努利概率参数
        self.tau = tau
        self.max_missing = max_missing

    def forward(self):
        # Gumbel-Softmax生成连续掩码
        u = torch.rand_like(self.logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-10))
        m_soft = torch.sigmoid((self.logits + gumbel_noise) / self.tau)

        # 投影到稀疏性约束
        if self.training:  # 训练时保留梯度
            return m_soft
        else:  # 推理时二值化
            threshold = torch.quantile(m_soft.flatten(), self.max_missing)
            return (m_soft > threshold).float()


# 填充模块（示例：线性插值）
class LinearImputer(nn.Module):
    def forward(self, x_masked):
        # x_masked: [B, T, D], 0表示缺失
        x_filled = x_masked.clone()
        for b in range(x_masked.size(0)):
            for d in range(x_masked.size(2)):
                t = 0
                while t < x_masked.size(1):
                    if x_masked[b, t, d] == 0:
                        # 找到缺失段起始和结束位置
                        start = t
                        while t < x_masked.size(1) and x_masked[b, t, d] == 0:
                            t += 1
                        end = t
                        # 线性插值
                        if start > 0 and end < x_masked.size(1):
                            x_start = x_masked[b, start - 1, d]
                            x_end = x_masked[b, end, d]
                            x_filled[b, start:end, d] = torch.linspace(x_start, x_end, steps=end - start)
        return x_filled


def adversarial_loss(y_pred, y_true, t_adv, mask, max_missing=0.2):
    # 分类性能差距
    ce = nn.CrossEntropyLoss()(y_pred, y_true)
    loss_gap = torch.abs(ce - t_adv)

    # 稀疏性惩罚
    missing_ratio = 1 - mask.mean()
    loss_sparsity = torch.relu(missing_ratio - max_missing)

    # 总损失
    total_loss = loss_gap + 10.0 * loss_sparsity  # λ=10
    return total_loss

# def main():
#     # 配置实验参数
#     config = {
#         'task_name': 'classification',
#         'is_training': 1,
#         'root_path': './dataset/EthanolLevel',
#         'model_id': 'EthanolLevel',
#         'model': 'TimesNet',
#         'data': 'UEA',
#         'e_layers': 2,
#         'batch_size': 16,
#         'd_model': 16,
#         'd_ff': 32,
#         'top_k': 3,
#         'des': 'Exp',
#         'itr': 1,
#         'learning_rate': 0.001,
#         'train_epochs': 5,
#         'patience': 10
#     }
#
#     # 设置CUDA可见设备
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
#     # 构建命令行参数列表
#     cmd = ["python", "-u", "run.py"]
#     for key, value in config.items():
#         cmd.append(f"--{key}")
#         cmd.append(str(value))
#
#     # 执行命令
#     process = subprocess.Popen(cmd,
#                                stdout=subprocess.PIPE,
#                                stderr=subprocess.PIPE,
#                                universal_newlines=True)
#
#     # 实时打印输出
#     for line in iter(process.stdout.readline, ''):
#         print(line, end='')
#
#     process.stdout.close()
#     return_code = process.wait()
#
#     if return_code != 0:
#         raise subprocess.CalledProcessError(return_code, cmd)
#
#
# if __name__ == "__main__":
#     main()
