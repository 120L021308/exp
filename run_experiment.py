import os
import subprocess
import sys


def main():
    # 配置实验参数
    config = {
        'task_name': 'classification',
        'is_training': 1,
        'root_path': './dataset/Chinatown/',
        'model_id': 'Chinatown_AdvTest',
        'model': 'Informer',
        'data': 'UEA',
        'e_layers': 2,
        'batch_size': 16,
        'd_model': 16,
        'd_ff': 32,
        'top_k': 3,
        'des': 'Exp',
        'itr': 1,
        'learning_rate': 0.001,
        'train_epochs': 20,
        'patience': 10,
        'max_missing': 0.4,
        'target_performance_drop': 0.2,
        'mask_learning_rate': 0.05,
        'performance_threshold': 0.05,
        'lambda_sparsity': 100.0




    }

    # 设置CUDA可见设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 构建命令行参数列表
    cmd = ["python", "-u", "run.py"]
    for key, value in config.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    cmd.append("--use_adversarial_mask")

    print("执行命令:", " ".join(cmd))

    # 执行命令，同时捕获标准输出和标准错误
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True,  # 这替代了 universal_newlines=True
                               encoding='utf-8')  # 显式使用 UTF-8

    # 创建函数来处理输出流
    def print_output(stream, prefix=''):
        for line in iter(stream.readline, ''):
            print(f"{prefix}{line}", end='')
            sys.stdout.flush()  # 确保实时显示

    # 实时打印标准输出
    print_output(process.stdout)

    # 获取错误输出
    stderr_output = process.stderr.read()

    # 等待进程完成
    return_code = process.wait()

    # 如果有错误输出，打印它
    if stderr_output:
        print("\n--- 错误输出 ---")
        print(stderr_output)

    # 检查返回码
    if return_code != 0:
        print(f"\n命令执行失败，退出代码: {return_code}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("执行失败，请检查上述错误信息")
    else:
        print("执行成功")