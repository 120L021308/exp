# run_experiment.py
import os
import subprocess


def main():
    # 配置实验参数
    config = {
        'task_name': 'classification',
        'is_training': 1,
        'root_path': './dataset/EthanolLevel',
        'model_id': 'EthanolLevel',
        'model': 'TimesNet',
        'data': 'UEA',
        'e_layers': 2,
        'batch_size': 16,
        'd_model': 16,
        'd_ff': 32,
        'top_k': 3,
        'des': 'Exp',
        'itr': 1,
        'learning_rate': 0.001,
        'train_epochs': 30,
        'patience': 10
    }

    # 设置CUDA可见设备
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 构建命令行参数列表
    cmd = ["python", "-u", "run.py"]
    for key, value in config.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    # 执行命令
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)

    # 实时打印输出
    for line in iter(process.stdout.readline, ''):
        print(line, end='')

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


if __name__ == "__main__":
    main()