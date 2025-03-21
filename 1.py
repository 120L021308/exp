import torch
print(torch.cuda.is_available())  # 应输出 True
print(torch.cuda.device_count()) # 应 > 0
print(torch.cuda.get_device_name(0)) # 应显示GPU型号