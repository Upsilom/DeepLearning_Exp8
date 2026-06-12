import torch

print("=" * 40)
print("GPU 检测")
print("=" * 40)

if torch.cuda.is_available():
    print(f"✅ CUDA 可用")
    print(f"   GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"   当前设备: {torch.cuda.current_device()}")
    print(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print(f"❌ CUDA 不可用")
    print(f"   将使用 CPU 训练")
    print(f"   CPU 核心数: {torch.get_num_threads()}")

print(f"\n   PyTorch版本: {torch.__version__}")
print("=" * 40)
