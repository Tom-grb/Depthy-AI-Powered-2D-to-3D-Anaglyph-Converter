import torch
print(f"Torch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. This usually means the CPU-only version of PyTorch is installed.")
    print("Please reinstall PyTorch with CUDA support using the command specific to your CUDA version.")
    print("Visit https://pytorch.org/get-started/locally/ for the command.")
