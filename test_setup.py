import torch
import torchvision
import numpy as np
import platform

print("=" * 50)
print("Setup Verification")
print("=" * 50)
print(f"Python version:      {platform.python_version()}")
print(f"PyTorch version:     {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"NumPy version:       {np.__version__}")
print(f"CUDA available:      {torch.cuda.is_available()}")
print(f"MPS available:       {torch.backends.mps.is_available()}")
print(f"CPU threads:         {torch.get_num_threads()}")

# Quick sanity check: do some math
x = torch.randn(3, 3)
y = torch.randn(3, 3)
z = x @ y
print(f"\nMatrix multiplication test: {'PASSED' if z.shape == (3, 3) else 'FAILED'}")

# Tiny neural network forward pass
import torch.nn as nn
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
out = model(torch.randn(2, 10))
print(f"Tiny network test:           {'PASSED' if out.shape == (2, 1) else 'FAILED'}")

print("\n✅ Everything looks good!" if z.shape == (3, 3) else "\n❌ Something went wrong")
