import torch

print(torch.__version__)
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
print(f'CUDA VERSION: {torch.version.cuda}')
print(f'CUDA current device: {torch.cuda.current_device()}')