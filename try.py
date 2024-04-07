import torch

for i in range(4, 12):
    print((torch.load(f"checkpoint_{i}.pt")).shape)
