import torch
import torch.nn as nn
from mambabyte import MambaConfig, Mamba
from torch.utils.data import Dataset, DataLoader


class PatchDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return self.tensor.size(0)

    def __getitem__(self, idx):
        return self.tensor[idx]


checkpoint_path = "checkpoint.pt"
latent_tensor = torch.load(checkpoint_path)
print(latent_tensor.shape)
print("latent vector loaded")
print(latent_tensor.size(1) * latent_tensor.size(2) * latent_tensor.size(3))
config = MambaConfig(
    dim=latent_tensor.size(1) * latent_tensor.size(2) * latent_tensor.size(3),
    depth=3,
    dt_rank=2,
    d_state=2,
    expand_factor=2,
    d_conv=3,
    dt_min=0.001,
    dt_max=0.1,
    dt_init="random",
    dt_scale=1.0,
    bias=False,
    conv_bias=True,
    pscan=True,
)

print("model config defined")
model = Mamba(config)
print("model loaded")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 32
dataset = PatchDataset(latent_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

for epoch in range(num_epochs):
    for inputs in train_loader:
        optimizer.zero_grad()
        B, C, H, W = inputs.size()
        inputs = inputs.view(B, C * H * W)

        caches = [
            (
                None,
                torch.zeros(B, config.d_inner, config.d_conv - 1, device=inputs.device),
            )
            for _ in range(config.depth)
        ]

        loss = 0
        for t in range(inputs.size(1)):
            outputs, caches = model.step(inputs[:, t], caches)
            loss += criterion(outputs, inputs[:, t])

        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item() / inputs.size(1)}")

torch.save(model.state_dict(), "model_checkpoint.pth")
