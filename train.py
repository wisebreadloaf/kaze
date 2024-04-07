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


config = MambaConfig(
    dim=256,
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
model = model.to("cuda")
print("model loaded")

for i in range(1, 4):
    checkpoint_path = f"./autoencoder/out_data/checkpoint_{i}.pt"
    latent_tensor = torch.load(checkpoint_path)
    print(latent_tensor.shape)
    print("latent vector loaded")
    print(latent_tensor.shape)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    batch_size = 64
    dataset = PatchDataset(latent_tensor)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    for epoch in range(num_epochs):
        for inputs in train_loader:
            inputs = inputs.to("cuda")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    torch.save(model.state_dict(), "./models/model_checkpoint.pth")
