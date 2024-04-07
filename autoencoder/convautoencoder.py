import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_patches(tensor):
    fig = plt.figure(figsize=(16, 16))
    grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0)
    for i, ax in enumerate(grid):
        patch = tensor[i].permute(1, 2, 0).cpu().detach().numpy()
        ax.imshow(patch)
        ax.axis("off")
    plt.show()


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 256 * 4 * 4),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
        self.to("cuda")

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x):
        encoded = self.encoder(x)
        return encoded

    def decode(self, x):
        decoded = self.decoder(x)
        return decoded


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = ConvAutoencoder().to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
# num_epochs = 50
# # for i in range(1, 5):
# tensor_data = torch.load("./raw_data/checkpoint.pt")
# for i in range(1, 14):
#     tensor_data = torch.cat(
#         tensor_data, torch.load(f"./raw_data/checkpoint_{1}.pt"), dim=0
#     )
# print("tensor loaded")
# tensor_data = tensor_data.to(device)
# split_idx = int(0.8 * len(tensor_data))
# train_data = TensorDataset(tensor_data[:split_idx])
# val_data = TensorDataset(tensor_data[split_idx:])
# batch_size = 64
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
# val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
# best_val_loss = float("inf")
#
# for epoch in range(num_epochs):
#     train_loss = 0.0
#     for img in train_loader:
#         img = img[0].to(device)
#         encoded, decoded = model(img)
#         loss = criterion(decoded, img)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#     train_loss /= len(train_loader)
#
#     val_loss = 0.0
#     with torch.no_grad():
#         for img in val_loader:
#             img = img[0].to(device)
#             encoded, decoded = model(img)
#             loss = criterion(decoded, img)
#             val_loss += loss.item()
#     val_loss /= len(val_loader)
#
#     print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), "./models/conv-auto-encoder-v5.pth")
