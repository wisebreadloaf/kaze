# import torch
import torch.nn as nn

# import torch.optim as optim


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 256 * 4 * 4),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
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
# optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
# num_epochs = 320
#
# tensor_data = torch.load("../checkpoint.pt")
# tensor_data = tensor_data.to(device)
#
# batch_size = 64
# data_loader = torch.utils.data.DataLoader(
#     tensor_data, batch_size=batch_size, shuffle=True
# )
#
# for epoch in range(num_epochs):
#     for img in data_loader:
#         encoded, decoded = model(img)
#         loss = criterion(decoded, img)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
#
# torch.save(model.state_dict(), "./models/conv-auto-encoder.pth")
