import torch
from convautoencoder import ConvAutoencoder

model = ConvAutoencoder()
model.load_state_dict(torch.load("./models/conv-auto-encoder.pth"))
model.eval()
checkpoint_path = "checkpoint.pt"
tensor_data = torch.load("../checkpoint.pt")

device = "cuda"
model = model.to(device)
tensor_data = tensor_data.to(device)
batch_size = 64
data_loader = torch.utils.data.DataLoader(
    tensor_data, batch_size=batch_size, shuffle=True
)

count = 0
for img in data_loader:
    with torch.no_grad():
        print(img.shape)
        encoded = model(img)
        print(encoded.shape)
        if count == 0:
            res = encoded.unsqueeze(0)
            count += 1
        else:
            res = torch.cat((res, encoded.unsqueeze(0)), dim=0)
            print(res.shape)
            count += 1
        torch.save(res, checkpoint_path)
print(count)
