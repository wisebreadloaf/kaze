import torch
from convautoencoderv2 import ConvAutoencoder

model = ConvAutoencoder()
model.load_state_dict(torch.load("./models/conv-auto-encoder.pth"))
model.eval()
for i in range(3, 4):
    checkpoint_path = f"./out_data/checkpoint_{i}.pt"
    count = 0
    tensor_data = torch.load(f"./raw_data/checkpoint_{i}.pt")
    device = "cuda"
    model = model.to(device)
    tensor_data = tensor_data.to(device)
    batch_size = 64
    data_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=batch_size, shuffle=True
    )

    for img in data_loader:
        with torch.no_grad():
            print(img.shape)
            encoded, decoded = model(img)
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
