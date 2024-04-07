import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from mpl_toolkits.axes_grid1 import ImageGrid

# from autoencoder.convautoencoder import ConvAutoencoder


class patchify(nn.Module):
    def __init__(self, patch_size=3):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = self.unfold(x)
        a = x.view(bs, c, self.patch_size, self.patch_size, -1).permute(0, 4, 1, 2, 3)
        return a


def plot_patches(tensor):
    fig = plt.figure(figsize=(16, 16))
    grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0)
    for i, ax in enumerate(grid):
        patch = tensor[i].permute(1, 2, 0).cpu().detach().numpy()
        ax.imshow(patch)
        ax.axis("off")
    plt.show()


# model = ConvAutoencoder()
# model.load_state_dict(torch.load("./autoencoder/models/conv-auto-encoder.pth"))
checkpoint_path = "checkpoint.pt"

count = 3
ccount = 3
for i in range(1228, 3148, 192):
    ccount += 1
    count = 4
    for j in range(i, i + 192):
        print(count)
        image_path = os.path.join(
            "./videoprocessing/images/Arrietty", f"Arrietty_frame_{j:08d}.jpg"
        )
        print("processing", image_path)
        input_image = Image.open(image_path)
        width = 512
        height = 512
        input_image_tensor = input_image.resize((width, height))
        input_image_tensor = np.array(input_image_tensor)
        input_image_tensor = input_image_tensor.astype("float32") / 255.0
        input_image_tensor = (torch.from_numpy(input_image_tensor)).to("cuda")
        input_image_tensor = input_image_tensor.unsqueeze(0)
        input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
        patch_size = 64
        patch = patchify(patch_size=patch_size)
        patches = patch(input_image_tensor)
        patches = patches.squeeze()
        encoded = patches
        if count == 4:
            res = encoded
            count += 1
        else:
            # res = torch.cat((res, encoded.unsqueeze(0)), dim=0)
            res = torch.cat((res, encoded), dim=0)
            print(encoded.shape)
            print(res.shape)
            count += 1

        print(f"checkpoint_{ccount}.pt")
        torch.save(res, f"checkpoint_{ccount}.pt")
