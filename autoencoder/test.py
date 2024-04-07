import torch
from convautoencoder import ConvAutoencoder
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
import numpy as np
import torch.nn as nn
import os


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


model = ConvAutoencoder()
model.load_state_dict(torch.load(f"./models/conv-auto-encoder-v{4}.pth"))
# model.load_state_dict(torch.load("./models/conv-auto-encoder-v3.pth"))
model.eval()

count = 0
image_path = os.path.join(
    "../videoprocessing/images/Arrietty/", f"Arrietty_frame_{3480:08d}.jpg"
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
img = patches.squeeze()
with torch.no_grad():
    print(img.shape)
    plot_patches(img)
    encoded, decoded = model(img)
    print(decoded.shape)
    plot_patches(decoded)
    criterion = nn.MSELoss()
    loss = criterion(decoded, img)
    print(loss)
    print(loss.item())
