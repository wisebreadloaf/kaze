import torch
from mambabyte import MambaConfig, Mamba
import os
from PIL import Image
import numpy as np
import torch.nn as nn
from autoencoder.convautoencoder import ConvAutoencoder
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


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


encoder_model = ConvAutoencoder()
encoder_model.load_state_dict(torch.load("./autoencoder/models/conv-auto-encoder.pth"))

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
mamba_model = Mamba(config)

mamba_model.load_state_dict(torch.load("model_checkpoint.pth"))

mamba_model = mamba_model.to("cuda")
count = 0
for i in range(1, 5):
    if i == 1:
        image_path = os.path.join(
            "./videoprocessing/images/", f"Arrietty_frame_{479:08d}.jpg"
        )
        print("processing", image_path)
        input_image = Image.open(image_path)
        width = 512
        height = 512
        input_image_tensor = input_image.resize((width, height))
        input_image_tensor = np.array(input_image_tensor)
        input_image_tensor = input_image_tensor.astype("float32") / 255.0
        input_image_tensor = torch.from_numpy(input_image_tensor)
        input_image_tensor = input_image_tensor.unsqueeze(0)
        input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
        input_image_tensor = input_image_tensor.cuda()
        patch_size = 64
        patch = patchify(patch_size=patch_size)
        patches = patch(input_image_tensor)
        patches = patches.squeeze()
        plot_patches(patches)
        encoded, decoded = encoder_model(patches)
        encoded = encoded.unsqueeze(0)
        print(encoded)
        print(encoded.shape)
        with torch.no_grad():
            outputs = mamba_model(encoded)

        print(outputs)
        print(outputs.shape)
        decoded = encoder_model.decode(outputs.squeeze())
        print(decoded)
        print(decoded.shape)
        plot_patches(decoded)
    # else:
    #     plot_patches(decoded)
    #     encoded, decoded = encoder_model(decoded)
    #     print(encoded)
    #     print(encoded.shape)
    #     print(decoded)
    #     print(decoded.shape)
    #     plot_patches(decoded)
