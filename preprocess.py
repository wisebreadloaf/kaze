import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
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


checkpoint_path = "checkpoint.pt"
if not os.path.exists(checkpoint_path):
    res = None
else:
    res = torch.load(checkpoint_path)

for i in range(408, 984):
    image_path = os.path.join("./videoprocessing/images/", f"frame_{i:05d}.jpg")
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
    if i == 408:
        res = patches
    else:
        res = torch.cat((res, patches), dim=0)
        print(res.shape)
    torch.save(res, checkpoint_path)
# res = res.reshape(x.shape)
# df = df.append(pd.DataFrame(res), ignore_index=True)
# x = torch.rand((17280, 3, 64, 64), dtype=torch.float32)
# y_flattened = x.flatten()
# y_reshaped = y_flattened.reshape(x.shape)
# print(torch.equal(x, y_reshaped))
# plot_patches(patches)

# config = MambaConfig(
#     dim=64,
#     depth=3,
#     dt_rank=2,
#     d_state=2,
#     expand_factor=2,
#     d_conv=3,
#     dt_min=0.001,
#     dt_max=0.1,
#     dt_init="random",
#     dt_scale=1.0,
#     bias=False,
#     conv_bias=True,
#     pscan=True,
# )
#
# device = "cuda"
# mamba_model = Mamba(config).to(device)
# output_patches_list = []
# scaler = torch.cuda.amp.GradScaler()
# with torch.cuda.amp.autocast():
#     for i in range(64):
#         output_patch = mamba_model(patches[i])
#         output_patch = output_patch.unsqueeze(0)
#         output_patches_list.append(output_patch)
