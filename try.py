import torchshow as ts
import torch
import numpy as np
from PIL import Image
import model_loader
from mambabyte import MambaConfig, Mamba


image_path = "./images/dog.jpeg"
input_image = Image.open(image_path)

WIDTH = 512
HEIGHT = 512


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

device = DEVICE

# model_file = "../data/v1-5-pruned-emaonly.ckpt"
# models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

models = model_loader.preload_models_from_standard_weights(DEVICE)
encoder = models["encoder"]
encoder.to(device)

input_image_tensor = input_image.resize((WIDTH, HEIGHT))
input_image_tensor = np.array(input_image_tensor)
input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
input_image_tensor = input_image_tensor.unsqueeze(0)
input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
input_image_tensor = input_image_tensor[0]
print(input_image_tensor)
print(input_image_tensor.shape)

# ts.show(input_image_tensor[0])
# output = input_image_tensor1.unsqueeze(0)
# result = torch.equal(input_image_tensor, output)
# print(result)

config = MambaConfig(
    dim=512,
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


model = Mamba(config)

out = model(input_image_tensor)
print(out)
print(out.shape)
ts.show(input_image_tensor)
ts.show(out)
