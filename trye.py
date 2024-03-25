import numpy as np
import model_loader
import torch
from PIL import Image

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

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


device = DEVICE
models = model_loader.preload_models_from_standard_weights(DEVICE)
encoder = models["encoder"]
encoder.to(device)

image_path = "./videoprocessing/images/frame_00434.jpg"
image = Image.open(image_path)
image = image.resize((WIDTH, HEIGHT))
image_tensor = torch.tensor(np.array(image), dtype=torch.float32)
image_tensor = rescale(image_tensor, (0, 255), (-1, 1))
image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.permute(0, 3, 1, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)
print(image_tensor.shape)
# encoder = models["encoder"].to(device)
# with torch.no_grad():
#     latents = encoder(image_tensor)
#
# print(latents)
# print(latents.shape)
