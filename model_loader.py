from clip import CLIP
from encoder import VAE_Encoder

# import model_converter


# def preload_models_from_standard_weights(ckpt_path, device):
#     state_dict = model_converter.load_from_standard_weights(ckpt_path, device)
#
#     encoder = VAE_Encoder().to(device)
#     encoder.load_state_dict(state_dict["encoder"], strict=True)
#
#     return {"encoder": encoder}


def preload_models_from_standard_weights(device):
    # state_dict = model_converter.load_from_standard_weights(device)

    encoder = VAE_Encoder().to(device)
    # encoder.load_state_dict(state_dict["encoder"], strict=True)
    clip = CLIP().to(device)
    return {"clip": clip, "encoder": encoder}
