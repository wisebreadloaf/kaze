import torch
import model_loader
from transformers import CLIPTokenizer

DEVICE = "cuda"
models = model_loader.preload_models_from_standard_weights(DEVICE)
clip = models["clip"]
device = "cuda"
tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 8k resolution"
tokens = tokenizer.batch_encode_plus(
    [prompt], padding="max_length", max_length=77
).input_ids
tokens = torch.tensor(tokens, dtype=torch.long, device=device)
context = clip(tokens)
print(context)
print(context.shape)
