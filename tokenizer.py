import torch

# =========================================================
# DEFINED VOCABULARY
# =========================================================
# Note: To ensure total consistency between train and api, 
# it's best to use a fixed string of all characters.
# For now, we will fallback to reading if it runs locally.

try:
    with open("dataset.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
except FileNotFoundError:
    # Fallback to a basic character set if dataset.txt isn't in the API folder
    # Ideally, save your 'chars' list to a JSON file during training and load it here!
    chars = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-()'\n")

vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# =========================================================
# ENCODE / DECODE
# =========================================================

def encode(text: str):
    return [stoi[c] for c in text if c in stoi]


def decode(tokens):
    return "".join([itos[i] for i in tokens])

# =========================================================
# DATASET TENSOR (Only used during training)
# =========================================================
dataset_tensor = None
try:
    with open("dataset.txt", "r", encoding="utf-8") as f:
        dataset_tensor = torch.tensor(
            encode(f.read()),
            dtype=torch.long
        )
except FileNotFoundError:
    pass