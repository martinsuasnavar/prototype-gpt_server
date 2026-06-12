import os
import torch

from config import (
    BLOCK_SIZE,
    BATCH_SIZE,
    TRAIN_STEPS,
    LEARNING_RATE,
    MODEL_PATH
)
from tokenizer import dataset_tensor
from model import MiniGPT

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch():
    if dataset_tensor is None:
        raise FileNotFoundError("dataset.txt is required for training!")
        
    ix = torch.randint(0, len(dataset_tensor) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([dataset_tensor[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([dataset_tensor[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)

if __name__ == "__main__":
    print(f"Using device: {device}")
    
    model = MiniGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...\n")
    for step in range(TRAIN_STEPS):
        xb, yb = get_batch()

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"step={step} loss={loss.item():.4f}")

    print("\nTraining complete.")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Checkpoint saved -> {MODEL_PATH}")