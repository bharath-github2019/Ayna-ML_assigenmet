import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    wandb.init(project="ayna-unet-colorization", name="unet-run-1")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, masks in loop:
            inputs, masks = inputs.to(device), masks.to(device).squeeze(1)  # Fix shape: [B, 256, 256]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, masks in val_loader:
                inputs, masks = inputs.to(device), masks.to(device).squeeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

    wandb.finish()
