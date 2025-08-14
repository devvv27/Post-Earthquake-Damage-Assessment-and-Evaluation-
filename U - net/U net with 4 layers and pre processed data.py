"""
Enhanced U-Net (4-stage) training + validation script.

Requirements:
 - PyTorch
 - Your preprocessed files:
    - "stacked_6channel.pt"  -> torch.Tensor shape [N, 6, 256, 256], dtype=float32, values in [0,1]
    - "masks.pt"             -> torch.Tensor shape [N, 1, 256, 256] (or [N,256,256]), values {0,1}
"""


import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
# -------------------------
# Main: load data, setup, run
# -------------------------
class StackedTensorDataset(Dataset):
    """
    Dataset for stacked tensors with on-the-fly augmentation.
    Only applies augmentation if augment=True.
    """
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[2])
                y = torch.flip(y, dims=[2])
            # Random vertical flip
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[1])
            # Random 90-degree rotation
            if torch.rand(1) < 0.5:
                k = random.choice([1, 2, 3])
                x = torch.rot90(x, k, dims=[1, 2])
                y = torch.rot90(y, k, dims=[1, 2])
            # Random Gaussian noise
            if torch.rand(1) < 0.3:
                noise = torch.randn_like(x) * 0.02
                x = torch.clamp(x + noise, 0, 1)
        return x, y

# -------------------------
# Reproducibility
# -------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# -------------------------
# Model: Enhanced U-Net (6 encoder stages)
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, dropout=0.0):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        layers = [
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0.0:
            # place dropout after the conv stack
            layers.append(nn.Dropout2d(p=dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        # in_ch: channels of the previous decoder feature
        # skip_ch: channels from encoder for concat
        # out_ch: desired output channels
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # handle size mismatch (if any) by padding
        if x.size() != skip.size():
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class EnhancedUNet6(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, base_feats=32):
        super().__init__()
        f = [base_feats, base_feats*2, base_feats*4, base_feats*8, base_feats*16, base_feats*16]  # six stages

        # Encoder (no pooling for the first conv)
        self.inc = DoubleConv(in_channels, f[0], dropout=0.0)
        self.down1 = Down(f[0], f[1], dropout=0.0)
        self.down2 = Down(f[1], f[2], dropout=0.0)
        # add dropout in deeper encoder layers to prevent overfitting
        self.down3 = Down(f[2], f[3], dropout=0.2)
        self.down4 = Down(f[3], f[4], dropout=0.3)
        self.down5 = Down(f[4], f[5], dropout=0.3)

        # Bottleneck (extra conv)
        self.bottleneck = DoubleConv(f[5], f[5], dropout=0.5)

        # Decoder (mirrors encoder)
        self.up5 = Up(f[5], f[4], f[4], dropout=0.3)
        self.up4 = Up(f[4], f[3], f[3], dropout=0.2)
        self.up3 = Up(f[3], f[2], f[2], dropout=0.1)
        self.up2 = Up(f[2], f[1], f[1], dropout=0.0)
        self.up1 = Up(f[1], f[0], f[0], dropout=0.0)

        # Final conv
        self.outc = nn.Conv2d(f[0], out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.inc(x)     # 256
        e2 = self.down1(e1)  # 128
        e3 = self.down2(e2)  # 64
        e4 = self.down3(e3)  # 32
        e5 = self.down4(e4)  # 16
        e6 = self.down5(e5)  # 8

        b = self.bottleneck(e6)  # 8

        d5 = self.up5(b, e5)  # up to 16
        d4 = self.up4(d5, e4) # up to 32
        d3 = self.up3(d4, e3) # up to 64
        d2 = self.up2(d3, e2) # up to 128
        d1 = self.up1(d2, e1) # up to 256

        out = self.outc(d1)
        return out  # logits (no sigmoid)

# -------------------------
# Losses & Metrics
# -------------------------
def dice_loss(probs, target, smooth=1e-6):
    # probs & target: B x 1 x H x W
    probs = probs.contiguous().view(probs.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (probs * target).sum(dim=1)
    denom = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        logits = logits
        probs = torch.sigmoid(logits)
        bce = self.bce(logits, targets)
        d = dice_loss(probs, targets)
        return bce + d

def iou_score(probs, targets, thr=0.5, eps=1e-6):
    # probs & targets: B x 1 x H x W
    preds = (probs >= thr).float()
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

# -------------------------
# Training / Validation loop
# -------------------------
def train_val_loop(
    model,
    train_loader,
    val_loader,
    device,
    epochs=50,
    patience=10,
    lr=1e-4,
    save_path="best_enhanced_unet.pth",
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = BCEDiceLoss()
    best_val_loss = float("inf")
    trigger = 0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Train", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)                    # shape B x 1 x H x W
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{train_loss / (pbar.n+1):.4f}"})

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                val_iou += iou_score(probs.detach().cpu(), yb.detach().cpu())

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

        # --- Early stopping & save best ---
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            trigger = 0
            torch.save(model.state_dict(), save_path)
            print("  -> Best model saved.")
        else:
            trigger += 1
            print(f"  -> No improvement (trigger {trigger}/{patience})")
            if trigger >= patience:
                print("Early stopping triggered. Stopping training.")
                break

# -------------------------
# Main: load data, setup, run
# -------------------------
if __name__ == "__main__":
    # Paths (change if needed)
    stacked_path = "stacked_6channel.pt"
    masks_path = "masks.pt"

    assert os.path.exists(stacked_path), f"{stacked_path} not found."
    assert os.path.exists(masks_path), f"{masks_path} not found."

    stacked = torch.load(stacked_path)  # expected [N,6,256,256]
    masks = torch.load(masks_path)      # expected [N,1,256,256] or [N,256,256]

    # normalize types & shapes
    stacked = stacked.float()
    masks = masks.float()
    if masks.dim() == 3:  # [N,H,W] -> [N,1,H,W]
        masks = masks.unsqueeze(1)

    # ensure binary masks
    masks = (masks >= 0.5).float()

    N = stacked.shape[0]
    print(f"Loaded: stacked {stacked.shape}, masks {masks.shape}, N={N}")

    # Random split: 60% train, 20% val, 20% test
    indices = np.arange(N)
    np.random.shuffle(indices)
    n_train = int(0.6 * N)
    n_val = int(0.2 * N)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    X_train, y_train = stacked[train_idx], masks[train_idx]
    X_val, y_val = stacked[val_idx], masks[val_idx]
    X_test, y_test = stacked[test_idx], masks[test_idx]

    print(f"Split -> train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}")


    # Create DataLoaders with augmentation for training set
    batch_size = 4   # change to 6 if GPU mem allows
    train_loader = DataLoader(StackedTensorDataset(X_train, y_train, augment=True), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(StackedTensorDataset(X_val, y_val, augment=False), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(StackedTensorDataset(X_test, y_test, augment=False), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Instantiate model
    model = EnhancedUNet6(in_channels=6, out_channels=1, base_feats=32).to(device)

    # Train with early stopping (patience=10)
    train_val_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=50,
        patience=10,
        lr=1e-4,
        save_path="best_enhanced_unet.pth",
    )

