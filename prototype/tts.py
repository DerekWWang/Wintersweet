# %%
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler



from slide_util import slide_to_tiles

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import os

from timm.data import resolve_model_data_config, create_transform



BASE_CHECKPOINT = "hf_hub:Snarcy/RedDino-base"
SMALL_CHECKPOINT = "hf_hub:Snarcy/RedDino-small"


class GatedAttentionMIL_RED(nn.Module):
    def __init__(self, base_checkpoint, M=500, L=128, attention_branches=1):
        super().__init__()
        self.M, self.L, self.B = M, L, attention_branches

        # Backbone returns a per-tile embedding (D=768)
        self.backbone = timm.create_model(base_checkpoint, pretrained=True, num_classes=0)  # no classifier head
        self.embed_dim = 384

        # Project to attention space
        self.feature_projector = nn.Sequential(
            nn.Linear(384, M),   # <<— your 768 here
            nn.ReLU(inplace=True),
        )

        # Gated attention
        self.attention_V = nn.Sequential(nn.Linear(M, L), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(M, L), nn.Sigmoid())
        self.attention_w = nn.Linear(L, self.B)

        # Bag classifier
        self.classifier = nn.Linear(M * self.B, 1)


    def forward(self, tiles, mask=None):
        """
        tiles: [K, 3, 244, 244]
        mask:  [K] boolean (optional)
        """
        H = self.backbone(tiles)          # [K, 768]
        H = self.feature_projector(H)     # [K, M]

        A_V = self.attention_V(H)         # [K, L]
        A_U = self.attention_U(H)         # [K, L]
        A = self.attention_w(A_V * A_U).transpose(0, 1)  # [B, K]

        if mask is not None:
            A = A.masked_fill(~mask.unsqueeze(0), float('-inf'))

        A = F.softmax(A, dim=1)           # over tiles
        Z = A @ H                         # [B, M]
        bag_repr = Z.reshape(1, -1)
        logits = self.classifier(bag_repr)
        y_prob = torch.sigmoid(logits)  # [1, 1]
        Y_hat = (y_prob >= 0.5).float()

        return logits, y_prob, Y_hat, A

class ImageBagDataset(Dataset):
    def __init__(self, csv_file, train=True):
        self.data = pd.read_csv(csv_file)
        if train:
            self.path = "C:\\Code\\DL\\bbosis\\data\\bags\\train\\"
        else:
            self.path = "C:\\Code\\DL\\bbosis\\data\\bags\\test\\"
        self.slide_ids = self.data["img_pth"].astype(str).tolist()
        self.labels = (self.data["label"]).astype(np.int64).to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bag_dir = os.path.join(self.path, self.slide_ids[idx])
        bag = []
        for img_name in sorted(os.listdir(bag_dir)):
            img_path = os.path.join(bag_dir, img_name)
            with Image.open(img_path) as img:
                img_array = np.array(img.convert("RGB"), dtype=np.uint8)
            bag.append(img_array)

        bag_np = np.asarray(bag)
        # If np.asarray produced an object dtype (e.g., empty list or ragged),
        # fall back to np.stack which will error earlier for inconsistent shapes.
        if bag_np.dtype == object:
            bag_np = np.stack(bag, axis=0)
        bag_np = bag_np.astype(np.float32) / 255.0
        bag_tensor = torch.from_numpy(bag_np).permute(0, 3, 1, 2)  # [K, 3, 224, 224]
        label = int(self.labels[idx])
        slide_id = self.slide_ids[idx]
        return bag_tensor, label, slide_id

train_dataset = ImageBagDataset("C:\\Code\\DL\\bbosis\\data\\train_set.csv", train=True)
test_dataset = ImageBagDataset("C:\\Code\\DL\\bbosis\\data\\test_set.csv", train=False)


import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
num_epochs = 5
learning_rate = 1e-4

# Checkpointing settings (global variables)
# Set to True to save a checkpoint every `save_every` epochs
save_checkpoints = True
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
save_every = 1

model = GatedAttentionMIL_RED(SMALL_CHECKPOINT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_label_counts = np.bincount(train_dataset.labels, minlength=2)
num_negative, num_positive = train_label_counts[0], train_label_counts[1]
pos_weight_value = (num_negative / num_positive) if num_positive > 0 else 1.0
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def single_slide_collate(batch):
    """Ensure each batch contains a single slide bag without extra stacking."""
    return batch[0]

def normalize_slide_id(slide_id):
    if isinstance(slide_id, (list, tuple)):
        slide_id = slide_id[0]
    if hasattr(slide_id, "item"):
        slide_id = slide_id.item()
    return str(slide_id)

class_weights = np.zeros_like(train_label_counts, dtype=np.float32)
nonzero_mask = train_label_counts > 0
class_weights[nonzero_mask] = (train_label_counts[nonzero_mask].sum()) / (len(class_weights) * train_label_counts[nonzero_mask])
sample_weights = class_weights[train_dataset.labels]
train_sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)
train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, collate_fn=single_slide_collate)
val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=single_slide_collate)

history = []
misclassified_history = []

for epoch in range(num_epochs):
    model.train()
    train_losses, train_probs, train_preds, train_labels = [], [], [], []
    train_misclassified = []

    # Training loop with progress bar
    pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train", leave=False)
    for bag, label, slide_id in pbar_train:
        bag = bag.to(device)
        label_tensor = torch.as_tensor(label, dtype=torch.float32, device=device).view(1, 1)

        optimizer.zero_grad()
        logits, y_prob, _, _ = model(bag)
        loss = criterion(logits, label_tensor)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        prob = y_prob.detach().cpu().item()
        pred = int(prob >= 0.5)
        true = int(label_tensor.detach().cpu().item())
        train_probs.append(prob)
        train_preds.append(pred)
        train_labels.append(true)

        if pred != true:
            train_misclassified.append(normalize_slide_id(slide_id))

        pbar_train.set_postfix({"loss": f"{loss.item():.4f}", "prob": f"{prob:.3f}"})

    train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
    train_precision = precision_score(train_labels, train_preds, zero_division=0) if train_labels else 0.0
    train_recall = recall_score(train_labels, train_preds, zero_division=0) if train_labels else 0.0
    # roc_auc_score is undefined when only one class is present in y_true.
    # Check that both classes are present before calling roc_auc_score.
    if len(set(train_labels)) < 2:
        train_auc = float('nan')
    else:
        train_auc = roc_auc_score(train_labels, train_probs)

    model.eval()
    val_losses, val_probs, val_preds, val_labels = [], [], [], []
    val_misclassified = []
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Val", leave=False)
        for bag, label, slide_id in pbar_val:
            bag = bag.to(device)
            label_tensor = torch.as_tensor(label, dtype=torch.float32, device=device).view(1, 1)

            logits, y_prob, _, _ = model(bag)
            loss = criterion(logits, label_tensor)

            val_losses.append(loss.item())
            prob = y_prob.detach().cpu().item()
            pred = int(prob >= 0.5)
            true = int(label_tensor.detach().cpu().item())
            val_probs.append(prob)
            val_preds.append(pred)
            val_labels.append(true)

            if pred != true:
                val_misclassified.append(normalize_slide_id(slide_id))
            pbar_val.set_postfix({"loss": f"{loss.item():.4f}", "prob": f"{prob:.3f}"})

    val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
    val_precision = precision_score(val_labels, val_preds, zero_division=0) if val_labels else 0.0
    val_recall = recall_score(val_labels, val_preds, zero_division=0) if val_labels else 0.0
    # Same check for validation labels
    if len(set(val_labels)) < 2:
        val_auc = float('nan')
    else:
        val_auc = roc_auc_score(val_labels, val_probs)

    history.append({
        "epoch": epoch + 1,
        "train": {"loss": train_loss, "precision": train_precision, "recall": train_recall, "auc": train_auc},
        "val": {"loss": val_loss, "precision": val_precision, "recall": val_recall, "auc": val_auc},
    })
    misclassified_history.append({
        "epoch": epoch + 1,
        "train": train_misclassified,
        "val": val_misclassified,
    })

    def fmt(value):
        return f"{value:.4f}" if not (isinstance(value, float) and math.isnan(value)) else "nan"

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(
        f"  Train -> Loss: {fmt(train_loss)} | Precision: {fmt(train_precision)} | Recall: {fmt(train_recall)} | AUC: {fmt(train_auc)}"
    )
    print(
        f"  Val   -> Loss: {fmt(val_loss)} | Precision: {fmt(val_precision)} | Recall: {fmt(val_recall)} | AUC: {fmt(val_auc)}"
    )
    print(
        f"  Misclassified slides -> Train: {len(train_misclassified)} | Val: {len(val_misclassified)}"
    )

    # Checkpointing: save model + optimizer + history every `save_every` epochs when enabled
    if save_checkpoints and ((epoch + 1) % save_every == 0):
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "misclassified_history": misclassified_history,
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

history, misclassified_history



