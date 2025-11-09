import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import os

from model import GatedAttentionMIL_RED, BASE_CHECKPOINT_CONFIG, SMALL_CHECKPOINT_CONFIG
from dataloader import ImageBagDataset
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

model = GatedAttentionMIL_RED(SMALL_CHECKPOINT_CONFIG).to(device)

train_dataset = ImageBagDataset("C:\\Code\\DL\\bbosis\\data\\small\\small_train.csv", train=True, model=model)
test_dataset = ImageBagDataset("C:\\Code\\DL\\bbosis\\data\\small\\small_test.csv", train=False, model=model)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_label_counts = np.bincount(train_dataset.labels, minlength=2)
num_negative, num_positive = train_label_counts[0], train_label_counts[1]
pos_weight_value = (num_negative / num_positive) if num_positive > 0 else 1.0
pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def best_threshold_by_f1(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds has len = len(precisions)-1; align
    f1s = (2*precisions[:-1]*recalls[:-1])/(precisions[:-1]+recalls[:-1]+1e-9)
    i = np.nanargmax(f1s) if len(f1s) else 0
    return thresholds[i] if len(thresholds) else 0.5, f1s[i] if len(f1s) else 0.0


def single_slide_collate(batch):
    """Ensure each batch contains a single slide bag without extra stacking."""
    return batch[0]

def normalize_slide_id(slide_id): 
    if isinstance(slide_id, (list, tuple)): 
        slide_id = slide_id[0] 
    if hasattr(slide_id, "item"): 
        slide_id = slide_id.item() 
    return str(slide_id)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                            collate_fn=single_slide_collate, num_workers=0)
val_loader   = DataLoader(test_dataset,   batch_size=1, shuffle=False,
                            collate_fn=single_slide_collate, num_workers=0)
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

    ## Threshold optimization based on the validation set
    val_ap = average_precision_score(val_labels, val_probs) if len(set(val_labels))==2 else float('nan')
    t_star, f1_star = best_threshold_by_f1(np.array(val_labels), np.array(val_probs))
    val_preds_opt = (np.array(val_probs) >= t_star).astype(int)
    val_prec_opt = precision_score(val_labels, val_preds_opt, zero_division=0)
    val_rec_opt  = recall_score(val_labels, val_preds_opt,  zero_division=0)

    print(f"  Val*  -> AP: {fmt(val_ap)} | best_th={t_star:.3f} | F1*: {f1_star:.4f} | P*: {fmt(val_prec_opt)} | R*: {fmt(val_rec_opt)}")

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
