import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
# from ..prototype.model import GatedAttentionMIL
try:
    from prototype.model import GatedAttentionMIL
except ImportError as exc:  # pragma: no cover - defensive guard in case layout changes
    raise SystemExit(
        "Unable to import GatedAttentionMIL from prototype.model. "
        "Ensure the repository root is on PYTHONPATH."
    ) from exc

def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_bags(batch: Sequence[Tuple[torch.Tensor, float, str]]) -> Tuple[List[torch.Tensor], torch.Tensor, List[str]]:
    bags, labels, bag_ids = zip(*batch)
    bags = [b.contiguous() for b in bags]
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    bag_ids_list = list(bag_ids)
    return bags, labels_tensor, bag_ids_list


def compute_classification_metrics(y_true: Sequence[float], y_prob: Sequence[float]) -> Dict[str, float]:
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "roc_auc": 0.0, "average_precision": 0.0}
    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    y_pred_arr = (y_prob_arr >= 0.5).astype(np.int32)

    metrics: Dict[str, float] = {}
    metrics["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))
    metrics["precision"] = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    if len(np.unique(y_true_arr)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
        metrics["average_precision"] = float(average_precision_score(y_true_arr, y_prob_arr))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    running_loss = 0.0
    sample_count = 0
    y_true: List[float] = []
    y_prob: List[float] = []

    progress = tqdm(loader, desc="train", leave=False)
    for bags, labels, _bag_ids in progress:
        optimizer.zero_grad(set_to_none=True)

        logits: List[torch.Tensor] = []
        probs: List[torch.Tensor] = []

        labels = labels.to(device)
        for bag in bags:
            bag = bag.to(device)
            logit, prob, _attention, _embedding = model(bag)
            logits.append(logit)
            probs.append(prob)

        logits_tensor = torch.stack(logits)
        probs_tensor = torch.stack(probs)

        loss = criterion(logits_tensor, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        running_loss += loss.item() * batch_size
        sample_count += batch_size
        y_true.extend(labels.detach().cpu().tolist())
        y_prob.extend(probs_tensor.detach().cpu().tolist())

        progress.set_postfix({"loss": loss.item()})

    avg_loss = running_loss / max(sample_count, 1)
    metrics = compute_classification_metrics(y_true, y_prob)
    return avg_loss, metrics


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], List[Dict[str, Any]]]:
    model.eval()
    running_loss = 0.0
    sample_count = 0
    y_true: List[float] = []
    y_prob: List[float] = []
    attention_records: List[Dict[str, Any]] = []

    for bags, labels, bag_ids in tqdm(loader, desc="eval", leave=False):
        logits: List[torch.Tensor] = []
        probs: List[torch.Tensor] = []
        attentions: List[torch.Tensor] = []

        labels = labels.to(device)
        for bag in bags:
            bag = bag.to(device)
            logit, prob, attention, _embedding = model(bag)
            logits.append(logit)
            probs.append(prob)
            attentions.append(attention.detach().cpu())

        logits_tensor = torch.stack(logits)
        probs_tensor = torch.stack(probs)
        loss = criterion(logits_tensor, labels)

        batch_size = labels.shape[0]
        running_loss += loss.item() * batch_size
        sample_count += batch_size

        labels_cpu = labels.detach().cpu().tolist()
        probs_cpu = probs_tensor.detach().cpu().tolist()
        y_true.extend(labels_cpu)
        y_prob.extend(probs_cpu)

        for bag_id, label_val, prob_val, attention in zip(bag_ids, labels_cpu, probs_cpu, attentions):
            attention_records.append(
                {
                    "bag_id": bag_id,
                    "label": float(label_val),
                    "probability": float(prob_val),
                    "attention": attention.clone(),
                }
            )

    avg_loss = running_loss / max(sample_count, 1)
    metrics = compute_classification_metrics(y_true, y_prob)
    return avg_loss, metrics, attention_records


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    val_loss: float,
    attention_records: List[Dict[str, Any]],
) -> None:
    ckpt_dir = output_dir / f"epoch_{epoch:03d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "val_loss": val_loss,
    }
    torch.save(checkpoint_payload, ckpt_dir / "model.pt")
    torch.save(attention_records, ckpt_dir / "attention_maps.pt")