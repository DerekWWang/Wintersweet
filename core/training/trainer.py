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
from .BagDataset import EmbeddingBagDataset
from .ModelConfig import TrainingConfig
from prototype.model import GatedAttentionMIL
from .trainer_util import (
    set_seed,
    collate_bags,
    compute_classification_metrics,
    train_one_epoch,
    evaluate,
    save_checkpoint,
)

def train(config: TrainingConfig) -> None:
    set_seed(config.seed)

    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # first load core dataset (could be full or training split)
    dataset = EmbeddingBagDataset(
        embeddings_path=config.embeddings_path,
        labels_path=config.labels_path,
        max_tiles=config.max_tiles,
        seed=config.seed,
        deterministic_tiles=config.deterministic_tiles,
    )

    train_ds: Dataset = None
    val_ds: Dataset = None

    if config.val_embeddings_path and config.val_labels_path:
        train_ds = dataset

        val_ds = EmbeddingBagDataset(
            embeddings_path=config.val_embeddings_path,
            labels_path=config.val_labels_path,
            max_tiles=config.max_tiles,
            seed=config.seed,
            deterministic_tiles=True,
        )
    else:
        val_size = int(len(dataset) * config.val_split)
        train_size = len(dataset) - val_size
        if val_size == 0 or train_size == 0:
            raise ValueError("val_split results in an empty train or validation set. Adjust val_split or dataset size.")

        generator = torch.Generator().manual_seed(config.seed)
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_bags,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_bags,
    )

    model = GatedAttentionMIL(
        in_dim=dataset.feature_dim,
        hidden=config.hidden_dim,
        att_dim=config.att_dim,
        dropout=config.dropout,
    ).to(device)

    if config.pos_weight is not None:
        pos_weight_tensor = torch.tensor([config.pos_weight], dtype=torch.float32, device=device)
    else:
        pos_weight_tensor = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: List[Dict[str, Any]] = []
    best_val_auc = -float("inf")
    best_checkpoint_dir: Optional[Path] = None

    for epoch in range(1, config.epochs + 1):
        print(f"\nEpoch {epoch}/{config.epochs}")
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics, attention_records = evaluate(model, val_loader, criterion, device)

        print(
            f"  Train loss: {train_loss:.4f}  "
            f"accuracy: {train_metrics['accuracy']:.4f}  "
            f"precision: {train_metrics['precision']:.4f}  "
            f"roc_auc: {train_metrics['roc_auc']:.4f}  "
            f"avg_prec: {train_metrics['average_precision']:.4f}"
        )
        print(
            f"  Val   loss: {val_loss:.4f}  "
            f"accuracy: {val_metrics['accuracy']:.4f}  "
            f"precision: {val_metrics['precision']:.4f}  "
            f"roc_auc: {val_metrics['roc_auc']:.4f}  "
            f"avg_prec: {val_metrics['average_precision']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
        )

        should_checkpoint = (epoch % config.checkpoint_every == 0)
        val_auc = val_metrics.get("roc_auc", float("nan"))
        if np.isnan(val_auc):
            val_auc = -float("inf")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            should_checkpoint = True

        if should_checkpoint:
            save_checkpoint(output_dir, epoch, model, optimizer, train_metrics, val_metrics, val_loss, attention_records)
            best_checkpoint_dir = output_dir / f"epoch_{epoch:03d}"

    metrics_path = output_dir / "training_history.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    if best_checkpoint_dir:
        print(f"Best checkpoint based on ROC-AUC stored in {best_checkpoint_dir}")
    print(f"Training history written to {metrics_path}")