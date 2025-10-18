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


def _to_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.tensor(x)


def _extract_bag_entry(entry: Any) -> Tuple[torch.Tensor, int, Optional[str]]:
    """
    Normalize raw entries loaded from embeddings.pt into (bag, label, bag_id).
    Accepts multiple common layouts for flexibility.
    """
    bag_id: Optional[str] = None

    if isinstance(entry, dict):
        bag = entry.get("bag") or entry.get("features") or entry.get("embeddings")
        label = entry.get("label")
        bag_id = entry.get("id") or entry.get("slide_id") or entry.get("name")
        if bag is None or label is None:
            raise ValueError("Dictionary entry must contain 'bag' (or aliases) and 'label' keys.")
        return _to_tensor(bag), int(label), bag_id

    if isinstance(entry, (list, tuple)):
        if len(entry) < 2:
            raise ValueError("Iterable entries must contain at least (bag, label).")
        bag, label, *rest = entry
        if rest:
            bag_id = rest[0]
        return _to_tensor(bag), int(label), bag_id

    raise TypeError(f"Unsupported entry type inside embeddings file: {type(entry)}")


@dataclass
class BagRecord:
    bag: torch.Tensor
    label: int
    bag_id: str


class EmbeddingBagDataset(Dataset):
    """
    Dataset backed by a serialized embeddings.pt file.

    Assumes each entry stores a bag of tile embeddings and an associated label.
    When bags contain more than `max_tiles`, they are randomly down-sampled to `max_tiles`
    during each __getitem__ call to encourage tile diversity over epochs.
    """

    def __init__(
        self,
        embeddings_path: Path,
        labels_path: Optional[Path] = None,
        max_tiles: int = 120,
        seed: Optional[int] = None,
        deterministic_tiles: bool = False,
    ):
        self.embeddings_path = Path(embeddings_path)
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found at {self.embeddings_path}")

        raw_entries = torch.load(self.embeddings_path, map_location="cpu")
        if labels_path is not None:
            labels_tensor = torch.load(labels_path, map_location="cpu")
            if len(labels_tensor) != len(raw_entries):
                raise ValueError("labels.pt and embeddings.pt contain different number of entries.")
            raw_entries = [(bag, int(label)) for bag, label in zip(raw_entries, labels_tensor)]

        records: List[BagRecord] = []
        for idx, entry in enumerate(raw_entries):
            bag, label, bag_id = _extract_bag_entry(entry)
            bag = bag.detach().cpu().float()
            if bag.dim() != 2:
                raise ValueError(f"Bag at index {idx} must be a 2D tensor, found shape {tuple(bag.shape)}.")
            if bag_id is None:
                bag_id = f"bag_{idx:05d}"
            records.append(BagRecord(bag=bag, label=label, bag_id=str(bag_id)))

        self.records = records
        self.max_tiles = max_tiles
        self.deterministic_tiles = deterministic_tiles
        self._rng = torch.Generator().manual_seed(seed if seed is not None else torch.seed())

        if not self.records:
            raise ValueError("No bags were loaded from the embeddings file.")
        self.feature_dim = self.records[0].bag.shape[1]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, str]:
        record = self.records[idx]
        bag = record.bag
        if bag.shape[0] > self.max_tiles:
            if self.deterministic_tiles:
                perm = torch.arange(self.max_tiles)
            else:
                perm = torch.randperm(bag.shape[0], generator=self._rng)[: self.max_tiles]
            bag = bag[perm]
        return bag.clone(), float(record.label), record.bag_id


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


@dataclass
class TrainingConfig:
    embeddings_path: Path = Path("data/tensors/embeddings.pt")
    labels_path: Optional[Path] = None
    output_dir: Path = Path("runs/mil")
    epochs: int = 20
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_tiles: int = 120
    val_split: float = 0.2
    checkpoint_every: int = 5
    num_workers: int = 0
    seed: int = 42
    deterministic_tiles: bool = False
    hidden_dim: int = 512
    att_dim: int = 512
    dropout: float = 0.2
    pos_weight: Optional[float] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


DEFAULT_CONFIG = TrainingConfig()


def main(config: TrainingConfig = DEFAULT_CONFIG) -> None:
    set_seed(config.seed)

    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = EmbeddingBagDataset(
        embeddings_path=config.embeddings_path,
        labels_path=config.labels_path,
        max_tiles=config.max_tiles,
        seed=config.seed,
        deterministic_tiles=config.deterministic_tiles,
    )

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


if __name__ == "__main__":
    main()
