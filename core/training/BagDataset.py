from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np


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
        labels_path: Path,
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

