from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch


@dataclass
class TrainingConfig:
    embeddings_path: Path = Path("data/tensors/embeddings.pt")
    labels_path: Path = Path("data/tensors/labels.pt")

    val_embeddings_path: Optional[Path] = None
    val_labels_path: Optional[Path] = None
    
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

