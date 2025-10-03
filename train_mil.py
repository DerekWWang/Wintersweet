# pip install torch numpy scikit-learn tqdm
import argparse, json, random
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import average_precision_score
from tqdm import tqdm

# ---- Data utilities ----

def infer_label_from_path(p: Path):
    # Adjust to your layout. This infers from folder names like .../positive/slide.tif
    parent = p.parent.name.lower()
    if "pos" in parent or "positive" in parent: return 1
    if "neg" in parent or "negative" in parent: return 0
    raise ValueError(f"Cannot infer label from path: {p}")

def load_label_map(split_dir: Path):
    csv = split_dir/"labels.csv"
    if csv.exists():
        # Expected columns: relative_path,label
        m = {}
        for line in open(csv):
            rel, lab = line.strip().split(",")
            m[split_dir/rel] = int(lab)
        return m
    else:
        # Fallback: infer from parent folder names
        m = {}
        for npz in split_dir.rglob("*.npz"):
            try:
                m[npz] = infer_label_from_path(npz.parent)
            except Exception:
                continue
        if not m:
            raise RuntimeError("No labels.csv and could not infer labels from folder names.")
        return m

class BagDataset(Dataset):
    def __init__(self, root: Path, max_tiles_per_bag=512, seed=13):
        self.root = Path(root)
        self.label_map = load_label_map(self.root)
        self.slides = sorted(self.label_map.keys())
        self.labels = [self.label_map[p] for p in self.slides]
        self.rng = random.Random(seed)
        self.max_tiles = max_tiles_per_bag

    def __len__(self): return len(self.slides)

    def __getitem__(self, idx):
        p = self.slides[idx]
        lab = self.labels[idx]
        z = np.load(p)
        X = z["embeddings"].astype(np.float32)  # (N, 768)
        if X.shape[0] == 0:
            X = np.zeros((1, X.shape[1] if X.ndim==2 else 768), np.float32)
        # random subsample tiles per epoch to regularize & fit memory
        if X.shape[0] > self.max_tiles:
            sel = np.array(random.sample(range(X.shape[0]), self.max_tiles))
            X = X[sel]
        return torch.from_numpy(X), torch.tensor(lab, dtype=torch.float32)

# ---- ABMIL (gated attention) ----
class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim=768, hidden=512):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2)
        )
        self.att_a = nn.Linear(hidden, hidden)
        self.att_b = nn.Linear(hidden, hidden)
        self.att_w = nn.Linear(hidden, 1)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, X):  # X: (n_instances, in_dim)
        H = self.embed(X)  # (n, h)
        A = torch.tanh(self.att_a(H)) * torch.sigmoid(self.att_b(H))
        A = self.att_w(A)  # (n, 1)
        A = torch.softmax(A.squeeze(1), dim=0)  # (n,)
        M = torch.sum(H * A.unsqueeze(1), dim=0)  # bag embedding (h,)
        logit = self.classifier(M).squeeze(0)  # ()
        return logit, A  # return attention for inspection

# ---- Training ----
def train_epoch(model, loader, opt, loss_fn, device):
    model.train(); losses=[]
    for X, y in tqdm(loader, desc="train", leave=False):
        opt.zero_grad()
        X = [x.to(device) for x in X] if isinstance(X, list) else X.to(device)
        y = y.to(device)
        logits = []
        for i in range(X.shape[0]):  # batch of bags = stack of variable-length? here we pack as list, but simplest: batch_size=1
            logit, _ = model(X[i])
            logits.append(logit)
        logits = torch.stack(logits)  # (B,)
        loss = loss_fn(logits, y)
        loss.backward(); opt.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0

@torch.inference_mode()
def eval_epoch(model, loader, device):
    model.eval(); ys=[], ps=[]
    for X, y in tqdm(loader, desc="eval", leave=False):
        X = X.to(device); y = y.numpy()
        logit, _ = model(X[0]) if X.dim()==3 else model(X)  # if collate keeps batch size 1
        p = torch.sigmoid(logit).cpu().numpy().reshape(-1)
        ys.append(y); ps.append(p)
    y = np.concatenate(ys); p = np.concatenate(ps)
    ap = average_precision_score(y, p) if len(np.unique(y))>1 else 0.0
    return ap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats_train", default="reddino_feats/training")
    ap.add_argument("--feats_val", default="reddino_feats/testing")
    ap.add_argument("--max_tiles_per_bag", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1, help="bags per batch; keep 1 for simplicity")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["bce","focal"], default="bce")
    ap.add_argument("--pos_weight", type=float, default=None, help="e.g., 19.0 for 5% positives")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    train_ds = BagDataset(Path(args.feats_train), args.max_tiles_per_bag)
    val_ds   = BagDataset(Path(args.feats_val),   args.max_tiles_per_bag)

    # Oversample positives to balance batches
    counts = np.bincount([lab for lab in train_ds.labels])
    pos_w = args.pos_weight if args.pos_weight is not None else (counts[0]/max(counts[1],1))
    sampler = WeightedRandomSampler(
        weights=[(pos_w if lab==1 else 1.0) for lab in train_ds.labels],
        num_samples=len(train_ds),
        replacement=True
    )

    def collate(batch):
        # batch is list of (X, y); use batch size 1 by default to avoid padding
        Xs, ys = zip(*batch)
        # keep Xs as a 3D tensor [1, n_tiles, feat] so code path is consistent
        X = Xs[0].unsqueeze(0)
        y = torch.stack(ys)
        return X, y.squeeze(0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)

    device = args.device
    model = GatedAttentionMIL(in_dim=768, hidden=512).to(device)

    if args.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
    else:
        # simple focal loss
        class Focal(nn.Module):
            def __init__(self, alpha=0.75, gamma=2.0): super().__init__(); self.a=alpha; self.g=gamma
            def forward(self, logits, targets):
                p = torch.sigmoid(logits); eps=1e-8
                pt = p*targets + (1-p)*(1-targets)
                w = self.a*targets + (1-self.a)*(1-targets)
                return torch.mean(-w*(1-pt).pow(self.g)*torch.log(pt+eps))
        loss_fn = Focal()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_ap, best_path = -1, Path("best_mil.pth")
    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, opt, loss_fn, device)
        ap = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}  val_AP={ap:.4f}")
        if ap > best_ap:
            best_ap = ap
            torch.save({"model": model.state_dict(), "ap": ap}, best_path)

    print("Best AP:", best_ap, "saved to", best_path)

if __name__ == "__main__":
    main()
