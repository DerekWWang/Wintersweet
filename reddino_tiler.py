# pip install torch torchvision timm pillow opencv-python numpy tqdm
import os, argparse
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch, timm
from torchvision import transforms
import cv2

ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def list_images(root: Path):
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in ALLOWED])

def tile_image(img: Image.Image, tile_size: int, stride: int):
    w, h = img.size
    tiles, coords = [], []
    for y in range(0, max(h - tile_size + 1, 1), stride):
        for x in range(0, max(w - tile_size + 1, 1), stride):
            crop = img.crop((x, y, x + tile_size, y + tile_size))
            if crop.size != (tile_size, tile_size):  # pad edge tiles
                pad = Image.new(img.mode, (tile_size, tile_size), (255, 255, 255))
                pad.paste(crop, (0, 0))
                crop = pad
            tiles.append(crop)
            coords.append((x, y))
    return tiles, np.asarray(coords, dtype=np.int32)

def variance_of_laplacian(pil_img: Image.Image) -> float:
    arr = np.array(pil_img.convert("L"))
    return cv2.Laplacian(arr, cv2.CV_64F).var()

def foreground_fraction(pil_img: Image.Image, white_thresh: int = 240) -> float:
    gray = np.array(pil_img.convert("L"))
    return float((gray < white_thresh).sum()) / gray.size

def load_reddino(device: str):
    # Model card shows ViT-base, patch14, DINOv2-style, embedding size 768
    model = timm.create_model("hf_hub:Snarcy/RedDino-base", pretrained=True)  # outputs features
    model.eval().to(device)
    # Their example uses 224x224 + ImageNet mean/std
    tx = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    return model, tx

@torch.inference_mode()
def encode_tiles(tiles, tx, model, device: str, batch_size: int = 256) -> np.ndarray:
    feats = []
    for i in range(0, len(tiles), batch_size):
        batch = torch.stack([tx(t.convert("RGB")) for t in tiles[i:i+batch_size]]).to(device)
        out = model(batch)
        if isinstance(out, (list, tuple)): out = out[0]
        feats.append(out.detach().cpu().float().numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 768), dtype=np.float32)

def process_split(split_dir: Path, out_dir: Path, tile_size, stride, min_fg, min_sharp, batch_size, device):
    model, tx = load_reddino(device)
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in tqdm(list_images(split_dir), desc=f"{split_dir.name}", unit="img"):
        rel = img_path.relative_to(split_dir)
        out_path = out_dir / rel.with_suffix(".npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[skip] {img_path}: {e}"); continue

        tiles, coords = tile_image(img, tile_size, stride)

        keep = np.ones(len(tiles), dtype=bool)
        if min_fg > 0:
            fg = np.array([foreground_fraction(t) for t in tiles]); keep &= fg >= min_fg
        if min_sharp > 0:
            shp = np.array([variance_of_laplacian(t) for t in tiles]); keep &= shp >= min_sharp

        tiles = [tiles[i] for i in np.where(keep)[0]]
        coords = coords[keep]
        emb = encode_tiles(tiles, tx, model, device, batch_size)

        np.savez_compressed(out_path,
            embeddings=emb.astype(np.float32),
            coords=coords.astype(np.int32),
            meta=dict(slide_path=str(img_path), tile_size=tile_size, stride=stride,
                      encoder="Snarcy/RedDino-base", input_size=224,
                      norm="ImageNet"))
    print("Done:", split_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_root", default="reddino_feats")
    ap.add_argument("--tile_size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--min_foreground", type=float, default=0.07)
    ap.add_argument("--min_sharpness", type=float, default=10.0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    for split in ["training", "testing"]:
        split_dir = Path(args.data_root)/split
        if split_dir.exists():
            process_split(split_dir, Path(args.out_root)/split,
                          args.tile_size, args.stride,
                          args.min_foreground, args.min_sharpness,
                          args.batch_size, args.device)
        else:
            print(f"[warn] missing {split_dir}")

if __name__ == "__main__":
    main()
