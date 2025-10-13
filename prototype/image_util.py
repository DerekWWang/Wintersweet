from typing import Tuple
from PIL import Image, ImageOps
import numpy as np

def image_colors_img(img_path: str = "", top_k: int = 10):
    """
    Analyzes an image file to find the most common colors given a image path.
    Args:
        img_path (str): Path to the image file.
        top_k (int): Number of least common colors to return.
    """
    im = ImageOps.exif_transpose(Image.open(img_path)).convert("RGBA")
    arr = np.asarray(im)                         # (H, W, 4) uint8
    pixels = arr.reshape(-1, arr.shape[-1])      # (H*W, 4)

    colors, counts = np.unique(pixels, axis=0, return_counts=True)

    # Example: bottom k most common colors (uncommon colors are better as slides have large background areas)
    top_idx = np.argsort(counts)[:top_k]
    for c, n in zip(colors[top_idx], counts[top_idx]):
        r, g, b, a = map(int, c)
        print(f"({r},{g},{b},{a}) -> {n}")
    return colors[top_idx], counts[top_idx]


def image_colors_np(arr: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyzes a numpy array representing an image to find the least common colors.
    Args:
        arr (np.ndarray): Image array with shape (H, W, 3) or (H, W, 4).
        top_k (int): Number of least common colors to return.
    """
    if arr.ndim != 3 or arr.shape[2] not in [3, 4]:
        raise ValueError("Input array must have shape (H, W, 3) or (H, W, 4)")

    pixels = arr.reshape(-1, arr.shape[-1])      # (H*W, C)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)

    top_idx = np.argsort(counts)[:top_k]
    return colors[top_idx], counts[top_idx]
