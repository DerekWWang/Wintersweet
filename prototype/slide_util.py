# Slide utility functions
from typing import List

import numpy as np


def slide_to_tiles(slide: np.ndarray, tile_size: int, stride: int, limit: int) -> List[np.ndarray]:
    """
    Converts a slide into a list of tiles of size tile_size x tile_size.

    Args:
        slide (np.ndarray): Slide array with shape (height, width, channels).
        tile_size (int): Size of the square tile.
        stride (int): Step size between tile origins. Use 0 for non-overlapping tiling.

    Returns:
        List[np.ndarray]: List of tiles, each with shape (tile_size, tile_size, channels).
    """
    if slide.ndim != 3:
        raise ValueError("slide must have shape (height, width, channels)")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if stride < 0:
        raise ValueError("stride must be non-negative")

    effective_stride = tile_size if stride == 0 else stride

    height, width, _ = slide.shape
    tiles: List[np.ndarray] = []

    # Iterate over valid top-left corners that yield full tiles.
    for y in range(0, height - tile_size + 1, effective_stride):
        for x in range(0, width - tile_size + 1, effective_stride):
            tile = slide[y : y + tile_size, x : x + tile_size, :]
            tiles.append(tile)

    return tiles if limit <= 0 else tiles[:limit]