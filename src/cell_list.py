"""Periodic two-dimensional cell list for short-range particle pairs.

Each cell is at least as wide as the search radius, so its own cell and the
eight periodic neighbors contain every pair within that radius. Returned
pairs are unique and sorted by particle index for deterministic reductions.
"""

from __future__ import annotations

import numpy as np

from geometry import as_box_array, mic_delta_np, wrap_points


def pairs_within_cutoff(
    points: np.ndarray, box_size, cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``i < j`` pairs and exact minimum-image distances <= cutoff.

    The input is wrapped into the periodic cell. At very large cutoffs there
    may be only one or two cells per axis; neighbor-cell deduplication still
    reports every particle pair exactly once.
    """
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be finite and positive")
    box = as_box_array(box_size)
    if np.any(~np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("box_size must contain finite positive lengths")
    x = wrap_points(points, box)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if len(x) < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=float)

    ncell = np.maximum(np.floor(box / float(cutoff)).astype(int), 1)
    cell_width = box / ncell
    cell_xy = np.minimum(np.floor(x / cell_width).astype(int), ncell - 1)
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, (cx, cy) in enumerate(cell_xy):
        buckets.setdefault((int(cx), int(cy)), []).append(index)

    pair_blocks: list[np.ndarray] = []
    for cell in sorted(buckets):
        left = np.asarray(buckets[cell], dtype=np.int64)
        neighbors = {
            ((cell[0] + dx) % int(ncell[0]), (cell[1] + dy) % int(ncell[1]))
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        }
        for neighbor in sorted(neighbors):
            if neighbor not in buckets or neighbor < cell:
                continue
            if neighbor == cell:
                if len(left) < 2:
                    continue
                ii, jj = np.triu_indices(len(left), k=1)
                candidates = np.column_stack((left[ii], left[jj]))
            else:
                right = np.asarray(buckets[neighbor], dtype=np.int64)
                candidates = np.column_stack(
                    (np.repeat(left, len(right)), np.tile(right, len(left)))
                )
                candidates.sort(axis=1)
            pair_blocks.append(candidates)

    if not pair_blocks:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=float)
    pairs = np.concatenate(pair_blocks)
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    delta = mic_delta_np(x[pairs[:, 0]] - x[pairs[:, 1]], box)
    distances = np.linalg.norm(delta, axis=-1)
    within = distances <= float(cutoff)
    return pairs[within], distances[within]


def minimum_image_distance(points: np.ndarray, box_size, initial_cutoff: float) -> float:
    """Find the global minimum pair distance by expanding a cell-list query."""
    box = as_box_array(box_size)
    maximum_possible = float(np.linalg.norm(box / 2.0))
    cutoff = min(float(initial_cutoff), maximum_possible)
    while True:
        _, distances = pairs_within_cutoff(points, box, cutoff)
        if distances.size:
            return float(np.min(distances))
        if cutoff >= maximum_possible:
            raise RuntimeError("no pair found in a configuration with at least two points")
        cutoff = min(2.0 * cutoff, maximum_possible)
