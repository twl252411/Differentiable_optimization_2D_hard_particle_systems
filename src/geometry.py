"""Periodic 2D geometry utilities."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np


def as_box_array(box_size) -> np.ndarray:
    arr = np.asarray(box_size, dtype=float)
    if arr.ndim == 0:
        return np.asarray([float(arr), float(arr)], dtype=float)
    if arr.shape == (2,):
        return arr.astype(float)
    raise ValueError("box_size must be a scalar or a length-2 sequence")


def wrap_points(points: np.ndarray, box_size) -> np.ndarray:
    box = as_box_array(box_size)
    return np.mod(np.asarray(points, dtype=float), box)


def mic_delta_np(delta: np.ndarray, box_size) -> np.ndarray:
    box = as_box_array(box_size)
    return delta - box * np.round(delta / box)


def mic_delta_jax(delta, box_size):
    box = jnp.asarray(box_size, dtype=jnp.float32)
    return delta - box * jnp.round(delta / box)


def pairwise_distances_jax(points, box_size, eps: float = 1.0e-12):
    delta = points[:, None, :] - points[None, :, :]
    delta = mic_delta_jax(delta, box_size)
    return jnp.sqrt(jnp.sum(delta * delta, axis=-1) + eps)


def particle_count_from_area_fraction(box_size, radius: float, area_fraction: float) -> int:
    box = as_box_array(box_size)
    return int(math.ceil(float(area_fraction) * float(np.prod(box)) / (math.pi * radius * radius)))


def realized_area_fraction(num_particles: int, radius: float, box_size) -> float:
    box = as_box_array(box_size)
    return float(num_particles) * math.pi * radius * radius / float(np.prod(box))
