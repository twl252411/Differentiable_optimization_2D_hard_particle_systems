"""Differentiable periodic statistics and losses for 2D disk RVEs."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from geometry import mic_delta_jax, pairwise_distances_jax


@dataclass(frozen=True)
class TargetStatistics2D:
    rdf_grid: jnp.ndarray | None = None
    rdf_values: jnp.ndarray | None = None
    ardf_r_grid: jnp.ndarray | None = None
    ardf_theta_grid: jnp.ndarray | None = None
    ardf_values: jnp.ndarray | None = None
    nn_grid: jnp.ndarray | None = None
    nn_values: jnp.ndarray | None = None
    voronoi_area_grid: jnp.ndarray | None = None
    voronoi_area_values: jnp.ndarray | None = None
    cv_value: jnp.ndarray | None = None
    coord_grid: jnp.ndarray | None = None
    coord_values: jnp.ndarray | None = None
    mean_coordination: jnp.ndarray | None = None


def upper_pair_distances(points, box_size):
    distances = pairwise_distances_jax(points, box_size)
    n = distances.shape[0]
    i, j = jnp.triu_indices(n, k=1)
    return distances[i, j]


def upper_pair_vectors(points, box_size):
    delta = points[:, None, :] - points[None, :, :]
    delta = mic_delta_jax(delta, box_size)
    n = delta.shape[0]
    i, j = jnp.triu_indices(n, k=1)
    return delta[i, j]


def normalized_kde(values, grid, bandwidth: float, eps: float = 1.0e-8):
    values = jnp.asarray(values, dtype=jnp.float32)
    grid = jnp.asarray(grid, dtype=jnp.float32)
    kernel = jnp.exp(-0.5 * ((values[:, None] - grid[None, :]) / float(bandwidth)) ** 2)
    density = jnp.mean(kernel, axis=0)
    return density / (jnp.sum(density) + eps)


def overlap_loss(points, box_size, radius: float):
    distances = pairwise_distances_jax(points, box_size)
    n = distances.shape[0]
    mask = 1.0 - jnp.eye(n, dtype=distances.dtype)
    overlap = jnp.maximum(2.0 * float(radius) - distances, 0.0) * mask
    return 0.5 * jnp.sum(jnp.triu(overlap * overlap, k=1))


def displacement_regularization(points, anchor_points, box_size):
    delta = mic_delta_jax(points - anchor_points, box_size)
    return jnp.mean(jnp.sum(delta * delta, axis=-1))


def rdf_grid(r_min: float, r_max: float, num_bins: int):
    return jnp.linspace(float(r_min), float(r_max), int(num_bins), dtype=jnp.float32)


def smooth_rdf(points, box_size, grid, sigma: float, shell_normalize: bool = True, eps: float = 1.0e-8):
    distances = upper_pair_distances(points, box_size)
    grid = jnp.asarray(grid, dtype=jnp.float32)
    kernel = jnp.exp(-0.5 * ((distances[:, None] - grid[None, :]) / float(sigma)) ** 2)
    counts = jnp.sum(kernel, axis=0) / (distances.shape[0] + eps)
    if shell_normalize:
        counts = counts / (2.0 * jnp.pi * jnp.maximum(grid, sigma) + eps)
    return counts / (jnp.sum(counts) + eps)


def rdf_loss(points, box_size, grid, target_values, sigma: float):
    pred = smooth_rdf(points, box_size, grid, sigma)
    target = jnp.asarray(target_values, dtype=jnp.float32)
    return jnp.mean((pred - target) ** 2)


def ardf_grids(r_min: float, r_max: float, num_r_bins: int, num_theta_bins: int):
    r_grid = jnp.linspace(float(r_min), float(r_max), int(num_r_bins), dtype=jnp.float32)
    theta_grid = jnp.linspace(0.0, jnp.pi, int(num_theta_bins), endpoint=False, dtype=jnp.float32)
    return r_grid, theta_grid


def angular_distance_pi(angles, grid):
    """Smallest signed angular difference for axial angles with period pi."""

    diff = angles[:, None] - grid[None, :]
    return 0.5 * jnp.arctan2(jnp.sin(2.0 * diff), jnp.cos(2.0 * diff))


def smooth_anisotropic_rdf(
    points,
    box_size,
    r_grid,
    theta_grid,
    sigma_r: float,
    sigma_theta: float,
    shell_normalize: bool = True,
    eps: float = 1.0e-8,
):
    vectors = upper_pair_vectors(points, box_size)
    distances = jnp.sqrt(jnp.sum(vectors * vectors, axis=-1) + eps)
    angles = jnp.mod(jnp.arctan2(vectors[:, 1], vectors[:, 0]), jnp.pi)
    r_grid = jnp.asarray(r_grid, dtype=jnp.float32)
    theta_grid = jnp.asarray(theta_grid, dtype=jnp.float32)
    radial_kernel = jnp.exp(-0.5 * ((distances[:, None] - r_grid[None, :]) / float(sigma_r)) ** 2)
    angular_delta = angular_distance_pi(angles, theta_grid)
    angular_kernel = jnp.exp(-0.5 * (angular_delta / float(sigma_theta)) ** 2)
    density = jnp.einsum("pr,pt->rt", radial_kernel, angular_kernel) / (distances.shape[0] + eps)
    if shell_normalize:
        density = density / (2.0 * jnp.pi * jnp.maximum(r_grid[:, None], sigma_r) + eps)
    return density / (jnp.sum(density) + eps)


def ardf_loss(points, box_size, r_grid, theta_grid, target_values, sigma_r: float, sigma_theta: float):
    pred = smooth_anisotropic_rdf(points, box_size, r_grid, theta_grid, sigma_r, sigma_theta)
    target = jnp.asarray(target_values, dtype=jnp.float32)
    return jnp.mean((pred - target) ** 2)


def nn_grid(grid_min: float, grid_max: float, num_grid: int):
    return jnp.linspace(float(grid_min), float(grid_max), int(num_grid), dtype=jnp.float32)


def soft_nearest_neighbor_distances(points, box_size, tau: float):
    distances = pairwise_distances_jax(points, box_size)
    n = distances.shape[0]
    masked = distances + jnp.eye(n, dtype=distances.dtype) * 1.0e6
    return -float(tau) * jax.nn.logsumexp(-masked / float(tau), axis=1)


def smooth_nn_distribution(points, box_size, grid, tau: float, bandwidth: float):
    soft_nn = soft_nearest_neighbor_distances(points, box_size, tau)
    return normalized_kde(soft_nn, grid, bandwidth)


def nn_loss(points, box_size, grid, target_values, tau: float, bandwidth: float):
    pred = smooth_nn_distribution(points, box_size, grid, tau, bandwidth)
    target = jnp.asarray(target_values, dtype=jnp.float32)
    return jnp.mean((pred - target) ** 2)


def soft_voronoi_areas(points, box_size, grid_resolution: int, softmax_temperature: float):
    """Differentiable periodic Voronoi-cell areas by soft grid quadrature."""

    box = jnp.asarray(box_size, dtype=jnp.float32)
    if box.ndim == 0:
        box = jnp.asarray([box, box], dtype=jnp.float32)
    res = int(grid_resolution)
    axis_x = (jnp.arange(res, dtype=jnp.float32) + 0.5) * box[0] / float(res)
    axis_y = (jnp.arange(res, dtype=jnp.float32) + 0.5) * box[1] / float(res)
    yy, xx = jnp.meshgrid(axis_y, axis_x, indexing="ij")
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    delta = grid_points[:, None, :] - points[None, :, :]
    delta = delta - box * jnp.round(delta / box)
    squared_distances = jnp.sum(delta * delta, axis=-1)
    weights = jax.nn.softmax(-squared_distances / float(softmax_temperature), axis=1)
    cell_area = jnp.prod(box) / float(res * res)
    return jnp.sum(weights, axis=0) * cell_area


def voronoi_area_coefficient_of_variation(
    points,
    box_size,
    grid_resolution: int,
    softmax_temperature: float,
    eps: float = 1.0e-8,
):
    areas = soft_voronoi_areas(points, box_size, grid_resolution, softmax_temperature)
    mean = jnp.mean(areas)
    std = jnp.sqrt(jnp.mean((areas - mean) ** 2) + float(eps))
    return std / (mean + float(eps))


def cv_loss(points, box_size, target_cv, grid_resolution: int, softmax_temperature: float, eps: float = 1.0e-8):
    pred_cv = voronoi_area_coefficient_of_variation(points, box_size, grid_resolution, softmax_temperature, eps)
    return (pred_cv - jnp.asarray(target_cv, dtype=jnp.float32)) ** 2


def voronoi_area_grid(grid_min: float, grid_max: float, num_grid: int):
    return jnp.linspace(float(grid_min), float(grid_max), int(num_grid), dtype=jnp.float32)


def smooth_voronoi_area_distribution(
    points,
    box_size,
    grid,
    grid_resolution: int,
    softmax_temperature: float,
    bandwidth: float,
):
    areas = soft_voronoi_areas(points, box_size, grid_resolution, softmax_temperature)
    return normalized_kde(areas, grid, bandwidth)


def voronoi_area_distribution_loss(
    points,
    box_size,
    grid,
    target_values,
    grid_resolution: int,
    softmax_temperature: float,
    bandwidth: float,
):
    pred = smooth_voronoi_area_distribution(points, box_size, grid, grid_resolution, softmax_temperature, bandwidth)
    target = jnp.asarray(target_values, dtype=jnp.float32)
    return jnp.mean((pred - target) ** 2)


def coord_grid(grid_min: float, grid_max: float, num_grid: int):
    return jnp.linspace(float(grid_min), float(grid_max), int(num_grid), dtype=jnp.float32)


def smooth_coordination_numbers(points, box_size, cutoff_radius: float, beta: float):
    distances = pairwise_distances_jax(points, box_size)
    n = distances.shape[0]
    contact = jax.nn.sigmoid(float(beta) * (float(cutoff_radius) - distances))
    contact = contact * (1.0 - jnp.eye(n, dtype=contact.dtype))
    return jnp.sum(contact, axis=1)


def smooth_coordination_distribution(points, box_size, grid, cutoff_radius: float, beta: float, bandwidth: float):
    values = smooth_coordination_numbers(points, box_size, cutoff_radius, beta)
    return normalized_kde(values, grid, bandwidth)


def coordination_loss(
    points,
    box_size,
    grid,
    target_values,
    cutoff_radius: float,
    beta: float,
    bandwidth: float,
    target_mean=None,
    use_mean: bool = True,
):
    pred = smooth_coordination_distribution(points, box_size, grid, cutoff_radius, beta, bandwidth)
    target = jnp.asarray(target_values, dtype=jnp.float32)
    loss = jnp.mean((pred - target) ** 2)
    if use_mean and target_mean is not None:
        values = smooth_coordination_numbers(points, box_size, cutoff_radius, beta)
        loss = loss + (jnp.mean(values) - jnp.asarray(target_mean, dtype=jnp.float32)) ** 2
    return loss


def build_targets_from_reference(points, config: dict) -> TargetStatistics2D:
    rdf_cfg = config.get("rdf", {})
    ardf_cfg = config.get("anisotropic_rdf", {})
    nn_cfg = config.get("nearest_neighbor", {})
    vor_cfg = config.get("voronoi_area_distribution", {})
    cv_cfg = config.get("cv", {})
    coord_cfg = config.get("coordination", {})

    rdf_g = rdf_grid(rdf_cfg["r_min"], rdf_cfg["r_max"], rdf_cfg["num_bins"]) if rdf_cfg.get("enabled", True) else None
    ardf_r_g, ardf_theta_g = (
        ardf_grids(ardf_cfg["r_min"], ardf_cfg["r_max"], ardf_cfg["num_r_bins"], ardf_cfg["num_theta_bins"])
        if ardf_cfg.get("enabled", False)
        else (None, None)
    )
    nn_g = nn_grid(nn_cfg["grid_min"], nn_cfg["grid_max"], nn_cfg["num_grid"]) if nn_cfg.get("enabled", True) else None
    vor_g = (
        voronoi_area_grid(vor_cfg["grid_min"], vor_cfg["grid_max"], vor_cfg["num_grid"])
        if vor_cfg.get("enabled", False)
        else None
    )
    coord_g = (
        coord_grid(coord_cfg["grid_min"], coord_cfg["grid_max"], coord_cfg["num_grid"])
        if coord_cfg.get("enabled", True)
        else None
    )
    coord_numbers = (
        smooth_coordination_numbers(points, 1.0, coord_cfg["cutoff_radius"], coord_cfg["beta"])
        if coord_cfg.get("enabled", True)
        else None
    )

    return TargetStatistics2D(
        rdf_grid=rdf_g,
        rdf_values=None if rdf_g is None else smooth_rdf(points, 1.0, rdf_g, rdf_cfg["sigma"]),
        ardf_r_grid=ardf_r_g,
        ardf_theta_grid=ardf_theta_g,
        ardf_values=(
            None
            if ardf_r_g is None or ardf_theta_g is None
            else smooth_anisotropic_rdf(
                points,
                1.0,
                ardf_r_g,
                ardf_theta_g,
                ardf_cfg["sigma_r"],
                ardf_cfg["sigma_theta"],
            )
        ),
        nn_grid=nn_g,
        nn_values=None if nn_g is None else smooth_nn_distribution(points, 1.0, nn_g, nn_cfg["tau"], nn_cfg["bandwidth"]),
        voronoi_area_grid=vor_g,
        voronoi_area_values=(
            None
            if vor_g is None
            else smooth_voronoi_area_distribution(
                points,
                1.0,
                vor_g,
                vor_cfg.get("grid_resolution", cv_cfg.get("grid_resolution", 64)),
                vor_cfg.get("softmax_temperature", cv_cfg.get("softmax_temperature", 5.0e-4)),
                vor_cfg["bandwidth"],
            )
        ),
        cv_value=(
            None
            if not cv_cfg.get("enabled", True)
            else voronoi_area_coefficient_of_variation(
                points,
                1.0,
                cv_cfg.get("grid_resolution", 64),
                cv_cfg.get("softmax_temperature", 5.0e-4),
                cv_cfg.get("eps", 1.0e-8),
            )
        ),
        coord_grid=coord_g,
        coord_values=(
            None
            if coord_g is None
            else smooth_coordination_distribution(
                points,
                1.0,
                coord_g,
                coord_cfg["cutoff_radius"],
                coord_cfg["beta"],
                coord_cfg["bandwidth"],
            )
        ),
        mean_coordination=None if coord_numbers is None else jnp.mean(coord_numbers),
    )
