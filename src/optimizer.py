"""Two-stage optimizer: the final configuration is exactly the Stage-2 state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from config import deep_copy_config
from cell_list import minimum_image_distance, pairs_within_cutoff
from geometry import (
    as_box_array,
    mic_delta_np,
    particle_count_from_area_fraction,
    realized_area_fraction,
    wrap_points,
)
from descriptor_statistics import (
    TargetStatistics2D,
    ardf_loss,
    build_targets_from_reference,
    coordination_loss,
    cv_loss,
    displacement_regularization,
    nn_loss,
    overlap_loss,
    rdf_loss,
    voronoi_area_distribution_loss,
)


CELL_LIST_MIN_PARTICLES = 512


@dataclass
class OptimizationResult2D:
    points_init: np.ndarray
    points_stage1: np.ndarray
    points_stage2: np.ndarray
    points_final: np.ndarray
    points_target: np.ndarray
    box_size: np.ndarray
    radius: float
    diameter: float
    plot_radius: float
    plot_diameter: float
    radius_scale: float
    num_particles: int
    area_fraction: float
    statistics_area_fraction: float
    history: list[dict[str, Any]]
    stage_metrics: dict[str, Any]
    final_terms: dict[str, float]
    runtime_seconds: float


def resolve_geometry(config: dict[str, Any]) -> tuple[np.ndarray, int, float, float, float, float, float]:
    geom = config.get("geometry", {})
    box = as_box_array(geom.get("box_size", 1.0))
    if geom.get("radius") is not None:
        plot_radius = float(geom["radius"])
    elif geom.get("diameter") is not None:
        plot_radius = 0.5 * float(geom["diameter"])
    else:
        raise ValueError("geometry.radius or geometry.diameter is required")
    if plot_radius <= 0.0:
        raise ValueError("radius must be positive")
    radius_scale = float(
        geom.get(
            "statistics_radius_scale",
            geom.get("hard_core_radius_scale", geom.get("radius_scale", 1.0)),
        )
    )
    if radius_scale <= 0.0:
        raise ValueError("geometry.statistics_radius_scale must be positive")
    radius = plot_radius * radius_scale

    if geom.get("num_particles") is None:
        if geom.get("area_fraction") is None:
            raise ValueError("geometry.num_particles or geometry.area_fraction is required")
        num_particles = particle_count_from_area_fraction(box, plot_radius, float(geom["area_fraction"]))
    else:
        num_particles = int(geom["num_particles"])
    if num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    return box, num_particles, radius, 2.0 * radius, plot_radius, 2.0 * plot_radius, radius_scale


def random_points(seed: int, num_particles: int, box_size: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.random((int(num_particles), 2)) * box_size


def sobol_points(seed: int, num_particles: int, box_size: np.ndarray) -> np.ndarray:
    """Generate a scrambled two-dimensional Sobol initialization."""

    from scipy.stats import qmc

    n = int(num_particles)
    exponent = int(math.ceil(math.log2(max(n, 1))))
    unit_points = qmc.Sobol(d=2, scramble=True, seed=int(seed)).random_base2(exponent)
    return np.asarray(unit_points[:n], dtype=float) * as_box_array(box_size)


def thomas_points(
    seed: int,
    num_particles: int,
    box_size: np.ndarray,
    parent_count: int = 20,
    cluster_std: float = 0.35,
) -> np.ndarray:
    """Generate a fixed-count periodic Thomas-cluster initialization.

    Parent centers are uniform in the periodic cell.  Each of the prescribed
    number of particles selects a parent uniformly and receives an isotropic
    Gaussian displacement before periodic wrapping.
    """

    rng = np.random.default_rng(int(seed))
    box = as_box_array(box_size)
    n = int(num_particles)
    n_parents = int(parent_count)
    sigma = float(cluster_std)
    if n_parents <= 0:
        raise ValueError("initialization.parent_count must be positive")
    if sigma <= 0.0:
        raise ValueError("initialization.cluster_std must be positive")
    parents = rng.random((n_parents, 2)) * box
    parent_index = rng.integers(0, n_parents, size=n)
    offspring = parents[parent_index] + rng.normal(0.0, sigma, size=(n, 2))
    return wrap_points(offspring, box)


def regular_points(
    seed: int,
    num_particles: int,
    box_size: np.ndarray,
    diameter: float,
    jitter_fraction: float = 0.0,
) -> np.ndarray:
    """Generate a periodic rectangular-lattice initialization with optional vacancies."""

    rng = np.random.default_rng(int(seed))
    box = as_box_array(box_size)
    n = int(num_particles)
    min_spacing = float(diameter)
    max_nx = max(1, int(math.floor(box[0] / min_spacing)))
    max_ny = max(1, int(math.floor(box[1] / min_spacing)))
    candidates: list[tuple[float, int, int]] = []
    for nx in range(1, max_nx + 1):
        ny = int(math.ceil(n / nx))
        if ny > max_ny:
            continue
        dx = box[0] / nx
        dy = box[1] / ny
        aspect_error = abs(math.log((dx / dy) / (box[0] / box[1])))
        vacancy_penalty = (nx * ny - n) / max(n, 1)
        candidates.append((aspect_error + 0.25 * vacancy_penalty, nx, ny))
    if not candidates:
        raise ValueError("regular initialization cannot fit the requested particles without overlap")

    _, nx, ny = min(candidates, key=lambda item: item[0])
    xs = (np.arange(nx, dtype=float) + 0.5) * box[0] / nx
    ys = (np.arange(ny, dtype=float) + 0.5) * box[1] / ny
    grid = np.array([(x, y) for y in ys for x in xs], dtype=float)
    if grid.shape[0] > n:
        keep = np.sort(rng.choice(grid.shape[0], size=n, replace=False))
        grid = grid[keep]
    if jitter_fraction > 0.0:
        spacing = min(box[0] / nx, box[1] / ny)
        jitter = (rng.random(grid.shape) - 0.5) * float(jitter_fraction) * max(spacing - min_spacing, 0.0)
        grid = wrap_points(grid + jitter, box)
    return grid


def initial_points(config: dict[str, Any], seed: int, num_particles: int, box: np.ndarray, diameter: float) -> np.ndarray:
    init_cfg = config.get("initialization", {})
    source = str(init_cfg.get("source", "random_overlap")).lower()
    if source in {"random", "random_overlap", "overlapped_random", "poisson"}:
        return random_points(seed, num_particles, box)
    if source in {"sobol", "scrambled_sobol"}:
        return sobol_points(seed, num_particles, box)
    if source in {"thomas", "thomas_cluster"}:
        return thomas_points(
            seed,
            num_particles,
            box,
            parent_count=int(init_cfg.get("parent_count", 20)),
            cluster_std=float(init_cfg.get("cluster_std", 0.35)),
        )
    if source in {"regular", "grid", "lattice", "particle_regular"}:
        return regular_points(
            seed,
            num_particles,
            box,
            diameter + float(init_cfg.get("clearance", 0.0)),
            jitter_fraction=float(init_cfg.get("jitter_fraction", 0.0)),
        )
    raise ValueError("initialization.source must be poisson, sobol, thomas, or regular")


def _relax_overlaps_dense(
    points: np.ndarray,
    box_size: np.ndarray,
    diameter: float,
    alpha: float,
    tolerance: float,
    max_iters: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Original all-pairs update, faster than cell setup for small systems."""
    box = as_box_array(box_size)
    x = wrap_points(points, box)
    n = x.shape[0]
    converged = False
    final_potential = math.inf
    overlap_pairs = 0
    for iteration in range(int(max_iters) + 1):
        disp = mic_delta_np(x[:, None, :] - x[None, :, :], box)
        dist = np.linalg.norm(disp, axis=-1)
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        active = upper & (dist < float(diameter))
        overlap_pairs = int(np.count_nonzero(active))
        depth = np.maximum(float(diameter) - dist, 0.0) * active
        final_potential = 0.5 * float(np.sum(depth * depth))
        if final_potential <= float(tolerance) or overlap_pairs == 0:
            converged = True
            break
        direction = np.zeros_like(disp)
        safe_dist = np.maximum(dist, 1.0e-12)
        direction[active] = disp[active] / safe_dist[active][:, None]
        step = np.zeros_like(x)
        i_idx, j_idx = np.where(active)
        updates = 0.5 * float(alpha) * depth[active, None] * direction[active]
        np.add.at(step, i_idx, updates)
        np.add.at(step, j_idx, -updates)
        x = wrap_points(x + step, box)
    return x, {
        "iterations": int(iteration),
        "converged": bool(converged),
        "overlap_pairs": int(overlap_pairs),
        "overlap_potential": float(final_potential),
    }


def relax_overlaps(
    points: np.ndarray,
    box_size: np.ndarray,
    diameter: float,
    alpha: float = 0.35,
    tolerance: float = 1.0e-10,
    max_iters: int = 1000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Periodic overlap projection with an adaptive cell-list backend."""

    box = as_box_array(box_size)
    x = wrap_points(points, box)
    if len(x) < CELL_LIST_MIN_PARTICLES:
        return _relax_overlaps_dense(x, box, diameter, alpha, tolerance, max_iters)
    converged = False
    final_potential = math.inf
    overlap_pairs = 0

    for iteration in range(int(max_iters) + 1):
        pairs, distances = pairs_within_cutoff(x, box, diameter)
        active = distances < float(diameter)
        pairs, distances = pairs[active], distances[active]
        overlap_pairs = len(pairs)
        depth = np.maximum(float(diameter) - distances, 0.0)
        final_potential = 0.5 * float(np.sum(depth * depth))
        if final_potential <= float(tolerance) or overlap_pairs == 0:
            converged = True
            break
        i_idx, j_idx = pairs.T
        disp = mic_delta_np(x[i_idx] - x[j_idx], box)
        direction = disp / np.maximum(distances, 1.0e-12)[:, None]
        step = np.zeros_like(x)
        updates = 0.5 * float(alpha) * depth[:, None] * direction
        np.add.at(step, i_idx, updates)
        np.add.at(step, j_idx, -updates)
        x = wrap_points(x + step, box)

    return x, {
        "iterations": int(iteration),
        "converged": bool(converged),
        "overlap_pairs": int(overlap_pairs),
        "overlap_potential": float(final_potential),
    }


def _evaluate_feasibility_dense(points: np.ndarray, box_size: np.ndarray, diameter: float) -> dict[str, Any]:
    """Original exact all-pairs check for small systems."""
    x = np.asarray(points, dtype=np.float64)
    n = len(x)
    potential = 0.0
    min_distance = math.inf
    pairs = 0
    for first in range(0, n, 256):
        disp = mic_delta_np(x[first:first + 256, None, :] - x[None, :, :], box_size)
        distances = np.linalg.norm(disp, axis=-1)
        mask = np.arange(n)[None, :] > np.arange(first, min(first + 256, n))[:, None]
        values = distances[mask]
        if values.size:
            min_distance = min(min_distance, float(np.min(values)))
            depth = np.maximum(float(diameter) - values, 0.0)
            pairs += int(np.count_nonzero(depth))
            potential += 0.5 * float(np.sum(depth * depth))
    return {"min_center_distance": min_distance,
            "max_overlap": max(float(diameter) - min_distance, 0.0),
            "overlap_pairs": pairs, "overlap_potential": potential}


def evaluate_feasibility(points: np.ndarray, box_size: np.ndarray, diameter: float) -> dict[str, Any]:
    """Read-only exact minimum-image checks using a periodic cell list.

    Distances and overlap depths are in the supplied coordinate units; the
    overlap potential is in their square. No smoothing epsilon is applied.
    """
    x = wrap_points(points, box_size)
    if len(x) < 2:
        raise ValueError("at least two points are required for feasibility checks")
    if len(x) < CELL_LIST_MIN_PARTICLES:
        return _evaluate_feasibility_dense(x, box_size, diameter)
    _, distances = pairs_within_cutoff(x, box_size, diameter)
    depth = np.maximum(float(diameter) - distances, 0.0)
    potential = 0.5 * float(np.sum(depth * depth))
    pairs = int(np.count_nonzero(depth))
    min_distance = (
        float(np.min(distances)) if distances.size
        else minimum_image_distance(x, box_size, diameter)
    )
    return {"min_center_distance": min_distance,
            "max_overlap": max(float(diameter) - min_distance, 0.0),
            "overlap_pairs": pairs, "overlap_potential": potential}


def normalize_config(config: dict[str, Any], scalar_box: float) -> dict[str, Any]:
    out = deep_copy_config(config)
    rdf = out.setdefault("rdf", {})
    for key in ("r_min", "r_max", "sigma"):
        if rdf.get(key) is not None:
            rdf[key] = float(rdf[key]) / scalar_box
    ardf = out.setdefault("anisotropic_rdf", {})
    for key in ("r_min", "r_max", "sigma_r"):
        if ardf.get(key) is not None:
            ardf[key] = float(ardf[key]) / scalar_box
    nn = out.setdefault("nearest_neighbor", {})
    for key in ("tau", "grid_min", "grid_max", "bandwidth"):
        if nn.get(key) is not None:
            nn[key] = float(nn[key]) / scalar_box
    vor = out.setdefault("voronoi_area_distribution", {})
    area_scale = scalar_box * scalar_box
    for key in ("grid_min", "grid_max", "bandwidth"):
        if vor.get(key) is not None:
            vor[key] = float(vor[key]) / area_scale
    coord = out.setdefault("coordination", {})
    for key in ("cutoff_radius", "bandwidth"):
        if coord.get(key) is not None:
            coord[key] = float(coord[key]) / scalar_box
    return out


def phase2_terms(points, anchor_points, radius_nd: float, config: dict[str, Any], targets: TargetStatistics2D):
    loss_cfg = config.get("loss", {})
    phase2_cfg = config.get("phase2", {})
    rdf_cfg = config.get("rdf", {})
    ardf_cfg = config.get("anisotropic_rdf", {})
    nn_cfg = config.get("nearest_neighbor", {})
    vor_cfg = config.get("voronoi_area_distribution", {})
    cv_cfg = config.get("cv", {})
    coord_cfg = config.get("coordination", {})

    x = jnp.mod(points, 1.0)
    anchor = jnp.mod(anchor_points, 1.0)
    rdf = jnp.array(0.0)
    ardf = jnp.array(0.0)
    nn = jnp.array(0.0)
    vor = jnp.array(0.0)
    cv = jnp.array(0.0)
    coord = jnp.array(0.0)

    if rdf_cfg.get("enabled", True) and targets.rdf_grid is not None:
        rdf = rdf_loss(x, 1.0, targets.rdf_grid, targets.rdf_values, rdf_cfg["sigma"])
    if ardf_cfg.get("enabled", False) and targets.ardf_r_grid is not None and targets.ardf_theta_grid is not None:
        ardf = ardf_loss(
            x,
            1.0,
            targets.ardf_r_grid,
            targets.ardf_theta_grid,
            targets.ardf_values,
            ardf_cfg["sigma_r"],
            ardf_cfg["sigma_theta"],
        )
    if nn_cfg.get("enabled", True) and targets.nn_grid is not None:
        nn = nn_loss(x, 1.0, targets.nn_grid, targets.nn_values, nn_cfg["tau"], nn_cfg["bandwidth"])
    if vor_cfg.get("enabled", False) and targets.voronoi_area_grid is not None:
        vor = voronoi_area_distribution_loss(
            x,
            1.0,
            targets.voronoi_area_grid,
            targets.voronoi_area_values,
            vor_cfg.get("grid_resolution", cv_cfg.get("grid_resolution", 64)),
            vor_cfg.get("softmax_temperature", cv_cfg.get("softmax_temperature", 5.0e-4)),
            vor_cfg["bandwidth"],
        )
    if cv_cfg.get("enabled", True) and targets.cv_value is not None:
        cv = cv_loss(
            x,
            1.0,
            targets.cv_value,
            cv_cfg.get("grid_resolution", 64),
            cv_cfg.get("softmax_temperature", 5.0e-4),
            cv_cfg.get("eps", 1.0e-8),
        )
    if coord_cfg.get("enabled", True) and targets.coord_grid is not None:
        coord = coordination_loss(
            x,
            1.0,
            targets.coord_grid,
            targets.coord_values,
            coord_cfg["cutoff_radius"],
            coord_cfg["beta"],
            coord_cfg["bandwidth"],
            target_mean=targets.mean_coordination,
            use_mean=coord_cfg.get("use_mean_coordination_loss", True),
        )

    reg = displacement_regularization(x, anchor, 1.0)
    overlap = overlap_loss(x, 1.0, radius_nd)
    total = (
        float(loss_cfg.get("lambda_rdf", 0.2)) * rdf
        + float(loss_cfg.get("lambda_ardf", 0.2)) * ardf
        + float(loss_cfg.get("lambda_nn", 0.2)) * nn
        + float(loss_cfg.get("lambda_voronoi", 0.2)) * vor
        + float(loss_cfg.get("lambda_cv", 0.2)) * cv
        + float(loss_cfg.get("lambda_coord", 0.2)) * coord
        + float(loss_cfg.get("lambda_reg", 1.0e-3)) * reg
        + float(phase2_cfg.get("lambda_overlap_guard", 4.0)) * overlap
    )
    terms = {
        "loss_total": total,
        "loss_rdf": rdf,
        "loss_ardf": ardf,
        "loss_nn": nn,
        "loss_voronoi": vor,
        "loss_cv": cv,
        "loss_coord": coord,
        "loss_reg": reg,
        "loss_overlap": overlap,
    }
    return total, terms


def _target_reference(config: dict[str, Any], box: np.ndarray, num_particles: int, diameter: float) -> np.ndarray:
    target = config.get("target", {})
    source = str(target.get("source", "random")).lower()
    if source == "random":
        seed = int(target.get("seed", int(config.get("project", {}).get("seed", 0)) + 10000))
        points = random_points(seed, num_particles, box)
        phase1 = config.get("phase1", {})
        relaxed, _ = relax_overlaps(
            points,
            box,
            diameter + float(phase1.get("clearance", 0.0)),
            alpha=float(phase1.get("alpha", 0.35)),
            tolerance=float(phase1.get("tolerance", 1.0e-10)),
            max_iters=int(phase1.get("max_iters", 1600)),
        )
        return relaxed
    if source in {"file", "points_file"}:
        path = Path(target["points_path"])
        points = np.loadtxt(path, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        return wrap_points(points[:, :2], box)
    raise ValueError("target.source must be random or points_file")


def optimize_rve_2d(config: dict[str, Any], seed_offset: int = 0) -> OptimizationResult2D:
    start = time.perf_counter()
    project = config.get("project", {})
    box, num_particles, radius, diameter, plot_radius, plot_diameter, radius_scale = resolve_geometry(config)
    scalar_box = float(box[0])
    radius_nd = radius / scalar_box

    seed = int(project.get("seed", 0)) + int(seed_offset)
    points_init = initial_points(config, seed, num_particles, box, diameter)
    target_points = _target_reference(config, box, num_particles, diameter)
    norm_cfg = normalize_config(config, scalar_box)
    targets = build_targets_from_reference(jnp.asarray(target_points, dtype=jnp.float32) / scalar_box, norm_cfg)

    phase1 = config.get("phase1", {})
    stage1_start = time.perf_counter()
    points_stage1, stage1_metrics = relax_overlaps(
        points_init,
        box,
        diameter + float(phase1.get("clearance", 0.0)),
        alpha=float(phase1.get("alpha", 0.35)),
        tolerance=float(phase1.get("tolerance", 1.0e-10)),
        max_iters=int(phase1.get("max_iters", 1600)),
    )
    stage1_metrics["seconds"] = time.perf_counter() - stage1_start

    phase2 = config.get("phase2", {})
    max_iters = int(phase2.get("max_iters", 600))
    min_iters = int(phase2.get("min_iters", 0))
    learning_rate = float(phase2.get("learning_rate", 0.02))
    log_every = max(1, int(phase2.get("log_every", 25)))
    target_loss = phase2.get("target_loss")
    target_loss = None if target_loss is None else float(target_loss)
    relative_tolerance = phase2.get("relative_tolerance")
    relative_tolerance = None if relative_tolerance is None else float(relative_tolerance)
    relative_window = max(1, int(phase2.get("relative_window", 20)))
    relative_stop_max_loss = phase2.get("relative_stop_max_loss")
    relative_stop_max_loss = None if relative_stop_max_loss is None else float(relative_stop_max_loss)
    require_feasible = bool(phase2.get("require_feasible", True))
    overlap_tolerance = float(phase2.get("overlap_tolerance", 1.0e-12))
    max_overlap_tolerance = float(phase2.get("max_overlap_tolerance", 0.0))

    points = jnp.asarray(points_stage1, dtype=jnp.float32) / scalar_box
    anchor = points
    m = jnp.zeros_like(points)
    v = jnp.zeros_like(points)
    history: list[dict[str, Any]] = []

    def objective(local_points):
        return phase2_terms(local_points, anchor, radius_nd, norm_cfg, targets)

    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    if bool(phase2.get("use_jit", True)):
        value_and_grad = jax.jit(value_and_grad)

    stage2_start = time.perf_counter()
    total_history: list[float] = []
    stage2_stop_reason = "max_iters"
    stage2_converged = False
    for iteration in range(max_iters + 1):
        (total, terms), grad = value_and_grad(points)
        total_float = float(total)
        overlap_float = float(terms["loss_overlap"])
        feasible = overlap_float <= overlap_tolerance
        # Validate the actual exported coordinates, without the differentiable
        # distance estimator's epsilon and without changing the configuration.
        geometry_check = None
        if feasible:
            geometry_check = evaluate_feasibility(np.asarray(points) * scalar_box, box, diameter)
            feasible = (
                geometry_check["max_overlap"] <= max_overlap_tolerance
                and geometry_check["overlap_potential"] / scalar_box**2 <= overlap_tolerance
            )
        total_history.append(total_float)
        if iteration % log_every == 0 or iteration == max_iters:
            row = {"stage": "stage2", "iteration": iteration}
            row.update({name: float(value) for name, value in terms.items()})
            history.append(row)

        stopping_feasible = (not require_feasible) or feasible
        reached_target = target_loss is not None and iteration >= min_iters and total_float <= target_loss and stopping_feasible
        reached_relative = False
        if (
            relative_tolerance is not None
            and iteration >= min_iters
            and len(total_history) > relative_window
            and (relative_stop_max_loss is None or total_float <= relative_stop_max_loss)
            and stopping_feasible
        ):
            ref = total_history[-relative_window - 1]
            rel = (ref - total_float) / max(abs(ref), 1.0e-12)
            reached_relative = 0.0 <= rel < relative_tolerance
        if reached_target or reached_relative:
            stage2_stop_reason = "target_loss" if reached_target else "relative_tolerance"
            stage2_converged = True
            break
        if iteration == max_iters:
            break

        t = iteration + 1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1.0e-8
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        points = jnp.mod(points - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps), 1.0)

        project_every = int(phase2.get("project_every", 0))
        if project_every > 0 and t % project_every == 0:
            projected, _ = relax_overlaps(
                np.asarray(points) * scalar_box,
                box,
                diameter + float(phase2.get("projection_clearance", 0.0)),
                alpha=float(phase2.get("projection_alpha", 0.15)),
                tolerance=float(phase2.get("projection_tolerance", overlap_tolerance)),
                max_iters=int(phase2.get("projection_iters", 20)),
            )
            points = jnp.asarray(projected, dtype=jnp.float32) / scalar_box

    # A terminal feasibility projection belongs to Stage 2.  It preserves the
    # historical numerical operation while removing the misleading Stage-3
    # label: the projected coordinates are the terminal Stage-2 coordinates.
    points_stage2 = np.asarray(jnp.mod(points, 1.0)) * scalar_box
    terminal_projection = {
        "enabled": bool(phase2.get("terminal_projection", True)),
        "iterations": 0,
        "converged": True,
        "overlap_pairs": 0,
        "overlap_potential": 0.0,
    }
    if terminal_projection["enabled"]:
        points_stage2, projection_metrics = relax_overlaps(
            points_stage2,
            box,
            diameter + float(phase2.get("terminal_projection_clearance", 1.0e-4)),
            alpha=float(phase2.get("terminal_projection_alpha", 0.2)),
            tolerance=float(phase2.get("terminal_projection_tolerance", overlap_tolerance)),
            max_iters=int(phase2.get("terminal_projection_iters", 800)),
        )
        terminal_projection.update(projection_metrics)
    stage2_seconds = time.perf_counter() - stage2_start
    points_final = points_stage2.copy()
    geometry_check = evaluate_feasibility(points_final, box, diameter)
    feasible = (
        geometry_check["max_overlap"] <= max_overlap_tolerance
        and geometry_check["overlap_potential"] / scalar_box**2 <= overlap_tolerance
    )

    final_total, final_terms_raw = phase2_terms(
        jnp.asarray(points_final, dtype=jnp.float32) / scalar_box,
        anchor,
        radius_nd,
        norm_cfg,
        targets,
    )
    final_terms = {name: float(value) for name, value in final_terms_raw.items()}
    final_terms["loss_total"] = float(final_total)
    # Always record the terminal state, even when stopping between log intervals.
    terminal_row = {"stage": "stage2", "iteration": int(iteration), **final_terms}
    if history and history[-1]["iteration"] == iteration:
        history[-1] = terminal_row
    else:
        history.append(terminal_row)
    runtime = time.perf_counter() - start

    return OptimizationResult2D(
        points_init=wrap_points(points_init, box),
        points_stage1=wrap_points(points_stage1, box),
        points_stage2=wrap_points(points_stage2, box),
        points_final=wrap_points(points_final, box),
        points_target=wrap_points(target_points, box),
        box_size=box,
        radius=float(radius),
        diameter=float(diameter),
        plot_radius=float(plot_radius),
        plot_diameter=float(plot_diameter),
        radius_scale=float(radius_scale),
        num_particles=int(num_particles),
        area_fraction=realized_area_fraction(num_particles, plot_radius, box),
        statistics_area_fraction=realized_area_fraction(num_particles, radius, box),
        history=history,
        stage_metrics={
            "stage1": stage1_metrics,
            "stage2": {
                "iterations": int(iteration),
                "converged": bool(stage2_converged),
                "stop_reason": stage2_stop_reason,
                "seconds": float(stage2_seconds),
                "optimizer_stop_loss": float(total_history[-1]),
                "final_loss": float(final_total),
                "require_feasible": bool(require_feasible),
                "overlap_tolerance": float(overlap_tolerance),
                "final_overlap_loss": float(final_terms["loss_overlap"]),
                "feasible": bool(feasible),
                "accepted": bool(stage2_converged and feasible),
                "max_overlap_tolerance": max_overlap_tolerance,
                "geometry": geometry_check,
                "terminal_projection": terminal_projection,
            },
        },
        final_terms=final_terms,
        runtime_seconds=float(runtime),
    )


def save_result(result: OptimizationResult2D, output_dir: str | Path, index: int = 0) -> Path:
    if not np.array_equal(result.points_final, result.points_stage2):
        raise ValueError("Final coordinates must equal the terminal Stage-2 coordinates.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"rve_{index:04d}"
    npz_path = out / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        points_init=result.points_init,
        points_stage1=result.points_stage1,
        points_stage2=result.points_stage2,
        points_final=result.points_final,
        points_target=result.points_target,
        box_size=result.box_size,
        radius=np.asarray(result.radius),
        diameter=np.asarray(result.diameter),
        plot_radius=np.asarray(result.plot_radius),
        plot_diameter=np.asarray(result.plot_diameter),
        radius_scale=np.asarray(result.radius_scale),
        num_particles=np.asarray(result.num_particles),
        area_fraction=np.asarray(result.area_fraction),
        statistics_area_fraction=np.asarray(result.statistics_area_fraction),
    )
    summary = {
        "algorithm": "two_stage_final_equals_stage2",
        "accepted": bool(result.stage_metrics["stage2"].get("accepted", False)),
        "file": npz_path.name,
        "box_size": result.box_size.tolist(),
        "radius": result.radius,
        "diameter": result.diameter,
        "plot_radius": result.plot_radius,
        "plot_diameter": result.plot_diameter,
        "radius_scale": result.radius_scale,
        "num_particles": result.num_particles,
        "area_fraction": result.area_fraction,
        "statistics_area_fraction": result.statistics_area_fraction,
        "runtime_seconds": result.runtime_seconds,
        "stage_metrics": result.stage_metrics,
        "final_terms": result.final_terms,
        "history": result.history,
    }
    (out / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savetxt(out / f"{stem}_points_final.txt", result.points_final)
    return npz_path
