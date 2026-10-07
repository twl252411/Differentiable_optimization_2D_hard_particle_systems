# file: fiber_pipeline.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology, segmentation
from skimage.feature import peak_local_max, blob_log
from skimage.filters import gaussian, threshold_sauvola


# ==========================
# High-recall defaults (tune here)
# ==========================
SHOW_RESULT_DEFAULT = True

# segmentation cleanup (lower => fewer misses, more false positives)
MIN_OBJECT_AREA = 60
FILL_HOLE_AREA = 120

# split knobs
SPLIT_MIN_DIST_SCALE_1 = 0.80
SPLIT_MIN_DIST_SCALE_2 = 0.45
CLUMP_AREA_RATIO = 1.8
MAX_RADIUS_RATIO = 1.6  # allow larger clumps; later validated/handled
WATERSHED_COMPACTNESS = 0.0
USE_CLUMP_EXTRA_PEAKS = True
CLUMP_EXTRA_MIN_DIST_SCALE = 0.22
CLUMP_EXTRA_MIN_PEAK_SCALE = 0.25

# binarization controls (stricter masks reduce false centers in gaps)
BINARY_USE_AND = True           # AND local and global thresholds
BINARY_GLOBAL_DELTA = -0.01     # global threshold offset relative to Otsu
CLOSING_SCALE = 0.08            # morph closing radius = scale * est_r
DIST_GAUSS_SIGMA = 0.4          # distance transform smoothing (smaller => more splits)
EROSION_SCALE = 0.0             # optional erosion for peak mask (0 disables)
USE_LOOSE_MASK = True           # watershed mask uses loose (OR) binary
USE_STRICT_PEAK_MASK = True     # peaks use strict (AND) binary

# intensity peak markers (to split connected fibers)
USE_INTENSITY_PEAKS = True
INTENSITY_PEAK_MIN_DIST_SCALE = 0.45
INTENSITY_PEAK_DELTA = 0.02     # require peak > Otsu + delta
PEAK_MERGE_MIN_DIST_SCALE = 0.35

# distance peaks (mask weak peaks to avoid gap centers)
USE_DISTANCE_PEAKS = True
DIST_PEAK_MIN_SCALE = 0.30

# blob (LoG) markers for connected fibers
USE_BLOB_MARKERS = True
BLOB_MIN_SIGMA_SCALE = 0.60      # relative to est_r/sqrt(2)
BLOB_MAX_SIGMA_SCALE = 1.40
BLOB_THRESHOLD = 0.02
BLOB_OVERLAP = 0.5
BLOB_MIN_R_SCALE = 0.80
BLOB_MAX_R_SCALE = 1.30

# h-maxima peaks on composite distance (helps split touching fibers)
USE_H_MAXIMA = True
H_MAXIMA_SCALE = 0.12            # h = scale * est_r
DIST_INTENSITY_ALPHA = 0.45      # add brightness to distance (0 disables)

# validation (relaxed: keep if (edge ok) OR (contrast ok))
EDGE_KEEP_PERCENTILE = 10         # keep top 90% by edge score (lower => keep more)
EDGE_ABS_MIN = 6.0                # absolute floor for edge score (uint8 grad)
CONTRAST_KEEP_PERCENTILE = 25     # keep top 75% by contrast (lower => keep more)
CONTRAST_ABS_MIN = 3.0            # absolute floor (uint8 intensity)
VALIDATION_USE_OR = True
REQUIRE_POSITIVE_CONTRAST = True  # reject candidates darker than background
CENTER_INTENSITY_MIN_DELTA = 0.015 # center must be brighter than Otsu + delta (0 disables)
CENTER_INTENSITY_WINDOW = 3       # odd window size for center mean

# circle fitting
MIN_CONTOUR_POINTS = 10
MIN_RADIUS_RATIO = 0.85       # reject tiny circles relative to median radius
BOUNDARY_RELAX_MIN_RADIUS_RATIO = 0.60  # allow smaller radii if region touches image border

# Hough completion
ENABLE_HOUGH_COMPLETION = True
HOUGH_DP = 1.2
HOUGH_PARAM1 = 120
HOUGH_PARAM2 = 16          # lower => more circles (more false positives)
HOUGH_MIN_R_SCALE = 0.85
HOUGH_MAX_R_SCALE = 1.25
HOUGH_MIN_DIST_SCALE = 1.6

# relaxed Hough completion for dim fibers missed by segmentation/watershed
ENABLE_RELAXED_HOUGH_COMPLETION = True
RELAXED_HOUGH_DP = 1.2
RELAXED_HOUGH_PARAM1 = 100
RELAXED_HOUGH_PARAM2 = 14
RELAXED_HOUGH_MIN_R_SCALE = 0.75
RELAXED_HOUGH_MAX_R_SCALE = 1.35
RELAXED_HOUGH_MIN_DIST_SCALE = 0.85
RELAXED_HOUGH_EXISTING_DIST_SCALE = 0.70
RELAXED_HOUGH_EDGE_MIN = 6.0
RELAXED_HOUGH_CONTRAST_MIN = 3.0

# non-overlap shrink (shrink larger circle)
SHRINK_ENABLE = True
SHRINK_GAP_SCALE = 0.02
SHRINK_MIN_RADIUS_SCALE = 0.85
SHRINK_MAX_ITER = 80
FINAL_SHRINK_ENABLE = True
FINAL_SHRINK_GAP_SCALE = 0.02
FINAL_SHRINK_MIN_RADIUS_SCALE = 0.12
FINAL_SHRINK_MAX_ITER = 120

# detection mode and NMS
DETECTION_MODE = "watershed"  # "watershed", "blob", "hybrid"
NMS_ENABLE = True
NMS_MIN_DIST_SCALE = 0.50

# boundary handling
ENABLE_REFLECT_PADDING = True
REFLECT_PADDING_SCALE = 2.25
REFLECT_PADDING_MIN = 24
REFLECT_PADDING_MAX_FRACTION = 0.12
KEEP_CENTER_MARGIN_SCALE = 0.15


@dataclass(frozen=True)
class Params:
    show_result: bool = SHOW_RESULT_DEFAULT
    boundary_margin: float = 1.5
    use_reflect_padding: bool = ENABLE_REFLECT_PADDING
    use_relaxed_completion: bool = ENABLE_RELAXED_HOUGH_COMPLETION


# ==========================
# Preprocess
# ==========================
def preprocess_gray_u8(gray_u8: np.ndarray) -> np.ndarray:
    """
    High recall preprocessing:
      - CLAHE for local contrast
      - white top-hat to enhance bright fibers
      - mild smoothing
    """
    if gray_u8.dtype != np.uint8:
        g = gray_u8.astype(np.float32)
        g -= g.min()
        denom = g.max() - g.min()
        gray_u8 = (255.0 * (g / denom if denom > 0 else g)).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray_u8)

    # top-hat (bright objects on darker background)
    k = max(9, int(round(min(g.shape) / 40)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel)

    # blend to avoid over-enhancement
    out = cv2.addWeighted(g, 0.7, tophat, 0.6, 0.0)
    out = cv2.GaussianBlur(out, (0, 0), 1.0)
    return out


def sobel_magnitude_u8(gray_u8: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)


def estimate_particle_radius(gray_u8: np.ndarray) -> float:
    """Estimate the typical fiber radius with the same Otsu-area logic used by watershed_split."""

    g = preprocess_gray_u8(gray_u8)
    g_f = g.astype(np.float32) / 255.0
    th0 = filters.threshold_otsu(g_f)
    binary0 = g_f > th0
    binary0 = morphology.remove_small_objects(binary0, MIN_OBJECT_AREA)
    binary0 = morphology.remove_small_holes(binary0, FILL_HOLE_AREA)
    if not np.any(binary0):
        return 0.0
    cc = measure.label(binary0)
    areas = np.array([r.area for r in measure.regionprops(cc)], dtype=np.float32)
    if areas.size == 0:
        return 0.0
    med_area = float(np.median(areas))
    return float(np.sqrt(max(med_area, 1.0) / np.pi))


def reflect_pad_for_boundary(gray_u8: np.ndarray, est_r: float) -> Tuple[np.ndarray, int]:
    """Pad the image by reflection so truncated edge particles are processed like interior particles."""

    if est_r <= 0:
        est_r = max(1.0, min(gray_u8.shape) / 80.0)
    pad = int(round(max(REFLECT_PADDING_MIN, REFLECT_PADDING_SCALE * est_r)))
    pad_max = int(round(REFLECT_PADDING_MAX_FRACTION * min(gray_u8.shape)))
    pad = max(0, min(pad, pad_max))
    if pad <= 0:
        return gray_u8, 0
    return cv2.copyMakeBorder(gray_u8, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101), pad


# ==========================
# Geometry helpers
# ==========================
def fit_circle_kasa(points_xy: np.ndarray) -> Optional[Tuple[float, float, float]]:
    if points_xy.shape[0] < MIN_CONTOUR_POINTS:
        return None

    x = points_xy[:, 0].astype(np.float64)
    y = points_xy[:, 1].astype(np.float64)

    a = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy, c0 = sol
    r2 = cx * cx + cy * cy + c0
    if not np.isfinite(r2) or r2 <= 0:
        return None
    return float(cx), float(cy), float(np.sqrt(r2))


def circle_edge_score(grad_u8: np.ndarray, cx: float, cy: float, r: float, n: int = 64) -> float:
    h, w = grad_u8.shape
    if r <= 1:
        return 0.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    xs = np.clip(np.round(cx + r * np.cos(angles)).astype(int), 0, w - 1)
    ys = np.clip(np.round(cy + r * np.sin(angles)).astype(int), 0, h - 1)
    return float(np.median(grad_u8[ys, xs]))


def refine_radius_by_edge(grad_u8: np.ndarray, cx: float, cy: float, r: float) -> Tuple[float, float]:
    scales = np.array([0.88, 0.94, 1.00, 1.06, 1.12], dtype=np.float32)
    best_r = r
    best_s = -1.0
    for s in scales:
        rr = float(r * s)
        sc = circle_edge_score(grad_u8, cx, cy, rr)
        if sc > best_s:
            best_s = sc
            best_r = rr
    return best_r, float(best_s)


def circle_contrast_score(gray_u8: np.ndarray, cx: float, cy: float, r: float, n: int = 48) -> float:
    h, w = gray_u8.shape
    if r <= 2:
        return 0.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rin = max(1.0, 0.45 * r)
    rout = 1.30 * r

    xin = np.clip(np.round(cx + rin * np.cos(angles)).astype(int), 0, w - 1)
    yin = np.clip(np.round(cy + rin * np.sin(angles)).astype(int), 0, h - 1)

    xout = np.clip(np.round(cx + rout * np.cos(angles)).astype(int), 0, w - 1)
    yout = np.clip(np.round(cy + rout * np.sin(angles)).astype(int), 0, h - 1)

    inside = gray_u8[yin, xin].astype(np.float32)
    outside = gray_u8[yout, xout].astype(np.float32)
    return float(np.mean(inside) - np.mean(outside))


# ==========================
# Segmentation + watershed
# ==========================
def watershed_split(gray_u8: np.ndarray) -> Tuple[np.ndarray, float]:
    g = preprocess_gray_u8(gray_u8)
    g_f = g.astype(np.float32) / 255.0

    # estimate radius using easy Otsu mask (not final)
    th0 = filters.threshold_otsu(g_f)
    binary0 = g_f > th0
    binary0 = morphology.remove_small_objects(binary0, MIN_OBJECT_AREA)
    binary0 = morphology.remove_small_holes(binary0, FILL_HOLE_AREA)

    if not np.any(binary0):
        return np.zeros_like(gray_u8, dtype=np.int32), 0.0

    cc = measure.label(binary0)
    areas = np.array([r.area for r in measure.regionprops(cc)], dtype=np.float32)
    med_area = float(np.median(areas)) if areas.size else 0.0
    est_r = float(np.sqrt(max(med_area, 1.0) / np.pi))

    # robust binary = Sauvola OR slightly relaxed Otsu
    win = int(max(15, round(4.0 * est_r)))
    if win % 2 == 0:
        win += 1
    th_local = threshold_sauvola(g_f, window_size=win, k=0.18)
    binary_local = g_f > th_local
    binary_otsu_relax = g_f > max(0.0, th0 + BINARY_GLOBAL_DELTA)
    binary_strict = binary_local & binary_otsu_relax
    binary_loose = binary_local | binary_otsu_relax
    if BINARY_USE_AND:
        binary = binary_strict
    else:
        binary = binary_loose

    # light morphological closing to connect weak boundaries (improves recall)
    se = morphology.disk(max(1, int(round(CLOSING_SCALE * est_r))))
    binary = morphology.binary_closing(binary, se)

    binary = morphology.remove_small_objects(binary, MIN_OBJECT_AREA)
    binary = morphology.remove_small_holes(binary, FILL_HOLE_AREA)

    if USE_LOOSE_MASK:
        binary_mask = morphology.binary_closing(binary_loose, se)
        binary_mask = morphology.remove_small_objects(binary_mask, MIN_OBJECT_AREA)
        binary_mask = morphology.remove_small_holes(binary_mask, FILL_HOLE_AREA)
    else:
        binary_mask = binary

    if not np.any(binary_mask):
        return np.zeros_like(gray_u8, dtype=np.int32), est_r

    peak_mask = binary_strict if USE_STRICT_PEAK_MASK else binary_mask
    if EROSION_SCALE > 0 and est_r > 1.0:
        se_er = morphology.disk(max(1, int(round(EROSION_SCALE * est_r))))
        peak_mask = morphology.binary_erosion(peak_mask, se_er)

    dist = ndi.distance_transform_edt(binary_mask).astype(np.float32)
    dist = gaussian(dist, sigma=DIST_GAUSS_SIGMA, preserve_range=True)

    def merge_peaks(a: np.ndarray, b: np.ndarray, min_dist: int) -> np.ndarray:
        if a.size == 0:
            return b
        if b.size == 0:
            return a
        keep = [tuple(p) for p in a]
        min_d2 = float(min_dist * min_dist)
        for p in b:
            py, px = float(p[0]), float(p[1])
            if all((py - q[0]) ** 2 + (px - q[1]) ** 2 >= min_d2 for q in keep):
                keep.append((int(py), int(px)))
        return np.array(keep, dtype=np.int32)

    def collect_peaks(min_dist: int) -> np.ndarray:
        peaks = np.empty((0, 2), dtype=np.int32)
        if USE_DISTANCE_PEAKS:
            min_peak = float(DIST_PEAK_MIN_SCALE * est_r)
            peak_mask_dist = peak_mask & (dist >= min_peak)
            peaks = peak_local_max(
                dist,
                min_distance=max(1, int(min_dist)),
                labels=peak_mask_dist.astype(np.uint8),
                exclude_border=False,
            )
        if USE_H_MAXIMA:
            h = max(0.1, float(H_MAXIMA_SCALE * est_r))
            if DIST_INTENSITY_ALPHA > 0:
                boost = np.clip(g_f - th0, 0.0, None) * (DIST_INTENSITY_ALPHA * est_r)
                dist_comp = dist + boost
            else:
                dist_comp = dist
            hmask = morphology.h_maxima(dist_comp, h)
            hmask = hmask & peak_mask
            peaks_h = np.column_stack(np.nonzero(hmask)).astype(np.int32)
            merge_dist = max(1, int(round(PEAK_MERGE_MIN_DIST_SCALE * est_r)))
            peaks = merge_peaks(peaks, peaks_h, merge_dist)
        if USE_INTENSITY_PEAKS:
            thr = float(th0 + INTENSITY_PEAK_DELTA)
            peaks_i = peak_local_max(
                g_f,
                min_distance=max(1, int(round(INTENSITY_PEAK_MIN_DIST_SCALE * est_r))),
                labels=peak_mask.astype(np.uint8),
                threshold_abs=thr,
                exclude_border=False,
            )
            merge_dist = max(1, int(round(PEAK_MERGE_MIN_DIST_SCALE * est_r)))
            peaks = merge_peaks(peaks, peaks_i, merge_dist)
        if USE_BLOB_MARKERS and est_r > 1.0:
            sigma0 = float(est_r / np.sqrt(2.0))
            min_sigma = max(1.0, BLOB_MIN_SIGMA_SCALE * sigma0)
            max_sigma = max(min_sigma + 0.5, BLOB_MAX_SIGMA_SCALE * sigma0)
            g_eq = exposure.equalize_adapthist(g_f, clip_limit=0.02)
            blobs = blob_log(
                g_eq,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=10,
                threshold=BLOB_THRESHOLD,
                overlap=BLOB_OVERLAP,
            )
            if blobs.size:
                ys = blobs[:, 0].astype(int)
                xs = blobs[:, 1].astype(int)
                rs = np.sqrt(2.0) * blobs[:, 2]
                keep = (
                    (ys >= 0)
                    & (ys < gray_u8.shape[0])
                    & (xs >= 0)
                    & (xs < gray_u8.shape[1])
                    & peak_mask[ys, xs]
                    & (rs >= BLOB_MIN_R_SCALE * est_r)
                    & (rs <= BLOB_MAX_R_SCALE * est_r)
                )
                peaks_b = np.stack([ys[keep], xs[keep]], axis=1) if np.any(keep) else np.empty((0, 2), np.int32)
                merge_dist = max(1, int(round(PEAK_MERGE_MIN_DIST_SCALE * est_r)))
                peaks = merge_peaks(peaks, peaks_b, merge_dist)
        return peaks

    def watershed_from_peaks(peaks: np.ndarray) -> np.ndarray:
        if peaks.size == 0:
            return np.zeros_like(gray_u8, dtype=np.int32)
        markers = np.zeros_like(gray_u8, dtype=np.int32)
        markers[peaks[:, 0], peaks[:, 1]] = np.arange(1, peaks.shape[0] + 1, dtype=np.int32)
        return segmentation.watershed(
            -dist,
            markers,
            mask=binary_mask,
            compactness=float(WATERSHED_COMPACTNESS),
        )

    def run_ws(min_dist: int) -> Tuple[np.ndarray, np.ndarray]:
        peaks = collect_peaks(min_dist)
        labels = watershed_from_peaks(peaks)
        return labels, peaks

    labels1, peaks1 = run_ws(max(2, int(round(SPLIT_MIN_DIST_SCALE_1 * est_r))))
    regs1 = measure.regionprops(labels1)
    if not regs1:
        return np.zeros_like(gray_u8, dtype=np.int32), est_r

    areas1 = np.array([r.area for r in regs1], dtype=np.float32)
    med1 = float(np.median(areas1)) if areas1.size else med_area
    clump_mask = np.zeros_like(labels1, dtype=bool)
    if USE_CLUMP_EXTRA_PEAKS:
        area_thr = CLUMP_AREA_RATIO * med1
        for r in regs1:
            if r.area > area_thr:
                clump_mask[labels1 == r.label] = True

    clump = bool(np.any(clump_mask))

    labels = labels1
    best_count = int(labels1.max())

    if clump:
        labels2, _ = run_ws(max(1, int(round(SPLIT_MIN_DIST_SCALE_2 * est_r))))
        if int(labels2.max()) > best_count:
            labels = labels2
            best_count = int(labels2.max())

        if USE_CLUMP_EXTRA_PEAKS:
            min_dist = max(1, int(round(CLUMP_EXTRA_MIN_DIST_SCALE * est_r)))
            min_peak = float(CLUMP_EXTRA_MIN_PEAK_SCALE * est_r)
            peak_mask_clump = peak_mask & clump_mask & (dist >= min_peak)
            peaks_extra = peak_local_max(
                dist,
                min_distance=min_dist,
                labels=peak_mask_clump.astype(np.uint8),
                exclude_border=False,
            )
            if peaks_extra.size:
                merge_dist = max(1, int(round(PEAK_MERGE_MIN_DIST_SCALE * est_r)))
                peaks_merged = merge_peaks(peaks1, peaks_extra, merge_dist)
                labels3 = watershed_from_peaks(peaks_merged)
                if int(labels3.max()) > best_count:
                    labels = labels3
                    best_count = int(labels3.max())

    return labels, est_r


def circles_from_labels(gray_u8: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if labels.max() <= 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    g = preprocess_gray_u8(gray_u8)
    g_f = g.astype(np.float32) / 255.0
    th0 = filters.threshold_otsu(g_f)
    grad = sobel_magnitude_u8(g)

    regions = measure.regionprops(labels)
    areas = np.array([r.area for r in regions], dtype=np.float32)
    if areas.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    med_area = float(np.median(areas))
    med_r = float(np.sqrt(max(med_area, 1.0) / np.pi))
    max_r_allowed = MAX_RADIUS_RATIO * med_r
    min_r_allowed = max(1.0, MIN_RADIUS_RATIO * med_r)
    min_r_boundary = max(1.0, BOUNDARY_RELAX_MIN_RADIUS_RATIO * med_r)
    h, w = gray_u8.shape

    candidates: list[Tuple[float, float, float, float, float]] = []

    for reg in regions:
        r_area = float(np.sqrt(reg.area / np.pi))
        touches_border = (reg.bbox[0] <= 0) or (reg.bbox[1] <= 0) or (reg.bbox[2] >= h) or (reg.bbox[3] >= w)
        if not touches_border:
            if r_area < min_r_allowed or r_area > max_r_allowed:
                continue
        else:
            if r_area > max_r_allowed:
                continue

        mask = (labels == reg.label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        pts = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(np.float32)

        fit = fit_circle_kasa(pts)
        if fit is None:
            # fallback: centroid + area radius (better recall)
            cy, cx = reg.centroid
            fit = (float(cx), float(cy), r_area)

        cx, cy, rr = fit
        min_r_use = min_r_boundary if touches_border else min_r_allowed
        if rr < min_r_use or rr > max_r_allowed * 1.3:
            continue

        if CENTER_INTENSITY_MIN_DELTA > 0:
            win = int(CENTER_INTENSITY_WINDOW)
            if win % 2 == 0:
                win += 1
            hw = win // 2
            x0 = max(0, int(round(cx)) - hw)
            x1 = min(g.shape[1], int(round(cx)) + hw + 1)
            y0 = max(0, int(round(cy)) - hw)
            y1 = min(g.shape[0], int(round(cy)) + hw + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            center_mean = float(np.mean(g_f[y0:y1, x0:x1]))
            if center_mean < (th0 + CENTER_INTENSITY_MIN_DELTA):
                continue

        rr, edge_s = refine_radius_by_edge(grad, cx, cy, rr)
        contrast_s = circle_contrast_score(g, cx, cy, rr)
        candidates.append((cx, cy, rr, edge_s, contrast_s))

    if not candidates:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    cand = np.array(candidates, dtype=np.float32)
    edge_scores = cand[:, 3]
    contrast_scores = cand[:, 4]

    edge_thr = max(EDGE_ABS_MIN, float(np.percentile(edge_scores, EDGE_KEEP_PERCENTILE)))
    pos_contrast = contrast_scores[contrast_scores > 0]
    if pos_contrast.size:
        con_thr = max(CONTRAST_ABS_MIN, float(np.percentile(pos_contrast, CONTRAST_KEEP_PERCENTILE)))
    else:
        con_thr = CONTRAST_ABS_MIN

    edge_ok = edge_scores >= edge_thr
    contrast_ok = contrast_scores >= con_thr
    if REQUIRE_POSITIVE_CONTRAST:
        contrast_ok &= contrast_scores > 0

    if VALIDATION_USE_OR:
        keep = edge_ok | contrast_ok
    else:
        keep = edge_ok & contrast_ok

    cand = cand[keep]
    if cand.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    centers = cand[:, 0:2].astype(np.float32)
    radii = cand[:, 2].astype(np.float32)
    return centers, radii


def circles_from_blob_log(gray_u8: np.ndarray, est_r: float) -> Tuple[np.ndarray, np.ndarray]:
    if not USE_BLOB_MARKERS or est_r <= 1.0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    g = preprocess_gray_u8(gray_u8)
    g_f = g.astype(np.float32) / 255.0
    th0 = filters.threshold_otsu(g_f)
    g_eq = exposure.equalize_adapthist(g_f, clip_limit=0.02)
    grad = sobel_magnitude_u8(g)

    sigma0 = float(est_r / np.sqrt(2.0))
    min_sigma = max(1.0, BLOB_MIN_SIGMA_SCALE * sigma0)
    max_sigma = max(min_sigma + 0.5, BLOB_MAX_SIGMA_SCALE * sigma0)

    blobs = blob_log(
        g_eq,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=10,
        threshold=BLOB_THRESHOLD,
        overlap=BLOB_OVERLAP,
    )
    if blobs.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    centers: list[Tuple[float, float]] = []
    radii: list[float] = []

    for y, x, s in blobs:
        r0 = float(np.sqrt(2.0) * s)
        if r0 < BLOB_MIN_R_SCALE * est_r or r0 > BLOB_MAX_R_SCALE * est_r:
            continue
        if CENTER_INTENSITY_MIN_DELTA > 0:
            w = int(CENTER_INTENSITY_WINDOW)
            if w % 2 == 0:
                w += 1
            hw = w // 2
            x0 = max(0, int(round(x)) - hw)
            x1 = min(g.shape[1], int(round(x)) + hw + 1)
            y0 = max(0, int(round(y)) - hw)
            y1 = min(g.shape[0], int(round(y)) + hw + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            center_mean = float(np.mean(g_f[y0:y1, x0:x1]))
            if center_mean < (th0 + CENTER_INTENSITY_MIN_DELTA):
                continue
        rr, edge_s = refine_radius_by_edge(grad, float(x), float(y), r0)
        contrast_s = circle_contrast_score(g, float(x), float(y), rr)

        edge_ok = edge_s >= EDGE_ABS_MIN
        contrast_ok = contrast_s >= CONTRAST_ABS_MIN
        if REQUIRE_POSITIVE_CONTRAST:
            contrast_ok &= contrast_s > 0

        keep = (edge_ok | contrast_ok) if VALIDATION_USE_OR else (edge_ok & contrast_ok)
        if not keep:
            continue
        centers.append((float(x), float(y)))
        radii.append(float(rr))

    if not centers:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    return np.asarray(centers, np.float32), np.asarray(radii, np.float32)


def score_candidates(gray_u8: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    if centers.size == 0 or radii.size == 0:
        return np.empty((0,), np.float32)
    g = preprocess_gray_u8(gray_u8)
    grad = sobel_magnitude_u8(g)
    scores = np.empty((len(radii),), np.float32)
    for i, ((x, y), r) in enumerate(zip(centers, radii)):
        edge_s = circle_edge_score(grad, float(x), float(y), float(r))
        contrast_s = circle_contrast_score(g, float(x), float(y), float(r))
        scores[i] = float(edge_s + 0.5 * contrast_s)
    return scores


def nms_circles(
    centers: np.ndarray,
    radii: np.ndarray,
    scores: np.ndarray,
    min_dist_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if centers.size == 0:
        return centers, radii
    order = np.argsort(scores)[::-1]
    keep = []
    suppressed = np.zeros(len(order), dtype=bool)
    for i_idx, i in enumerate(order):
        if suppressed[i_idx]:
            continue
        keep.append(i)
        ci = centers[i]
        ri = radii[i]
        for j_idx in range(i_idx + 1, len(order)):
            if suppressed[j_idx]:
                continue
            j = order[j_idx]
            cj = centers[j]
            rj = radii[j]
            dx = float(ci[0] - cj[0])
            dy = float(ci[1] - cj[1])
            d = (dx * dx + dy * dy) ** 0.5
            if d < min_dist_scale * (ri + rj):
                suppressed[j_idx] = True
    keep = np.array(keep, dtype=int)
    return centers[keep], radii[keep]


# ==========================
# Hough completion (relaxed)
# ==========================
def complete_missing_fibers(gray_u8: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not ENABLE_HOUGH_COMPLETION:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)
    if centers.size == 0 or radii.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    g = preprocess_gray_u8(gray_u8)
    grad = sobel_magnitude_u8(g)

    base_r = float(np.median(radii))
    if not np.isfinite(base_r) or base_r <= 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    occupied = np.zeros_like(g, dtype=np.uint8)
    for (x, y), r in zip(centers, radii):
        cv2.circle(occupied, (int(round(x)), int(round(y))), int(round(1.03 * r)), 255, -1)

    search = cv2.bitwise_and(g, g, mask=cv2.bitwise_not(occupied))

    circles = cv2.HoughCircles(
        search,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=float(HOUGH_MIN_DIST_SCALE * base_r),
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=max(1, int(round(HOUGH_MIN_R_SCALE * base_r))),
        maxRadius=max(2, int(round(HOUGH_MAX_R_SCALE * base_r))),
    )
    if circles is None:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    new_c: list[list[float]] = []
    new_r: list[float] = []

    for x, y, r in circles[0].astype(np.float32):
        d = np.linalg.norm(centers - np.array([x, y], np.float32), axis=1)
        if np.any(d < (radii + r) * 0.95):
            continue

        rr, edge_s = refine_radius_by_edge(grad, float(x), float(y), float(r))
        contrast_s = circle_contrast_score(g, float(x), float(y), rr)

        # relaxed OR validation, but reject negative contrast if enabled
        if REQUIRE_POSITIVE_CONTRAST and contrast_s <= 0:
            continue
        if (edge_s < EDGE_ABS_MIN) and (contrast_s < CONTRAST_ABS_MIN):
            continue

        new_c.append([float(x), float(y)])
        new_r.append(float(rr))

    if not new_c:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    return np.asarray(new_c, np.float32), np.asarray(new_r, np.float32)


def complete_dim_fibers_relaxed_hough(
    gray_u8: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Second Hough pass on the full image to recover dim or weakly segmented fibers."""

    if not ENABLE_RELAXED_HOUGH_COMPLETION:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)
    if centers.size == 0 or radii.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    g = preprocess_gray_u8(gray_u8)
    grad = sobel_magnitude_u8(g)
    base_r = float(np.median(radii))
    if not np.isfinite(base_r) or base_r <= 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    circles = cv2.HoughCircles(
        g,
        cv2.HOUGH_GRADIENT,
        dp=RELAXED_HOUGH_DP,
        minDist=float(RELAXED_HOUGH_MIN_DIST_SCALE * base_r),
        param1=RELAXED_HOUGH_PARAM1,
        param2=RELAXED_HOUGH_PARAM2,
        minRadius=max(1, int(round(RELAXED_HOUGH_MIN_R_SCALE * base_r))),
        maxRadius=max(2, int(round(RELAXED_HOUGH_MAX_R_SCALE * base_r))),
    )
    if circles is None:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    new_c: list[list[float]] = []
    new_r: list[float] = []
    for x, y, r in circles[0].astype(np.float32):
        d = np.linalg.norm(centers - np.array([x, y], np.float32), axis=1)
        if np.any(d < (radii + r) * RELAXED_HOUGH_EXISTING_DIST_SCALE):
            continue

        rr, edge_s = refine_radius_by_edge(grad, float(x), float(y), float(r))
        contrast_s = circle_contrast_score(g, float(x), float(y), rr)
        if edge_s < RELAXED_HOUGH_EDGE_MIN or contrast_s < RELAXED_HOUGH_CONTRAST_MIN:
            continue

        new_c.append([float(x), float(y)])
        new_r.append(float(rr))

    if not new_c:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32)

    return np.asarray(new_c, np.float32), np.asarray(new_r, np.float32)


# ==========================
# Non-overlap: shrink larger circle
# ==========================
def enforce_non_overlapping_shrink_larger(
    centers: np.ndarray,
    radii: np.ndarray,
    *,
    gap: float = 0.0,
    min_radius: Optional[float] = None,
    max_iter: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(centers, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32)
    if centers.size == 0 or radii.size == 0:
        return centers.reshape(0, 2), radii.reshape(0)

    base_r = float(np.median(radii))
    if not np.isfinite(base_r) or base_r <= 0:
        base_r = float(np.mean(radii)) if radii.size else 1.0
    if min_radius is None:
        min_radius = 0.5 * base_r

    keep = np.ones(len(radii), dtype=bool)

    for _ in range(max_iter):
        idx = np.flatnonzero(keep)
        if idx.size <= 1:
            break
        c = centers[idx]
        r_new = radii[idx].copy()
        changed = False

        n = len(r_new)
        for i in range(n - 1):
            dx = c[i, 0] - c[i + 1 :, 0]
            dy = c[i, 1] - c[i + 1 :, 1]
            d = np.sqrt(dx * dx + dy * dy)

            overlap = d < (r_new[i] + r_new[i + 1 :] + gap)
            if not np.any(overlap):
                continue
            js = np.flatnonzero(overlap)
            for k in js:
                j = i + 1 + int(k)
                dij = float(d[k])

                ri = float(r_new[i])
                rj = float(r_new[j])

                if ri > rj:
                    big, small = i, j
                elif rj > ri:
                    big, small = j, i
                else:
                    big, small = j, i

                limit = dij - float(r_new[small]) - gap
                if limit < r_new[big]:
                    r_new[big] = float(limit)
                    changed = True

        drop_local = r_new < float(min_radius)
        if np.any(drop_local):
            keep[idx[drop_local]] = False
            changed = True
            continue

        radii[idx] = r_new
        if not changed:
            break

    keep &= radii >= float(min_radius)
    return centers[keep], radii[keep]


# ==========================
# Boundary + normalize + plot
# ==========================
def detect_boundary_fibers(centers: np.ndarray, radii: np.ndarray, image_shape: Tuple[int, int], p: Params) -> np.ndarray:
    if centers.size == 0:
        return np.empty((0,), dtype=np.uint8)
    h, w = image_shape
    mean_r = float(np.mean(radii)) if radii.size else 0.0
    margin = float(p.boundary_margin * mean_r)
    flags = (
        (centers[:, 0] < margin)
        | (centers[:, 0] > (w - margin))
        | (centers[:, 1] < margin)
        | (centers[:, 1] > (h - margin))
    )
    return flags.astype(np.uint8)


def normalize_fibers(
    centers: np.ndarray,
    radii: np.ndarray,
    image_shape: Tuple[int, int],
    target_short: float = 10.0,
    region_origin: Tuple[float, float] = (0.0, 0.0),
    region_size: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    h, w = image_shape
    scale = float(target_short / min(h, w)) if min(h, w) > 0 else 1.0
    if centers.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0,), np.float32), (float(h * scale), float(w * scale))

    centers_norm = centers.astype(np.float32) * scale
    radii_norm = radii.astype(np.float32) * scale

    x0, y0 = region_origin
    x1, y1 = x0 + region_size, y0 + region_size
    mask = (
        (centers_norm[:, 0] >= x0)
        & (centers_norm[:, 0] <= x1)
        & (centers_norm[:, 1] >= y0)
        & (centers_norm[:, 1] <= y1)
    )
    return centers_norm[mask], radii_norm[mask], (float(h * scale), float(w * scale))


def plot_result(gray_u8: np.ndarray, centers: np.ndarray, radii: np.ndarray, title: str, save_path: Optional[Path] = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(gray_u8, cmap="gray")
    for (x, y), r in zip(centers, radii):
        ax.add_patch(plt.Circle((x, y), r, fill=False, edgecolor="r", lw=0.8))
        ax.plot(x, y, "b.", ms=2)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_normalized(centers_norm: np.ndarray, radii_norm: np.ndarray, region_size: float = 10.0, title: str = "Normalized fibers", save_path: Optional[Path] = None) -> None:
    fig = plt.figure(figsize=(6, 6))
    if centers_norm.size:
        plt.scatter(centers_norm[:, 0], centers_norm[:, 1], s=20)
        ax = plt.gca()
        for (x, y), r in zip(centers_norm, radii_norm):
            ax.add_patch(plt.Circle((x, y), r, fill=False, edgecolor="r", lw=0.8))
    plt.xlim(0, region_size)
    plt.ylim(0, region_size)
    plt.gca().set_aspect("equal")
    plt.title(title)
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ==========================
# Pipeline
# ==========================
def detect_candidate_circles(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run the legacy marker/segmentation/completion stack without final overlap cleanup."""

    labels, est_r = watershed_split(gray)
    centers, radii = circles_from_labels(gray, labels)

    if DETECTION_MODE in ("blob", "hybrid"):
        c_blob, r_blob = circles_from_blob_log(gray, est_r)
        if DETECTION_MODE == "blob":
            centers, radii = c_blob, r_blob
        elif c_blob.size:
            centers = np.vstack([centers, c_blob]).astype(np.float32)
            radii = np.hstack([radii, r_blob]).astype(np.float32)

    c2, r2 = complete_missing_fibers(gray, centers, radii)
    if c2.size:
        centers = np.vstack([centers, c2]).astype(np.float32)
        radii = np.hstack([radii, r2]).astype(np.float32)

    return centers.astype(np.float32), radii.astype(np.float32), float(est_r)


def detect_candidate_circles_with_boundary_padding(
    gray: np.ndarray,
    p: Params,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Detect candidates on a reflected image and crop them back to the original coordinates."""

    if not p.use_reflect_padding:
        centers, radii, est_r = detect_candidate_circles(gray)
        return centers, radii, est_r, 0

    est_r0 = estimate_particle_radius(gray)
    gray_pad, pad = reflect_pad_for_boundary(gray, est_r0)
    if pad <= 0:
        centers, radii, est_r = detect_candidate_circles(gray)
        return centers, radii, est_r, 0

    centers_pad, radii, est_r = detect_candidate_circles(gray_pad)
    if centers_pad.size == 0:
        return centers_pad, radii, est_r, pad

    centers = centers_pad.copy()
    centers[:, 0] -= float(pad)
    centers[:, 1] -= float(pad)

    h, w = gray.shape
    base_r = float(np.median(radii)) if radii.size else float(est_r0)
    if not np.isfinite(base_r) or base_r <= 0:
        base_r = max(1.0, float(est_r0))
    keep_margin = float(KEEP_CENTER_MARGIN_SCALE * base_r)
    keep = (
        (centers[:, 0] >= -keep_margin)
        & (centers[:, 0] <= (w - 1 + keep_margin))
        & (centers[:, 1] >= -keep_margin)
        & (centers[:, 1] <= (h - 1 + keep_margin))
    )
    centers = centers[keep]
    radii = radii[keep]
    return centers.astype(np.float32), radii.astype(np.float32), float(est_r), pad


def finalize_circles(gray: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the legacy shrink and NMS cleanup after optional boundary padding is cropped away."""

    if SHRINK_ENABLE and radii.size:
        base_r = float(np.median(radii))
        if not np.isfinite(base_r) or base_r <= 0:
            base_r = float(np.mean(radii))

        centers, radii = enforce_non_overlapping_shrink_larger(
            centers,
            radii,
            gap=float(SHRINK_GAP_SCALE * base_r),
            min_radius=float(SHRINK_MIN_RADIUS_SCALE * base_r),
            max_iter=SHRINK_MAX_ITER,
        )

    if NMS_ENABLE and centers.size:
        scores = score_candidates(gray, centers, radii)
        centers, radii = nms_circles(centers, radii, scores, NMS_MIN_DIST_SCALE)

    return centers.astype(np.float32), radii.astype(np.float32)


def shrink_final_overlaps(centers: np.ndarray, radii: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Final radius-only cleanup so displayed/exported particles do not intersect."""

    if not FINAL_SHRINK_ENABLE or centers.size == 0 or radii.size == 0:
        return centers.astype(np.float32), radii.astype(np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32).copy()
    base_r = float(np.median(radii))
    if not np.isfinite(base_r) or base_r <= 0:
        base_r = float(np.mean(radii)) if radii.size else 1.0
    gap = float(FINAL_SHRINK_GAP_SCALE * base_r)
    min_radius = max(1.0e-3, float(FINAL_SHRINK_MIN_RADIUS_SCALE * base_r))

    for _ in range(int(FINAL_SHRINK_MAX_ITER)):
        changed = False
        for i in range(len(radii) - 1):
            delta = centers[i + 1 :] - centers[i]
            dist = np.sqrt(np.sum(delta * delta, axis=1))
            overlap = radii[i] + radii[i + 1 :] + gap - dist
            js = np.flatnonzero(overlap > 0.0)
            for local_j in js:
                j = i + 1 + int(local_j)
                dij = float(dist[local_j])
                allowed_sum = max(0.0, dij - gap)
                if radii[i] + radii[j] <= allowed_sum + 1.0e-6:
                    continue

                if allowed_sum <= 2.0 * min_radius:
                    new_ri = max(1.0e-3, allowed_sum * 0.5)
                    new_rj = max(1.0e-3, allowed_sum - new_ri)
                elif radii[i] >= radii[j]:
                    new_rj = max(min_radius, min(float(radii[j]), allowed_sum - min_radius))
                    new_ri = max(min_radius, allowed_sum - new_rj)
                    new_ri = min(float(radii[i]), new_ri)
                    if new_ri + new_rj > allowed_sum:
                        new_ri = allowed_sum - new_rj
                else:
                    new_ri = max(min_radius, min(float(radii[i]), allowed_sum - min_radius))
                    new_rj = max(min_radius, allowed_sum - new_ri)
                    new_rj = min(float(radii[j]), new_rj)
                    if new_ri + new_rj > allowed_sum:
                        new_rj = allowed_sum - new_ri

                if new_ri < radii[i] - 1.0e-6 or new_rj < radii[j] - 1.0e-6:
                    radii[i] = max(1.0e-3, float(new_ri))
                    radii[j] = max(1.0e-3, float(new_rj))
                    changed = True
        if not changed:
            break

    return centers.astype(np.float32), radii.astype(np.float32)


def run_pipeline(
    image_path: Path,
    output_txt: Path,
    p: Params,
    target_short: float = 10.0,
    region_origin: Tuple[float, float] = (0.0, 0.0),
    region_size: float = 10.0,
    save_boundary_flags: Optional[Path] = None,
    fig_dir: Optional[Path] = None,
) -> None:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    centers, radii, est_r, pad = detect_candidate_circles_with_boundary_padding(gray, p)
    centers, radii = finalize_circles(gray, centers, radii)

    if p.use_relaxed_completion:
        c_relaxed, r_relaxed = complete_dim_fibers_relaxed_hough(gray, centers, radii)
        if c_relaxed.size:
            if NMS_ENABLE:
                relaxed_scores = score_candidates(gray, c_relaxed, r_relaxed)
                c_relaxed, r_relaxed = nms_circles(c_relaxed, r_relaxed, relaxed_scores, NMS_MIN_DIST_SCALE)
            centers = np.vstack([centers, c_relaxed]).astype(np.float32)
            radii = np.hstack([radii, r_relaxed]).astype(np.float32)
    centers, radii = shrink_final_overlaps(centers, radii)

    boundary_flag = detect_boundary_fibers(centers, radii, gray.shape, p)
    if save_boundary_flags is not None:
        np.savetxt(str(save_boundary_flags), boundary_flag.reshape(-1, 1), fmt="%d")

    centers_norm, radii_norm, new_size = normalize_fibers(
        centers,
        radii,
        gray.shape,
        target_short=target_short,
        region_origin=region_origin,
        region_size=region_size,
    )

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(output_txt), np.column_stack([centers_norm, radii_norm]), fmt="%.4f")

    print("Estimated radius (px):", round(float(est_r), 3))
    print("Reflect padding (px):", int(pad))
    print("Detected circles:", len(centers))
    print("Normalized image size (h, w):", new_size)
    print(f"Fibers inside {region_size}x{region_size}:", len(centers_norm))
    print("Saved:", output_txt)

    if p.show_result:
        if fig_dir is None:
            fig_dir = output_txt.parent
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_result(
            gray,
            centers,
            radii,
            title=f"Detected fibers (high recall): {len(centers)}",
            save_path=fig_dir / "image_detection_overlay.png",
        )
        plot_normalized(
            centers_norm,
            radii_norm,
            region_size=region_size,
            title=f"Normalized fibers in {region_size}x{region_size}",
            save_path=fig_dir / "image_detection_normalized.png",
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fiber detection (high recall mode)")
    ap.add_argument("--image", type=Path, default=Path("fiber_composites_section_v1.jpg"))
    ap.add_argument("--out", type=Path, default=Path("result_files/fiber_centers_normalized.txt"))
    ap.add_argument("--target-short", type=float, default=10.0)
    ap.add_argument("--region-size", type=float, default=10.0)
    ap.add_argument("--region-x0", type=float, default=0.0)
    ap.add_argument("--region-y0", type=float, default=0.0)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--no-boundary-padding", action="store_true")
    ap.add_argument("--no-relaxed-completion", action="store_true")
    ap.add_argument("--save-boundary", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    params = Params(
        show_result=not args.no_show,
        use_reflect_padding=not args.no_boundary_padding,
        use_relaxed_completion=not args.no_relaxed_completion,
    )
    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image} (Working directory: {Path.cwd()})")

    base_dir = Path(__file__).resolve().parent
    for parent in [base_dir] + list(base_dir.parents):
        if (parent / "main.py").is_file():
            base_dir = parent
            break

    out_dir = base_dir / "result_files"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = args.out if args.out.is_absolute() else (out_dir / args.out.name)
    save_boundary = None
    if args.save_boundary is not None:
        save_boundary = args.save_boundary if args.save_boundary.is_absolute() else (out_dir / args.save_boundary.name)

    run_pipeline(
        image_path=args.image,
        output_txt=out_path,
        p=params,
        target_short=args.target_short,
        region_origin=(args.region_x0, args.region_y0),
        region_size=args.region_size,
        save_boundary_flags=save_boundary,
        fig_dir=out_dir,
    )


if __name__ == "__main__":
    main()
