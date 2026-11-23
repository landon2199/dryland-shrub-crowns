"""Reference shrub crowns from the NEON airborne lidar point cloud.

NEON's own canopy height model (DP3.30015.001) sets heights under 2 m to zero, which removes
almost every desert shrub, and NEON's ground class includes most shrub returns at this site
(see `lidar_qa` in results/metrics.json for the share of returns above 0.3 m that are classed as ground).
So the terrain and the 0.5 m canopy height model are rebuilt here from all returns:

1. DTM: lowest return in each 1 m cell, then a 3 x 3 minimum filter (a cell fully under a crown
   takes the lowest neighbour) and a 3 x 3 mean filter.
2. Height above ground (HAG) of every return = z - DTM (bilinear).
3. CHM: highest HAG in each 0.5 m cell.

Data: NEON AOP discrete-return lidar, classified point cloud (DP1.30003.001), Jornada 2022
(visit 2022_JORN_5, flown September 2022), public Google Cloud bucket used by the NEON portal.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
from scipy import ndimage

from .grid import Grid

NEON_BUCKET = "https://storage.googleapis.com/neon-aop-products"
NEON_VISIT = dict(year=2022, domain="D14", site="JORN", visit=5)
NOISE_CLASSES = (7, 18)  # ASPRS low noise, high noise


def tile_url(e: int, n: int, v: dict = NEON_VISIT) -> str:
    base = f"{NEON_BUCKET}/{v['year']}/FullSite/{v['domain']}/{v['year']}_{v['site']}_{v['visit']}"
    name = f"NEON_{v['domain']}_{v['site']}_DP1_{e}_{n}_classified_point_cloud_colorized.laz"
    return f"{base}/L1/DiscreteLidar/ClassifiedPointCloud/{name}"


def tiles_for(grid: Grid) -> list[tuple[int, int]]:
    x0, y0, x1, y1 = grid.bounds
    es = range(int(np.floor(x0 / 1000) * 1000), int(np.ceil(x1 / 1000) * 1000), 1000)
    ns = range(int(np.floor(y0 / 1000) * 1000), int(np.ceil(y1 / 1000) * 1000), 1000)
    return [(e, n) for e in es for n in ns]


def download_tiles(grid: Grid, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for e, n in tiles_for(grid):
        p = out_dir / f"{e}_{n}.laz"
        if not p.exists() or p.stat().st_size == 0:
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(tile_url(e, n), tmp)
            tmp.replace(p)
        paths.append(p)
    return paths


def _read_points(path: Path):
    import laspy

    las = laspy.read(path)
    c = np.asarray(las.classification)
    keep = ~np.isin(c, NOISE_CLASSES)
    return np.asarray(las.x)[keep], np.asarray(las.y)[keep], np.asarray(las.z)[keep], c[keep]


def _cell_index(x, y, x0, y1, res, h, w):
    col = np.floor((x - x0) / res).astype(np.int64)
    row = np.floor((y1 - y) / res).astype(np.int64)
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    return row, col, ok


def grid_reduce(values: np.ndarray, row: np.ndarray, col: np.ndarray, shape: tuple[int, int], how: str) -> np.ndarray:
    """Per-cell min or max of point values (NaN where a cell has no point)."""
    h, w = shape
    idx = row * w + col
    fill = np.inf if how == "min" else -np.inf
    out = np.full(h * w, fill)
    (np.minimum if how == "min" else np.maximum).at(out, idx, values)
    out[np.isinf(out)] = np.nan
    return out.reshape(h, w)


def terrain_from_zmin(zmin: np.ndarray, min_size: int = 3, mean_size: int = 3) -> np.ndarray:
    """DTM from the per-cell lowest return: fill empty cells, minimum filter, mean filter."""
    z = zmin.copy()
    if np.isnan(z).any():
        idx = ndimage.distance_transform_edt(np.isnan(z), return_distances=False, return_indices=True)
        z = z[tuple(idx)]
    z = ndimage.minimum_filter(z, size=min_size, mode="nearest")
    return ndimage.uniform_filter(z, size=mean_size, mode="nearest")


def build_chm(grid: Grid, laz_paths: list[Path], dtm_res: float = 1.0) -> dict:
    """Two passes over the tiles: per-cell lowest return -> DTM; then HAG -> 0.5 m CHM."""
    x0, y0, x1, y1 = grid.bounds
    hd, wd = int(round((y1 - y0) / dtm_res)), int(round((x1 - x0) / dtm_res))
    zmin = np.full((hd, wd), np.nan)
    n_pts = 0
    n_ground_high = 0
    n_high = 0
    for p in laz_paths:
        x, y, z, _ = _read_points(p)
        r, c, ok = _cell_index(x, y, x0, y1, dtm_res, hd, wd)
        part = grid_reduce(z[ok], r[ok], c[ok], (hd, wd), "min")
        zmin = np.fmin(zmin, part)
    dtm = terrain_from_zmin(zmin)

    chm = np.full((grid.height, grid.width), np.nan, dtype=np.float64)
    count = np.zeros(grid.height * grid.width, dtype=np.int32)
    for p in laz_paths:
        x, y, z, cls = _read_points(p)
        rows = (y1 - y) / dtm_res - 0.5
        cols = (x - x0) / dtm_res - 0.5
        hag = z - ndimage.map_coordinates(dtm, [rows, cols], order=1, mode="nearest")
        r, c, ok = _cell_index(x, y, x0, y1, grid.res, grid.height, grid.width)
        part = grid_reduce(hag[ok], r[ok], c[ok], chm.shape, "max")
        chm = np.fmax(chm, part)
        count += np.bincount(r[ok] * grid.width + c[ok], minlength=count.size).astype(np.int32)
        n_pts += int(ok.sum())
        high = ok & (hag > 0.3)
        n_high += int(high.sum())
        n_ground_high += int((high & (cls == 2)).sum())
    empty = np.isnan(chm)
    chm = np.clip(np.nan_to_num(chm, nan=0.0), 0, None).astype(np.float32)
    return {
        "chm": chm,
        "dtm": dtm.astype(np.float32),
        "density": count.reshape(chm.shape),
        "qa": {
            "points": n_pts,
            "points_per_m2": n_pts / ((x1 - x0) * (y1 - y0)),
            "empty_fine_cells_share": float(empty.mean()),
            "share_of_points_above_0p3m_classed_ground_by_neon": n_ground_high / max(n_high, 1),
        },
    }


def segment_crowns(height: np.ndarray, mask: np.ndarray, min_distance_px: int = 3, sigma: float = 1.0,
                   min_px: int = 4) -> np.ndarray:
    """Split a crown mask into crowns: marker-controlled watershed on a smoothed height-like
    surface (CHM or a vegetation index). Mask components without a peak keep their own label.
    Objects smaller than `min_px` cells are dropped."""
    from skimage.feature import peak_local_max
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import watershed

    mask = remove_small_objects(mask.astype(bool), max_size=min_px - 1) if min_px > 1 else mask.astype(bool)
    if not mask.any():
        return np.zeros(mask.shape, np.int32)
    s = ndimage.gaussian_filter(height.astype(np.float32), sigma) if sigma > 0 else height.astype(np.float32)
    comp, ncomp = ndimage.label(mask)
    peaks = peak_local_max(s, min_distance=min_distance_px, labels=comp, exclude_border=False)
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    lab = watershed(-s, markers, mask=mask)
    # components that got no marker
    missing = mask & (lab == 0)
    if missing.any():
        extra, _ = ndimage.label(missing)
        lab = np.where(missing, extra + lab.max(), lab)
    lab = relabel(lab)
    # drop crowns that are too small after splitting
    sizes = np.bincount(lab.ravel())
    small = sizes < min_px
    small[0] = False
    if small.any():
        lab[small[lab]] = 0
        lab = relabel(lab)
    return lab.astype(np.int32)


def relabel(lab: np.ndarray) -> np.ndarray:
    """Consecutive labels 1..n (0 stays background)."""
    u, inv = np.unique(lab, return_inverse=True)
    inv = inv.reshape(lab.shape)
    if u[0] != 0:
        return (inv + 1).astype(np.int32)
    return inv.astype(np.int32)


def reference_crowns(chm: np.ndarray, min_height: float = 0.5, min_area_m2: float = 1.0, res: float = 0.5) -> np.ndarray:
    """Reference shrubs: CHM cells at least `min_height` tall, split into crowns on CHM peaks."""
    min_px = max(1, int(round(min_area_m2 / res**2)))
    return segment_crowns(chm, chm >= min_height, min_distance_px=3, sigma=1.0, min_px=min_px)


def save_json(obj: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))
