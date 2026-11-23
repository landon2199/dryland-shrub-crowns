"""Study area, analysis grid and spatial blocks.

Everything is on one north-up grid in WGS 84 / UTM zone 13N (EPSG:32613), the CRS of the NEON
lidar and of the Sentinel-2 tiles over Jornada. The 0.5 m grid nests exactly in the 10 m
Sentinel-2 grid (20 x 20 fine cells per 10 m pixel), so cover can be aggregated without
resampling Sentinel-2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CRS = "EPSG:32613"
FINE = 0.5  # m, analysis grid (NAIP is resampled from 0.6 m; lidar is gridded directly)
COARSE = 10.0  # m, Sentinel-2

# 3 km x 3 km around the NEON Jornada (JORN) tower, southern New Mexico, Chihuahuan Desert.
# Lower-left corners of the nine 1 km NEON lidar tiles.
SITE = dict(name="NEON Jornada (JORN), New Mexico", x0=326000.0, y0=3606000.0, size=3000.0)


@dataclass(frozen=True)
class Grid:
    x0: float
    y1: float
    width: int
    height: int
    res: float
    crs: str = CRS

    @property
    def transform(self):
        from rasterio.transform import from_origin

        return from_origin(self.x0, self.y1, self.res, self.res)

    @property
    def bounds(self):
        return (self.x0, self.y1 - self.height * self.res, self.x0 + self.width * self.res, self.y1)

    def lonlat_bbox(self):
        from rasterio.warp import transform_bounds

        return transform_bounds(self.crs, "EPSG:4326", *self.bounds, densify_pts=21)


def site_grid(res: float = FINE, site: dict = SITE) -> Grid:
    n = int(round(site["size"] / res))
    return Grid(x0=site["x0"], y1=site["y0"] + site["size"], width=n, height=n, res=res)


def block_ids(shape: tuple[int, int], res: float, block_m: float) -> np.ndarray:
    """Integer block id for every cell of a grid (row-major blocks of block_m metres)."""
    h, w = shape
    k = int(round(block_m / res))
    nbx = int(np.ceil(w / k))
    r = np.arange(h) // k
    c = np.arange(w) // k
    return (r[:, None] * nbx + c[None, :]).astype(np.int32)


def split_blocks(n_blocks: int, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded random split of block ids into (dev, test)."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n_blocks)
    n_test = int(round(test_frac * n_blocks))
    return np.sort(ids[n_test:]), np.sort(ids[:n_test])


def aggregate(fine: np.ndarray, factor: int, func: str = "mean") -> np.ndarray:
    """Block-aggregate a 2-D array by an integer factor (shape must divide)."""
    h, w = fine.shape
    a = fine.reshape(h // factor, factor, w // factor, factor)
    if func == "mean":
        return a.mean(axis=(1, 3))
    if func == "sum":
        return a.sum(axis=(1, 3))
    if func == "max":
        return a.max(axis=(1, 3))
    raise ValueError(func)
