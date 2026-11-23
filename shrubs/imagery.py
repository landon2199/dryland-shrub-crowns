"""NAIP and Sentinel-2 / Sentinel-1 from the Microsoft Planetary Computer (no account needed).

- NAIP: USDA 4-band (R, G, B, NIR) aerial imagery, 0.6 m, New Mexico 2022 (flown 29 May 2022 here).
  Digital numbers, not reflectance. Stored in NAD83 / UTM 13N and reprojected onto the WGS 84 grid.
- Sentinel-2 L2A: surface reflectance, scaled by 1/10000 after removing the -1000 offset that
  processing baseline 04.00 and later adds (from 25 Jan 2022).
- Sentinel-1 RTC: radiometrically terrain-corrected gamma-nought backscatter (linear power).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .grid import Grid

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]
SCL_CLEAR = (4, 5)  # vegetation, not vegetated (bare soil); 6 water, 11 snow are excluded too


def catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def fetch_naip(grid: Grid, year: int, out_path: Path, shift_m: tuple[float, float] = (0.0, 0.0)) -> tuple[np.ndarray, dict]:
    """NAIP of `year` reprojected (bilinear) onto `grid`; `shift_m` = (dx, dy) added to the NAIP
    georeference before resampling (co-registration to the lidar). Cached as .npy + .json."""
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    out_path = Path(out_path)
    meta_path = out_path.with_suffix(".json")
    if out_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if tuple(meta["shift_m"]) == tuple(shift_m):
            return np.load(out_path), meta
    items = [i for i in catalog().search(collections=["naip"], bbox=grid.lonlat_bbox()).items()
             if i.datetime.year == year]
    if not items:
        raise RuntimeError(f"No NAIP {year} over this grid")
    dst = np.zeros((4, grid.height, grid.width), np.uint8)
    for item in sorted(items, key=lambda i: i.id):
        with rasterio.open(item.assets["image"].href) as src:
            tmp = np.zeros_like(dst)
            # read to an array first: reproject() ignores src_transform when given a dataset band
            arr = src.read([1, 2, 3, 4])
            st = Affine.translation(*shift_m) * src.transform
            reproject(arr, tmp, src_transform=st, src_crs=src.crs, src_nodata=0,
                      dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=0,
                      resampling=Resampling.bilinear)
            src_crs = str(src.crs)
        empty = dst.sum(0) == 0
        dst[:, empty] = tmp[:, empty]
    meta = {"items": [i.id for i in items], "dates": sorted({str(i.datetime.date()) for i in items}),
            "src_crs": src_crs, "shift_m": list(shift_m), "nodata_share": float((dst.sum(0) == 0).mean())}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, dst)
    meta_path.write_text(json.dumps(meta, indent=2))
    return dst, meta


def naip_indices(naip: np.ndarray) -> dict[str, np.ndarray]:
    """NDVI and MSAVI from NAIP digital numbers scaled to 0-1 (not reflectance), plus brightness."""
    r, g, b, n = (naip[i].astype(np.float32) / 255.0 for i in range(4))
    return {"ndvi": ndvi(n, r), "msavi": msavi(n, r), "brightness": (r + g + b) / 3.0}


def ndvi(nir, red):
    return (nir - red) / np.maximum(nir + red, 1e-6)


def msavi(nir, red):
    """MSAVI2 of Qi et al. (1994): self-adjusting soil line, no soil factor to choose."""
    a = 2.0 * nir + 1.0
    return (a - np.sqrt(np.maximum(a * a - 8.0 * (nir - red), 0.0))) / 2.0


def coregistration_shift(moving: np.ndarray, fixed: np.ndarray, res: float, upsample: int = 10) -> tuple[float, float]:
    """(dx, dy) in metres that moves `moving` onto `fixed` (phase correlation, sub-pixel).
    Positive dx = east, positive dy = north."""
    from skimage.registration import phase_cross_correlation

    shift, _, _ = phase_cross_correlation(fixed, moving, upsample_factor=upsample, normalization=None)
    drow, dcol = shift
    return float(dcol * res), float(-drow * res)


def s2_items(bbox, start: str, end: str, max_cloud: float = 20.0):
    items = catalog().search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{start}/{end}",
                             query={"eo:cloud_cover": {"lt": max_cloud}}).items()
    return sorted(items, key=lambda i: i.datetime)


def _offset(item) -> float:
    pb = item.properties.get("s2:processing_baseline", "0")
    try:
        return 1000.0 if float(pb) >= 4.0 else 0.0
    except ValueError:
        return 0.0


def s2_scenes(grid: Grid, start: str, end: str, max_cloud: float = 20.0, bands=S2_BANDS) -> dict:
    """Per-date Sentinel-2 reflectance stack on `grid` (10 m), clouds masked with the SCL layer.
    Returns {'dates', 'refl': {band: (t, h, w)}, 'clear': (t, h, w) bool, 'items'}."""
    import odc.stac
    from odc.geo.geobox import GeoBox

    items = s2_items(grid.lonlat_bbox(), start, end, max_cloud)
    if not items:
        raise RuntimeError("No Sentinel-2 scenes")
    gbox = GeoBox.from_bbox(grid.bounds, crs=grid.crs, resolution=grid.res)
    ds = odc.stac.load(items, bands=list(bands), geobox=gbox, groupby="solar_day", resampling="nearest",
                       chunks={})
    ds = ds.compute()
    # offset per solar day: use the first item of that day
    by_day = {}
    for it in items:
        by_day.setdefault(str(it.datetime.date()), it)
    dates = [str(np.datetime_as_string(t, unit="D")) for t in ds.time.values]
    scl = ds["SCL"].values
    clear = np.isin(scl, SCL_CLEAR)
    refl = {}
    for b in bands:
        if b == "SCL":
            continue
        v = ds[b].values.astype(np.float32)
        off = np.array([_offset(by_day[d]) if d in by_day else 0.0 for d in dates], np.float32)[:, None, None]
        r = (v - off) / 10000.0
        r[v == 0] = np.nan
        refl[b] = r
    return {"dates": dates, "refl": refl, "clear": clear, "items": [i.id for i in items]}


def s2_indices(refl: dict) -> dict[str, np.ndarray]:
    """Vegetation and soil/salt indices from Sentinel-2 reflectance (same shapes as the bands)."""
    b, g, r, n = refl["B02"], refl["B03"], refl["B04"], refl["B08"]
    out = {"ndvi": ndvi(n, r), "msavi": msavi(n, r)}
    if "B11" in refl:
        s1, s2 = refl["B11"], refl["B12"]
        out["brightness"] = np.sqrt((r * r + n * n) / 2.0)  # soil brightness from red and NIR
        out["si_khan"] = np.sqrt(b * r)  # salinity index SI = sqrt(B * R), Khan et al. (2005)
        out["ndsi"] = (r - n) / np.maximum(r + n, 1e-6)  # normalized difference salinity index
        out["swir_ratio"] = s1 / np.maximum(s2, 1e-6)
    return out


def composite(scenes: dict, func=np.nanmedian, use_scl: bool = True) -> dict[str, np.ndarray]:
    """Per-pixel median over clear dates (SCL mask), or over all dates with data (use_scl=False:
    for salt crusts, which the older Sen2Cor scene classification labels as cloud)."""
    out = {}
    first = next(iter(scenes["refl"].values()))
    clear = scenes["clear"] if use_scl else np.isfinite(first)
    for b, v in scenes["refl"].items():
        m = np.where(clear, v, np.nan)
        out[b] = func(m, axis=0)
    out["n_clear"] = clear.sum(0)
    return out
