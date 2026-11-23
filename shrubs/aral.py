"""Aral add-on: saxaul plantation plots on the dried Aral Sea bed (Kazakhstan), 2018-2025.

Plot points, planting years and 2020 field results come from Table I and the plot descriptions of
Salmukhanbetova et al. (2021, Central Asian Journal of Water Research 7(1): 128-157,
doi:10.29258/CAJWR/2020-R1.v7-1/128-157.eng). Each plot is one GPS point for a 1-2 ha plot, so a
plot is drawn here as a 50 m circle around its point (about 0.8 ha, inside even a 1 ha plot if
the point is near its centre); where the point sits in the plot is not known.

Surveyed cover: the paper's own value where it gives one; otherwise trees per ha x mean crown
area (marked 'derived'). Plots 3 and 4 are reported together.
"""
from __future__ import annotations

import numpy as np

AREA_CRS = "EPSG:32641"  # UTM 41N
BOX = (60.60, 45.82, 60.73, 45.91)  # west, south, east, north (WGS 84)
PLOT_RADIUS_M = 50.0
CONTROL_MIN_DIST_M = 600.0  # control pixels: at least this far from every plot point

# plot: (lat, lon, planting year, cover_2020 fraction or None, trees_per_ha, mean crown m2, survival %, note)
PLOTS = {
    1: (45.84917, 60.67361, 2008, 0.40, 1000, 1.7, 64.5, "replanted 2010"),
    2: (45.84964, 60.67186, 2009, None, 80, 2.8, 9.4, "replanted 2010; takyr-like solonchak"),
    3: (45.88092, 60.70606, 2011, 0.25, 500, 5.1, None, "reported with plot 4"),
    4: (45.88158, 60.70675, 2013, 0.25, 500, 5.1, None, "reported with plot 3"),
    17: (45.85792, 60.65922, 2016, None, 700, 0.6, 78.0, "coastal sandy loam"),
    18: (45.85781, 60.65908, 2017, 0.18, 700, 0.9, 72.4, ""),
    19: (45.85908, 60.65092, 2017, None, 900, 1.4, None, "grown from seed"),
    20: (45.84892, 60.65064, 2018, None, 200, 0.2, 20.0, ""),
    21: (45.84925, 60.65117, 2018, None, 60, None, 6.0, "crust solonchak"),
    22: (45.84847, 60.65917, 2018, None, 45, None, 4.5, "crust solonchak"),
    23: (45.85639, 60.62839, 2019, 0.063, 700, 0.9, 70.0, ""),
}


def plot_cover(p: tuple) -> tuple[float | None, str]:
    lat, lon, year, cover, tph, crown, surv, note = p
    if cover is not None:
        return cover, "reported"
    if tph is not None and crown is not None:
        return tph * crown / 10000.0, "derived (trees/ha x crown area)"
    return None, "not reported"


def plot_xy(plots: dict = PLOTS) -> dict[int, tuple[float, float]]:
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", AREA_CRS, always_xy=True)
    return {k: tr.transform(v[1], v[0]) for k, v in plots.items()}


def area_grid(res: float = 10.0):
    from rasterio.warp import transform_bounds

    from .grid import Grid

    l, b, r, t = transform_bounds("EPSG:4326", AREA_CRS, *BOX, densify_pts=21)
    x0, y1 = np.floor(l / res) * res, np.ceil(t / res) * res
    return Grid(x0=x0, y1=y1, width=int(np.ceil((r - x0) / res)), height=int(np.ceil((y1 - b) / res)),
                res=res, crs=AREA_CRS)


def plot_masks(grid, xy: dict, radius: float = PLOT_RADIUS_M, control_min: float = CONTROL_MIN_DIST_M):
    """Boolean mask per plot (pixel centres within `radius`) and a control mask (pixel centres
    farther than `control_min` from every plot)."""
    x0, y0, x1, y1 = grid.bounds
    xs = x0 + (np.arange(grid.width) + 0.5) * grid.res
    ys = y1 - (np.arange(grid.height) + 0.5) * grid.res
    X, Y = np.meshgrid(xs, ys)
    masks, dmin = {}, np.full(X.shape, np.inf)
    for k, (x, y) in xy.items():
        d = np.hypot(X - x, Y - y)
        masks[k] = d <= radius
        dmin = np.minimum(dmin, d)
    return masks, dmin > control_min


def season_window(year: int) -> tuple[str, str]:
    """Late growing season (1 Aug - 30 Sep): saxaul in leaf, spring ephemerals dry."""
    return f"{year}-08-01", f"{year}-09-30"


def s1_composite(grid, start: str, end: str) -> dict:
    """Median VV and VH (dB, from linear power) of Sentinel-1 RTC scenes, one orbit direction."""
    import odc.stac
    from odc.geo.geobox import GeoBox

    from .imagery import catalog

    items = list(catalog().search(collections=["sentinel-1-rtc"], bbox=grid.lonlat_bbox(),
                                  datetime=f"{start}/{end}").items())
    if not items:
        return {}
    direc = max({i.properties.get("sat:orbit_state") for i in items},
                key=lambda d: sum(i.properties.get("sat:orbit_state") == d for i in items))
    items = [i for i in items if i.properties.get("sat:orbit_state") == direc]
    gbox = GeoBox.from_bbox(grid.bounds, crs=grid.crs, resolution=grid.res)
    ds = odc.stac.load(items, bands=["vv", "vh"], geobox=gbox, groupby="solar_day", chunks={}).compute()
    out = {"orbit_state": direc, "n_dates": int(ds.sizes["time"])}
    for b in ("vv", "vh"):
        v = ds[b].values.astype(np.float32)
        v[v <= 0] = np.nan
        out[b] = 10 * np.log10(np.nanmedian(v, axis=0))
    return out
