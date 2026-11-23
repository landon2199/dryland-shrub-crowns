"""The scaling bridge: shrub cover in 10 m cells against Sentinel-2 indices.

The question is the cover level below which a Sentinel-2 pixel can no longer be told apart from
bare ground. Two views:

- Separability: for each cover class, the area under the ROC curve (AUC) for telling its cells
  apart from bare cells (reference cover under 1 %) with the index alone. AUC 0.5 is a coin flip,
  1.0 is perfect. The detection limit is the lowest cover class whose AUC reaches 0.8.
- Sensitivity: a straight-line fit of the index on cover gives the index change per 1 % cover;
  dividing three standard deviations of the index over bare cells by that slope gives a
  "three-sigma" detection limit, like a laboratory limit of detection.
"""
from __future__ import annotations

import numpy as np

COVER_BINS = (0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 1.01)
BARE_MAX = 0.01


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC AUC = P(index of a random positive > index of a random negative), ties count half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def separability(cover: np.ndarray, index: np.ndarray, blocks: np.ndarray | None = None,
                 n_boot: int = 500, seed: int = 0, bins=COVER_BINS) -> list[dict]:
    """AUC of each cover class against bare cells, with a block-bootstrap 95 % interval."""
    c, v = cover.ravel(), index.ravel()
    ok = np.isfinite(c) & np.isfinite(v)
    c, v = c[ok], v[ok]
    b = blocks.ravel()[ok] if blocks is not None else None
    bare = c < BARE_MAX
    rng = np.random.default_rng(seed)
    out = []
    ub = np.unique(b) if b is not None else None
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (c >= lo) & (c < hi)
        row = {"cover_lo": lo, "cover_hi": min(hi, 1.0), "n": int(sel.sum()), "n_bare": int(bare.sum()),
               "auc": auc(v[sel], v[bare])}
        if b is not None and sel.sum() > 0 and n_boot > 0:
            # resample whole blocks; index lookup per block
            idx_by_block = {k: np.nonzero(b == k)[0] for k in ub}
            draws = []
            for _ in range(n_boot):
                pick = np.concatenate([idx_by_block[k] for k in rng.choice(ub, len(ub), replace=True)])
                s, bb = sel[pick], bare[pick]
                if s.sum() and bb.sum():
                    draws.append(auc(v[pick][s], v[pick][bb]))
            row["auc_lo"], row["auc_hi"] = (float(x) for x in np.nanpercentile(draws, [2.5, 97.5]))
        out.append(row)
    return out


def detection_limit(rows: list[dict], auc_min: float = 0.8) -> float | None:
    """Lower edge of the first cover class whose AUC reaches `auc_min` (and stays there for the
    next class), or None."""
    for i, r in enumerate(rows):
        nxt = rows[i + 1]["auc"] if i + 1 < len(rows) else r["auc"]
        if r["n"] >= 20 and r["auc"] >= auc_min and nxt >= auc_min:
            return float(r["cover_lo"])
    return None


def sensitivity(cover: np.ndarray, index: np.ndarray, fit_max: float = 0.40) -> dict:
    """Least-squares line index = a + b * cover (cover as a fraction, cells up to `fit_max`),
    the bare-cell spread and the three-sigma detection limit 3 * sd_bare / b."""
    c, v = cover.ravel(), index.ravel()
    ok = np.isfinite(c) & np.isfinite(v) & (c <= fit_max)
    c, v = c[ok], v[ok]
    b, a = np.polyfit(c, v, 1)
    pred = a + b * c
    r2 = 1 - ((v - pred) ** 2).sum() / ((v - v.mean()) ** 2).sum()
    sd_bare = float(np.std(v[c < BARE_MAX], ddof=1))
    return {"intercept": float(a), "slope_per_unit_cover": float(b), "slope_per_1pct": float(b / 100),
            "r2": float(r2), "sd_bare": sd_bare, "lod_3sigma_cover": float(3 * sd_bare / b) if b > 0 else None,
            "n": int(ok.sum())}


def binned(cover: np.ndarray, index: np.ndarray, edges=np.arange(0, 0.42, 0.02)) -> list[dict]:
    """Median and interquartile range of the index in cover bins (for plots)."""
    c, v = cover.ravel(), index.ravel()
    ok = np.isfinite(c) & np.isfinite(v)
    c, v = c[ok], v[ok]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (c >= lo) & (c < hi)
        if s.sum() >= 10:
            q = np.percentile(v[s], [25, 50, 75])
            out.append({"cover_mid": float((lo + hi) / 2), "n": int(s.sum()), "q25": float(q[0]),
                        "median": float(q[1]), "q75": float(q[2])})
    return out


def crown_area_for_cover(cover_frac: float, pixel_m: float = 10.0) -> float:
    """Crown area (m2) a single shrub needs to cover `cover_frac` of one pixel."""
    return cover_frac * pixel_m * pixel_m


def planting_cover(crown_area_m2: float, plants_per_ha: float) -> float:
    """Ground cover fraction of a planting: crown area x density (ignoring overlap)."""
    return crown_area_m2 * plants_per_ha / 10000.0
