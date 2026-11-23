"""Object matching, block tallies and spatial block-bootstrap confidence intervals.

A detected crown matches a reference crown when their intersection over union (IoU) is at least
`iou_min`; matching is one-to-one, highest IoU first. Every object is assigned to the spatial
block that holds its centroid, so a block's tallies can be resampled as a unit (block bootstrap):
crowns near each other share errors, and resampling whole blocks keeps that dependence.
"""
from __future__ import annotations

import numpy as np

SIZE_BINS_M2 = (1.0, 2.0, 4.0, 8.0, 16.0, np.inf)


def object_stats(label: np.ndarray, blocks: np.ndarray, res: float) -> dict:
    """Area (m2), centroid block and centroid (row, col) of every object 1..n."""
    n = int(label.max())
    flat = label.ravel()
    cnt = np.bincount(flat, minlength=n + 1).astype(np.float64)
    rows, cols = np.indices(label.shape)
    r = np.bincount(flat, weights=rows.ravel(), minlength=n + 1) / np.maximum(cnt, 1)
    c = np.bincount(flat, weights=cols.ravel(), minlength=n + 1) / np.maximum(cnt, 1)
    ri = np.clip(np.round(r).astype(int), 0, label.shape[0] - 1)
    ci = np.clip(np.round(c).astype(int), 0, label.shape[1] - 1)
    return {"area": cnt[1:] * res * res, "block": blocks[ri, ci][1:], "row": r[1:], "col": c[1:]}


def match(ref: np.ndarray, pred: np.ndarray, iou_min: float = 0.3) -> dict:
    """One-to-one greedy IoU matching of two label rasters. Returns matched (ref_id, pred_id, iou),
    ids are 1-based."""
    nr, npd = int(ref.max()), int(pred.max())
    both = (ref > 0) & (pred > 0)
    key = ref[both].astype(np.int64) * (npd + 1) + pred[both].astype(np.int64)
    u, inter = np.unique(key, return_counts=True)
    ri, pi = u // (npd + 1), u % (npd + 1)
    ar = np.bincount(ref.ravel(), minlength=nr + 1)
    ap = np.bincount(pred.ravel(), minlength=npd + 1)
    iou = inter / (ar[ri] + ap[pi] - inter)
    order = np.argsort(-iou, kind="stable")
    used_r = np.zeros(nr + 1, bool)
    used_p = np.zeros(npd + 1, bool)
    mr, mp, mi = [], [], []
    for k in order:
        if iou[k] < iou_min:
            break
        a, b = ri[k], pi[k]
        if used_r[a] or used_p[b]:
            continue
        used_r[a] = used_p[b] = True
        mr.append(a)
        mp.append(b)
        mi.append(iou[k])
    return {"ref_id": np.array(mr, int), "pred_id": np.array(mp, int), "iou": np.array(mi, float),
            "n_ref": nr, "n_pred": npd}


def block_tallies(ref: np.ndarray, pred: np.ndarray, blocks: np.ndarray, n_blocks: int, res: float,
                  iou_min: float = 0.3) -> dict:
    """Per-block counts for precision / recall / count / cover / size statistics."""
    m = match(ref, pred, iou_min)
    rs, ps = object_stats(ref, blocks, res), object_stats(pred, blocks, res)
    bc = lambda idx, w=None: np.bincount(idx, weights=w, minlength=n_blocks).astype(np.float64)
    rb, pb = rs["block"], ps["block"]
    mrb = rb[m["ref_id"] - 1] if len(m["ref_id"]) else np.array([], int)
    mpb = pb[m["pred_id"] - 1] if len(m["pred_id"]) else np.array([], int)
    ra = rs["area"][m["ref_id"] - 1] if len(m["ref_id"]) else np.array([])
    pa = ps["area"][m["pred_id"] - 1] if len(m["pred_id"]) else np.array([])
    t = {
        "n_ref": bc(rb), "n_pred": bc(pb),
        "tp_ref": bc(mrb), "tp_pred": bc(mpb),
        "cover_ref_px": bc(blocks.ravel(), (ref > 0).ravel().astype(float)),
        "cover_pred_px": bc(blocks.ravel(), (pred > 0).ravel().astype(float)),
        "px": bc(blocks.ravel()),
        "m_area_ref": bc(mrb, ra), "m_area_pred": bc(mrb, pa),
        "m_log_ratio": bc(mrb, np.log(np.maximum(pa, 1e-9) / np.maximum(ra, 1e-9))),
        "m_iou": bc(mrb, m["iou"]),
    }
    # recall by reference crown size class
    rbin = np.digitize(rs["area"], SIZE_BINS_M2[1:-1])
    matched = np.zeros(len(rs["area"]), bool)
    matched[m["ref_id"] - 1] = True
    for k in range(len(SIZE_BINS_M2) - 1):
        sel = rbin == k
        t[f"size{k}_n"] = bc(rb[sel])
        t[f"size{k}_tp"] = bc(rb[sel & matched])
    return t


def metrics_from_tallies(t: dict, blocks_sel: np.ndarray) -> dict:
    s = {k: v[blocks_sel].sum() for k, v in t.items()}
    p = s["tp_pred"] / max(s["n_pred"], 1)
    r = s["tp_ref"] / max(s["n_ref"], 1)
    out = {
        "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
        "count_ratio": s["n_pred"] / max(s["n_ref"], 1),
        "cover_ref": s["cover_ref_px"] / s["px"], "cover_pred": s["cover_pred_px"] / s["px"],
        "matched_area_ratio": s["m_area_pred"] / max(s["m_area_ref"], 1e-9),
        "matched_median_like_ratio": float(np.exp(s["m_log_ratio"] / max(s["tp_ref"], 1))),
        "mean_iou_matched": s["m_iou"] / max(s["tp_ref"], 1),
        "n_ref": s["n_ref"], "n_pred": s["n_pred"],
    }
    out["cover_ratio"] = out["cover_pred"] / max(out["cover_ref"], 1e-12)
    k = 0
    while f"size{k}_n" in s:
        out[f"recall_size{k}"] = s[f"size{k}_tp"] / s[f"size{k}_n"] if s[f"size{k}_n"] > 0 else np.nan
        k += 1
    return out


def block_bootstrap(t: dict, blocks_sel: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Point estimate plus percentile interval from resampling whole blocks with replacement."""
    rng = np.random.default_rng(seed)
    est = metrics_from_tallies(t, blocks_sel)
    draws = {k: [] for k in est}
    for _ in range(n_boot):
        b = rng.choice(blocks_sel, size=len(blocks_sel), replace=True)
        m = metrics_from_tallies(t, b)
        for k, v in m.items():
            draws[k].append(v)
    out = {}
    for k, v in est.items():
        d = np.asarray(draws[k], float)
        lo, hi = np.nanpercentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[k] = {"est": float(v), "lo": float(lo), "hi": float(hi)}
    return out


def cell_agreement(a: np.ndarray, b: np.ndarray) -> dict:
    """Agreement of two per-cell quantities: R2 of b against a (1:1 line), RMSE, bias, Pearson r."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    ss_res = ((b - a) ** 2).sum()
    ss_tot = ((a - a.mean()) ** 2).sum()
    return {"n": int(ok.sum()), "r2_1to1": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "pearson_r": float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan"),
            "rmse": float(np.sqrt(((b - a) ** 2).mean())), "bias": float((b - a).mean())}
