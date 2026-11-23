"""Crown detection on NAIP: an index-threshold + watershed baseline, and Segment Anything (SAM)
with automatic masks or with point prompts at vegetation-index peaks.

All methods return a label raster on the 0.5 m grid (0 = background, 1..n = crowns), so the same
scoring code applies to every method and to the lidar reference.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .lidar import relabel, segment_crowns

SAM_URL_VIT_H = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"


# ----------------------------------------------------------------------------- baseline
def index_watershed(index: np.ndarray, threshold: float, sigma: float = 1.0, min_distance_px: int = 3,
                    min_px: int = 4) -> np.ndarray:
    """Pixels whose smoothed index exceeds `threshold`, split into crowns by a marker-controlled
    watershed on index peaks (same splitting rule as the lidar reference)."""
    s = ndimage.gaussian_filter(index.astype(np.float32), sigma) if sigma > 0 else index
    return segment_crowns(s, s > threshold, min_distance_px=min_distance_px, sigma=0, min_px=min_px)


# ----------------------------------------------------------------------------- helpers
def stretch(naip: np.ndarray, bands=(0, 1, 2), lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Percentile stretch of 3 NAIP bands to uint8 HxWx3 (SAM expects an 8-bit RGB image)."""
    out = []
    for b in bands:
        v = naip[b].astype(np.float32)
        a, z = np.percentile(v, [lo, hi])
        out.append(np.clip((v - a) / max(z - a, 1e-6) * 255.0, 0, 255).astype(np.uint8))
    return np.ascontiguousarray(np.stack(out, -1))


def tile_starts(n: int, size: int, step: int) -> list[int]:
    if size >= n:
        return [0]
    s = list(range(0, n - size, step))
    s.append(n - size)
    return sorted(set(s))


def core_bounds(start: int, size: int, n: int, margin: int) -> tuple[int, int]:
    """Part of a tile that 'owns' objects: the tile minus `margin` on sides shared with a neighbour."""
    lo = start + (margin if start > 0 else 0)
    hi = start + size - (margin if start + size < n else 0)
    return lo, hi


def paint(label: np.ndarray, masks: list[np.ndarray], scores: list[float], r0: int, c0: int,
          core: tuple[int, int, int, int], next_id: int, max_taken: float = 0.3) -> int:
    """Write tile masks into the global label raster, best score first. A mask is skipped if its
    centroid is outside the tile core or if more than `max_taken` of it is already labelled."""
    cr0, cr1, cc0, cc1 = core
    for i in np.argsort(scores)[::-1]:
        m = masks[i]
        rr, cc = np.nonzero(m)
        if rr.size == 0:
            continue
        gr, gc = rr + r0, cc + c0
        cy, cx = gr.mean(), gc.mean()
        if not (cr0 <= cy < cr1 and cc0 <= cx < cc1):
            continue
        taken = label[gr, gc] > 0
        if taken.mean() > max_taken:
            continue
        free = ~taken
        label[gr[free], gc[free]] = next_id
        next_id += 1
    return next_id


def filter_labels(label: np.ndarray, index: np.ndarray, min_mean_index: float, min_px: int = 4,
                  max_px: int | None = None) -> np.ndarray:
    """Keep objects whose mean index is at least `min_mean_index` and whose size is in range."""
    lab = relabel(label)
    n = lab.max()
    if n == 0:
        return lab
    ids = np.arange(1, n + 1)
    size = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    mean = ndimage.mean(index, labels=lab, index=ids)
    keep = (size >= min_px) & (mean >= min_mean_index)
    if max_px is not None:
        keep &= size <= max_px
    lut = np.zeros(n + 1, np.int32)
    lut[1:][keep] = np.arange(1, keep.sum() + 1)
    return lut[lab]


# ----------------------------------------------------------------------------- SAM
def load_sam(checkpoint: str, model_type: str = "vit_h", device: str = "cuda"):
    from segment_anything import sam_model_registry

    return sam_model_registry[model_type](checkpoint=checkpoint).to(device).eval()


def sam_automatic(sam, rgb: np.ndarray, tile: int = 256, overlap: int = 64, max_px: int = 2000,
                  progress=None, **kwargs) -> np.ndarray:
    """SAM automatic mask generation on small tiles (SAM upsamples each 256 px tile to 1024 px,
    so a 2 m shrub is about 16 px wide inside the model). Masks larger than `max_px` (soil patches,
    whole tiles) are dropped before painting."""
    from segment_anything import SamAutomaticMaskGenerator

    params = dict(points_per_side=48, pred_iou_thresh=0.80, stability_score_thresh=0.85,
                  crop_n_layers=0, min_mask_region_area=0, points_per_batch=64, box_nms_thresh=0.7)
    params.update(kwargs)
    gen = SamAutomaticMaskGenerator(sam, **params)
    H, W = rgb.shape[:2]
    label = np.zeros((H, W), np.int32)
    nid = 1
    rows, cols = tile_starts(H, tile, tile - overlap), tile_starts(W, tile, tile - overlap)
    k = 0
    for r0 in rows:
        for c0 in cols:
            chip = np.ascontiguousarray(rgb[r0:r0 + tile, c0:c0 + tile])
            res = [m for m in gen.generate(chip) if m["area"] <= max_px]
            cr = core_bounds(r0, tile, H, overlap // 2)
            cc = core_bounds(c0, tile, W, overlap // 2)
            nid = paint(label, [m["segmentation"] for m in res],
                        [m["predicted_iou"] * m["stability_score"] for m in res], r0, c0,
                        (cr[0], cr[1], cc[0], cc[1]), nid)
            k += 1
            if progress:
                progress(k, len(rows) * len(cols))
    return relabel(label)


def index_peaks(index: np.ndarray, threshold: float, sigma: float = 1.0, min_distance_px: int = 3) -> np.ndarray:
    """(row, col) of local maxima of the smoothed index above `threshold` (point prompts)."""
    from skimage.feature import peak_local_max

    s = ndimage.gaussian_filter(index.astype(np.float32), sigma)
    return peak_local_max(s, min_distance=min_distance_px, threshold_abs=threshold, exclude_border=False)


def sam_points(sam, rgb: np.ndarray, peaks: np.ndarray, tile: int = 256, overlap: int = 64,
               max_px: int = 2000, batch: int = 64, device: str = "cuda", progress=None) -> np.ndarray:
    """Point-prompted SAM: one positive point per index peak; SAM returns 3 candidate masks per
    point and the one with the highest predicted IoU is kept."""
    import torch
    from segment_anything import SamPredictor

    pred = SamPredictor(sam)
    H, W = rgb.shape[:2]
    label = np.zeros((H, W), np.int32)
    nid = 1
    rows, cols = tile_starts(H, tile, tile - overlap), tile_starts(W, tile, tile - overlap)
    k = 0
    for r0 in rows:
        for c0 in cols:
            k += 1
            cr = core_bounds(r0, tile, H, overlap // 2)
            cc = core_bounds(c0, tile, W, overlap // 2)
            inside = (peaks[:, 0] >= cr[0]) & (peaks[:, 0] < cr[1]) & (peaks[:, 1] >= cc[0]) & (peaks[:, 1] < cc[1])
            pts = peaks[inside]
            if len(pts) == 0:
                continue
            chip = np.ascontiguousarray(rgb[r0:r0 + tile, c0:c0 + tile])
            pred.set_image(chip)
            xy = np.stack([pts[:, 1] - c0, pts[:, 0] - r0], -1).astype(np.float32)
            masks, scores = [], []
            for i in range(0, len(xy), batch):
                p = torch.as_tensor(pred.transform.apply_coords(xy[i:i + batch], chip.shape[:2]), device=device)[:, None, :]
                lab_t = torch.ones(p.shape[:2], dtype=torch.int64, device=device)
                m, s, _ = pred.predict_torch(p, lab_t, multimask_output=True)
                best = s.argmax(1)
                ar = torch.arange(len(best), device=device)
                masks.append(m[ar, best].cpu().numpy())
                scores.append(s[ar, best].cpu().numpy())
            masks = np.concatenate(masks)
            scores = np.concatenate(scores)
            ok = masks.reshape(len(masks), -1).sum(1) <= max_px
            # the prompt point owns the mask: paint uses the mask centroid, so require it in the core too
            nid = paint(label, list(masks[ok]), list(scores[ok]), r0, c0, (cr[0], cr[1], cc[0], cc[1]), nid)
            if progress:
                progress(k, len(rows) * len(cols))
    return relabel(label)
