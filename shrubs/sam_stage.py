"""Run SAM in its own process: `python -m shrubs.sam_stage <naip.npy> <out.npy> <mode> <image> [prompt_q]`.

On Windows, torch's bundled DLLs clash with the GDAL / OpenMP libraries that the rest of the
pipeline loads (fbgemm.dll fails to load, or two OpenMP runtimes start). A separate process that
imports only Pillow, torch, numpy, scipy and scikit-image avoids both. run.py calls this module.
"""
from __future__ import annotations

import PIL.Image  # noqa: F401  isort: skip  (Pillow before torch)
import torch  # noqa: F401  isort: skip

import sys
import time

import numpy as np
from scipy import ndimage

from shrubs import detect as Dt
from shrubs.imagery import naip_indices

BANDS = {"truecolor": (0, 1, 2), "falsecolor": (3, 0, 1)}


def main(argv):
    naip_path, out_path, mode, image = argv[:4]
    prompt_q = float(argv[4]) if len(argv) > 4 else 85.0
    ckpt = argv[5] if len(argv) > 5 else "data/weights/sam_vit_h_4b8939.pth"
    naip = np.load(naip_path)
    img = Dt.stretch(naip, BANDS[image])
    sam = Dt.load_sam(ckpt, device="cuda" if torch.cuda.is_available() else "cpu")
    t = time.time()
    prog = lambda k, n: print(f"  SAM {mode} ({image}) tile {k}/{n}, {time.time() - t:.0f} s", flush=True) \
        if k % 200 == 0 else None
    if mode == "points":
        msavi = naip_indices(naip)["msavi"]
        peaks = Dt.index_peaks(msavi, np.percentile(ndimage.gaussian_filter(msavi, 1), prompt_q))
        print(f"  {len(peaks)} prompt points", flush=True)
        lab = Dt.sam_points(sam, img, peaks, progress=prog)
    elif mode == "auto":
        lab = Dt.sam_automatic(sam, img, progress=prog)
    else:
        raise ValueError(mode)
    np.save(out_path, lab)
    print(f"  SAM {mode} ({image}) done: {lab.max()} raw objects, {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
