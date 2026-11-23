"""Shrub crowns on sub-metre imagery, and what Sentinel-2 can see of them.

    python run.py                 # everything (lidar reference, NAIP, detection, scaling, Aral)
    python run.py --skip-sam      # no GPU: baseline only
    python run.py --skip-aral     # Jornada part only

Writes results/metrics.json, results/*.csv and figures/*.png. Inputs are cached in data/.
"""
from __future__ import annotations


import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from shrubs import aral as A
from shrubs import detect as Dt
from shrubs import evaluate as E
from shrubs import grid as G
from shrubs import imagery as I
from shrubs import lidar as L
from shrubs import scaling as S

ROOT = Path(__file__).parent
DATA, RES, FIG = ROOT / "data", ROOT / "results", ROOT / "figures"
SAM_CKPT = DATA / "weights" / "sam_vit_h_4b8939.pth"
ARAL_MAX_CLOUD = 60.0
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def cached(path: Path, fn):
    if path.exists():
        return np.load(path)
    a = fn()
    np.save(path, a)
    return a


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def tune(ref, blocks, nb, dev, make, grid_values) -> tuple:
    """Pick the parameter with the best F1 on the development blocks."""
    rows = []
    for v in grid_values:
        lab = make(v)
        m = E.metrics_from_tallies(E.block_tallies(ref, lab, blocks, nb, G.FINE), dev)
        rows.append({"value": v, "f1": float(m["f1"]), "precision": float(m["precision"]), "recall": float(m["recall"])})
    best = max(rows, key=lambda r: r["f1"])
    return best["value"], rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block", type=float, default=250.0, help="spatial block size, m")
    ap.add_argument("--test-frac", type=float, default=0.5)
    ap.add_argument("--ref-height", type=float, default=0.3, help="main reference: CHM at least this tall, m")
    ap.add_argument("--alt-ref-height", type=float, default=0.5, help="stricter reference for sensitivity, m")
    ap.add_argument("--min-crown", type=float, default=2.0, help="minimum crown area, m2 (reference and detections)")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-sam", action="store_true")
    ap.add_argument("--skip-aral", action="store_true")
    args = ap.parse_args()
    for d in (DATA, RES, FIG):
        d.mkdir(exist_ok=True)
    t_all = time.time()
    out: dict = {"settings": vars(args)}
    g = G.site_grid()
    min_px = int(round(args.min_crown / G.FINE**2))

    # ---------------------------------------------------------------- 1. lidar reference
    t = time.time()
    if not (DATA / "chm.npy").exists():
        laz = L.download_tiles(g, DATA / "neon" / "laz")
        built = L.build_chm(g, laz)
        np.save(DATA / "chm.npy", built["chm"])
        np.save(DATA / "dtm.npy", built["dtm"])
        (DATA / "chm_qa.json").write_text(json.dumps(built["qa"], indent=2))
    chm = np.load(DATA / "chm.npy")
    out["lidar_qa"] = json.loads((DATA / "chm_qa.json").read_text())
    refs = {h: cached(DATA / f"ref_{h:.1f}m.npy", lambda h=h: L.reference_crowns(chm, h, args.min_crown, G.FINE))
            for h in (args.ref_height, args.alt_ref_height)}
    ref = refs[args.ref_height]
    log(f"lidar reference: {ref.max()} crowns >= {args.ref_height} m, {refs[args.alt_ref_height].max()} >= "
        f"{args.alt_ref_height} m ({time.time() - t:.0f} s)")

    # ---------------------------------------------------------------- 2. NAIP + co-registration
    t = time.time()
    raw, _ = I.fetch_naip(g, 2022, DATA / "naip_2022_raw.npy")
    norm = lambda a: (lambda z: (z - z.mean()) / z.std())(ndimage.gaussian_filter(a, 1))
    target = norm(np.clip(chm, 0, 2))
    shift = I.coregistration_shift(norm(I.naip_indices(raw)["msavi"]), target, G.FINE)
    quads = [I.coregistration_shift(norm(I.naip_indices(raw[:, r:r + 3000, c:c + 3000])["msavi"]),
                                    target[r:r + 3000, c:c + 3000], G.FINE) for r in (0, 3000) for c in (0, 3000)]
    del raw
    shift = (round(shift[0], 2), round(shift[1], 2))
    naip, naip_meta = I.fetch_naip(g, 2022, DATA / "naip_2022.npy", shift_m=shift)
    idx = I.naip_indices(naip)
    resid = I.coregistration_shift(norm(idx["msavi"]), target, G.FINE)
    out["naip"] = {**naip_meta, "shift_applied_m": shift, "shift_by_quadrant_m": quads, "residual_shift_m": resid}
    # lidar height noise away from green plants: CHM in cells more than 3 m from any pixel whose smoothed
    # NAIP MSAVI is above 0 (the basis for the 0.3 m reference height)
    far = ndimage.distance_transform_edt(ndimage.gaussian_filter(idx["msavi"], 1) < 0.0) > 6
    out["lidar_qa"]["chm_far_from_green"] = {"share_of_cells": float(far.mean()),
                                             **{f"p{q}": float(np.percentile(chm[far], q)) for q in (50, 90, 95, 99)}}
    del far
    log(f"NAIP {naip_meta['dates']}, shift {shift} m, residual {resid} ({time.time() - t:.0f} s)")

    # ---------------------------------------------------------------- 3. blocks
    blocks = G.block_ids(chm.shape, G.FINE, args.block)
    nb = int(blocks.max()) + 1
    dev, test = G.split_blocks(nb, args.test_frac, args.seed)
    out["blocks"] = {"size_m": args.block, "n": nb, "dev": dev.tolist(), "test": test.tolist()}
    np.save(RES / "blocks_test.npy", test)

    # ---------------------------------------------------------------- 4. baseline
    t = time.time()
    methods: dict[str, np.ndarray] = {}
    tuning: dict = {}
    best = None
    for name in ("msavi", "ndvi"):
        sm = ndimage.gaussian_filter(idx[name], 1)
        qs = [86, 88, 90, 92, 94, 96]
        q, rows = tune(ref, blocks, nb, dev, lambda q: Dt.index_watershed(idx[name], np.percentile(sm, q), min_px=min_px), qs)
        f1 = max(r["f1"] for r in rows)
        tuning[f"baseline_{name}"] = rows
        if best is None or f1 > best[2]:
            best = (name, q, f1, float(np.percentile(sm, q)))
    name, q, _, thr = best
    methods["baseline"] = Dt.index_watershed(idx[name], thr, min_px=min_px)
    out["baseline"] = {"index": name, "percentile": q, "threshold": thr}
    log(f"baseline: {name} > {thr:.4f} (percentile {q}), {methods['baseline'].max()} crowns ({time.time() - t:.0f} s)")
    msavi_s = ndimage.gaussian_filter(idx["msavi"], 1)

    # ---------------------------------------------------------------- 5. SAM
    if not args.skip_sam:
        if not SAM_CKPT.exists():
            import urllib.request

            SAM_CKPT.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(Dt.SAM_URL_VIT_H, SAM_CKPT)
        prompt_q = 85
        peaks = Dt.index_peaks(idx["msavi"], np.percentile(msavi_s, prompt_q))
        out["sam_points"] = {"prompt_threshold_percentile": prompt_q, "n_prompts": int(len(peaks)), "by_image": {}}

        def run_sam(mode, image, path):
            """SAM runs in a child process (see shrubs/sam_stage.py for why)."""
            if path.exists():
                return np.load(path)
            cmd = [sys.executable, "-u", "-m", "shrubs.sam_stage", str(DATA / "naip_2022.npy"), str(path), mode,
                   image, str(prompt_q), str(SAM_CKPT)]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            for line in proc.stdout.splitlines():
                if "done" in line or "prompt points" in line:
                    log(line.strip())
            if proc.returncode != 0:
                raise RuntimeError("SAM stage failed: " + proc.stderr[-3000:])
            return np.load(path)

        thr_grid = [-0.10, -0.08, -0.06, -0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02]

        # point prompts on both image versions; the better one on the dev blocks is kept
        best_pts = None
        for iname in ("truecolor", "falsecolor"):
            t = time.time()
            raw = run_sam("points", iname, DATA / f"sam_points_raw_{iname}.npy")
            thr, rows = tune(ref, blocks, nb, dev, lambda v: Dt.filter_labels(raw, msavi_s, v, min_px, 800), thr_grid)
            f1 = max(r["f1"] for r in rows)
            tuning[f"sam_points_{iname}_min_mean_msavi"] = rows
            out["sam_points"]["by_image"][iname] = {"min_mean_msavi": thr, "dev_f1": f1, "raw_objects": int(raw.max()),
                                                    "seconds_or_cached": round(time.time() - t, 1)}
            log(f"SAM points {iname}: dev F1 {f1:.3f} ({time.time() - t:.0f} s)")
            if best_pts is None or f1 > best_pts[1]:
                best_pts = (iname, f1, thr, raw)
        iname, _, thr_pts, raw_pts = best_pts
        methods["sam_points"] = Dt.filter_labels(raw_pts, msavi_s, thr_pts, min_px, 800)
        out["sam_points"].update({"image": iname, "min_mean_msavi": thr_pts})
        log(f"SAM points: {iname}, {methods['sam_points'].max()} crowns")
        del best_pts, raw_pts

        # automatic SAM (about five times slower) on the image version chosen above
        t = time.time()
        raw_auto = run_sam("auto", iname, DATA / f"sam_auto_raw_{iname}.npy")
        thr_auto, rows = tune(ref, blocks, nb, dev, lambda v: Dt.filter_labels(raw_auto, msavi_s, v, min_px, 800), thr_grid)
        tuning["sam_auto_min_mean_msavi"] = rows
        methods["sam_auto"] = Dt.filter_labels(raw_auto, msavi_s, thr_auto, min_px, 800)
        out["sam_auto"] = {"image": iname, "min_mean_msavi": thr_auto, "raw_objects": int(raw_auto.max()),
                           "seconds_or_cached": round(time.time() - t, 1)}
        log(f"SAM auto: {methods['sam_auto'].max()} crowns ({time.time() - t:.0f} s)")
        del raw_auto
    out["tuning_dev_blocks"] = tuning
    for k, v in methods.items():
        np.save(DATA / f"labels_{k}.npy", v)

    # ---------------------------------------------------------------- 6. accuracy on held-out blocks
    t = time.time()
    acc = {}
    for h, rlab in refs.items():
        for mname, lab in methods.items():
            tal = E.block_tallies(rlab, lab, blocks, nb, G.FINE, args.iou)
            acc[f"{mname}@{h:.1f}m"] = {"test_blocks": int(len(test)), **E.block_bootstrap(tal, test, args.n_boot, args.seed)}
    # IoU 0.5 for the main reference
    for mname, lab in methods.items():
        tal = E.block_tallies(ref, lab, blocks, nb, G.FINE, 0.5)
        acc[f"{mname}@{args.ref_height:.1f}m_iou0.5"] = {"test_blocks": int(len(test)),
                                                       **E.block_bootstrap(tal, test, args.n_boot, args.seed)}
    out["accuracy_test_blocks"] = acc
    out["size_bins_m2"] = list(E.SIZE_BINS_M2[:-1])

    # per-hectare counts and 10 m cover on test blocks
    test_px = np.isin(blocks, test)
    ha = G.aggregate(test_px.astype(float), 200, "mean") == 1  # 100 m cells fully in test blocks
    c10_test = G.aggregate(test_px.astype(float), 20, "mean") == 1

    def count_per_cell(lab, f):
        st = E.object_stats(lab, np.zeros(lab.shape, np.int32), G.FINE)
        r, c = (st["row"] // f).astype(int), (st["col"] // f).astype(int)
        n = lab.shape[0] // f
        return np.bincount(r * n + c, minlength=n * n).reshape(n, n).astype(float)

    n_ref_ha = count_per_cell(ref, 200)
    cover = {"reference": G.aggregate((ref > 0).astype(float), 20)}
    out["per_hectare_counts_test"] = {}
    out["cover_10m_test"] = {}
    for mname, lab in methods.items():
        out["per_hectare_counts_test"][mname] = E.cell_agreement(n_ref_ha[ha], count_per_cell(lab, 200)[ha])
        cover[mname] = G.aggregate((lab > 0).astype(float), 20)
        out["cover_10m_test"][mname] = E.cell_agreement(cover["reference"][c10_test], cover[mname][c10_test])
    log(f"accuracy done ({time.time() - t:.0f} s)")

    # ---------------------------------------------------------------- 7. scaling bridge
    t = time.time()
    g10 = G.site_grid(G.COARSE)
    blocks10 = G.block_ids((g10.height, g10.width), G.COARSE, args.block)
    s2 = {}
    scal = {}
    for season, (a, b) in {"dry": ("2022-05-15", "2022-06-30"), "monsoon": ("2022-08-25", "2022-10-10")}.items():
        sc = I.s2_scenes(g10, a, b, max_cloud=20)
        comp = I.composite(sc)
        ix = I.s2_indices(comp)
        s2[season] = {"msavi": ix["msavi"], "ndvi": ix["ndvi"]}
        np.save(DATA / f"s2_{season}_msavi.npy", ix["msavi"])
        np.save(DATA / f"s2_{season}_ndvi.npy", ix["ndvi"])
        scal[season] = {"dates": sc["dates"], "min_clear_dates_per_pixel": int(comp["n_clear"].min())}
        for iname in ("msavi", "ndvi"):
            for fct in (1, 2, 3):  # 10, 20, 30 m
                cv = G.aggregate(cover["reference"], fct) if fct > 1 else cover["reference"]
                iv = G.aggregate(ix[iname], fct) if fct > 1 else ix[iname]
                bl = blocks10[::fct, ::fct] if fct > 1 else blocks10
                sep = S.separability(cv, iv, bl, n_boot=300 if fct == 1 else 0, seed=args.seed)
                key = f"{iname}_{10 * fct}m"
                scal[season][key] = {"separability": sep, "detection_limit_auc0.8": S.detection_limit(sep),
                                     "sensitivity": S.sensitivity(cv, iv)}
            scal[season][f"{iname}_10m"]["binned"] = S.binned(cover["reference"], ix[iname])
        # what detected cover from NAIP would give against the same Sentinel-2 pixels
        for mname in methods:
            if mname in cover:
                scal[season][f"msavi_10m_vs_{mname}_cover"] = S.sensitivity(cover[mname], ix["msavi"])
    scal["reference_cover_share_of_cells"] = {
        "bare_lt_1pct": float((cover["reference"] < 0.01).mean()),
        "lt_5pct": float((cover["reference"] < 0.05).mean()),
        "lt_10pct": float((cover["reference"] < 0.10).mean()),
        "median_cover": float(np.median(cover["reference"])),
        "p90_cover": float(np.percentile(cover["reference"], 90)),
    }
    np.save(DATA / "cover10_reference.npy", cover["reference"])
    out["scaling"] = scal
    log(f"scaling done ({time.time() - t:.0f} s)")

    # ---------------------------------------------------------------- 8. Aral add-on
    if not args.skip_aral:
        t = time.time()
        out["aral"] = run_aral()
        log(f"Aral done ({time.time() - t:.0f} s)")

    # ---------------------------------------------------------------- 9. figures + save
    from shrubs import plots

    t = time.time()
    plots.make_all(out, naip=naip, chm=chm, ref=ref, methods=methods, cover=cover, s2=s2, blocks=blocks,
                   test=test, fig_dir=FIG, data_dir=DATA)
    log(f"figures done ({time.time() - t:.0f} s)")
    out["runtime_s"] = round(time.time() - t_all, 1)
    (RES / "metrics.json").write_text(json.dumps(jsonable(out), indent=1))
    from shrubs.summary import summarize

    (RES / "summary.json").write_text(json.dumps(summarize(json.loads((RES / "metrics.json").read_text())), indent=1))
    (ROOT / "run_log.txt").write_text("\n".join(LOG) + "\n")
    log(f"done in {out['runtime_s']:.0f} s")


def run_aral() -> dict:
    g = A.area_grid()
    xy = A.plot_xy()
    masks, ctrl = A.plot_masks(g, xy)
    groups = {"1": [1], "2": [2], "3+4": [3, 4], "17+18": [17, 18], "19": [19], "20+21": [20, 21],
              "22": [22], "23": [23]}
    gmask = {k: np.any([masks[p] for p in v], axis=0) for k, v in groups.items()}
    res = {"box_wgs84": A.BOX, "plot_radius_m": A.PLOT_RADIUS_M, "control_min_dist_m": A.CONTROL_MIN_DIST_M,
           "s2_max_scene_cloud_pct": ARAL_MAX_CLOUD, "s2_cloud_mask": "none (median of all dates)",
           "groups": {}, "years": {}}
    for k, v in groups.items():
        covers = [A.plot_cover(A.PLOTS[p]) for p in v]
        vals = [c for c, _ in covers if c is not None]
        res["groups"][k] = {"plots": v, "planted": [A.PLOTS[p][2] for p in v],
                            "survey_cover_2020": float(np.mean(vals)) if vals else None,
                            "cover_source": "; ".join(sorted({s for _, s in covers})),
                            "survival_pct": [A.PLOTS[p][6] for p in v], "pixels": int(gmask[k].sum())}
    series = {}
    for year in range(2018, 2026):
        a, b = A.season_window(year)
        # Salt crust is labelled cloud by the older scene classification (and inflates scene cloud
        # cover), so: scenes up to 60 % "cloud", median of all dates, no SCL mask (see README)
        sc = I.s2_scenes(g, a, b, max_cloud=ARAL_MAX_CLOUD)
        comp = I.composite(sc, use_scl=False)
        ix = I.s2_indices(comp)
        yr = {"n_dates": len(sc["dates"]), "control": {}, "plots": {}}
        if year >= 2022:  # newer processing classifies the seabed as bare soil: compare with the SCL mask
            ixm = I.s2_indices(I.composite(sc, use_scl=True))
            yr["scl_masked_check"] = {
                "control_msavi": float(np.nanmedian(ixm["msavi"][ctrl])),
                "plots_msavi": {k: float(np.nanmedian(ixm["msavi"][m])) for k, m in gmask.items()},
                "control_si_khan": float(np.nanmedian(ixm["si_khan"][ctrl]))}
        for iname in ("msavi", "ndvi", "brightness", "si_khan", "ndsi"):
            v = ix[iname]
            yr["control"][iname] = {"median": float(np.nanmedian(v[ctrl])),
                                    "p90": float(np.nanpercentile(v[ctrl], 90))}
            for k, m in gmask.items():
                yr["plots"].setdefault(k, {})[iname] = float(np.nanmedian(v[m]))
        s1 = A.s1_composite(g, a, b)
        if s1:
            yr["s1"] = {"orbit_state": s1["orbit_state"], "n_dates": s1["n_dates"],
                        "control": {p: float(np.nanmedian(s1[p][ctrl])) for p in ("vv", "vh")},
                        "plots": {k: {p: float(np.nanmedian(s1[p][m])) for p in ("vv", "vh")} for k, m in gmask.items()}}
        series[str(year)] = yr
        if year == 2020:
            np.save(DATA / "aral_msavi_2020.npy", ix["msavi"])
        if year == 2025:
            np.save(DATA / "aral_msavi_2025.npy", ix["msavi"])
        log(f"  Aral {year}: {len(sc['dates'])} dates")
    res["years"] = series
    # survey month: 1-30 Sep 2020 composite vs surveyed cover
    sc = I.s2_scenes(g, "2020-09-01", "2020-09-30", max_cloud=ARAL_MAX_CLOUD)
    ix = I.s2_indices(I.composite(sc, use_scl=False))
    ctrl_m = float(np.nanmedian(ix["msavi"][ctrl]))
    res["survey_month_sep2020"] = {"dates": sc["dates"], "control_msavi": ctrl_m,
                                   "plots": {k: {"msavi": float(np.nanmedian(ix["msavi"][m])),
                                                 "msavi_minus_control": float(np.nanmedian(ix["msavi"][m]) - ctrl_m)}
                                             for k, m in gmask.items()}}
    return res


if __name__ == "__main__":
    main()
