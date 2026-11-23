"""Figures. Colours follow one validated categorical order (blue, orange, aqua; checked with the
dataviz skill's validator for all pairs, light surface): SAM points = blue, baseline = orange,
SAM automatic = aqua. The lidar reference is drawn in ink, Sentinel-2 maps in one-hue ramps.
Aqua is under 3:1 contrast on white, so every aqua mark is also direct-labelled."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SURFACE = "#fcfcfb"
COL = {"sam_points": "#2a78d6", "baseline": "#eb6834", "sam_auto": "#1baf7a"}
NAME = {"sam_points": "SAM, point prompts", "baseline": "Index + watershed", "sam_auto": "SAM, automatic"}
SEASON = {"dry": ("Dry season (15 May - 30 Jun)", "#2a78d6"), "monsoon": ("Monsoon (25 Aug - 10 Oct)", "#eb6834")}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def _blues():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("blues", BLUES)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def window_rc(chm_shape, row0, col0, size):
    return slice(row0, row0 + size), slice(col0, col0 + size)


def hero(out, naip, ref, methods, cover, s2, path, win=(2600, 2600, 200)):
    """NAIP with lidar crowns and SAM detections | Sentinel-2 10 m MSAVI of the same 100 m |
    separability curve."""
    from shrubs.detect import stretch

    r0, c0, n = win
    R, C = slice(r0, r0 + n), slice(c0, c0 + n)
    fig = plt.figure(figsize=(14, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.25], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    ax.imshow(stretch(naip[:, R, C]), extent=(0, n * 0.5, 0, n * 0.5))
    ax.contour(np.flipud(ref[R, C] > 0), [0.5], colors=["#ffffff"], linewidths=1.6,
               extent=(0, n * 0.5, 0, n * 0.5))
    ax.contour(np.flipud(ref[R, C] > 0), [0.5], colors=[INK], linewidths=0.8, extent=(0, n * 0.5, 0, n * 0.5))
    m = "sam_points" if "sam_points" in methods else "baseline"
    ax.contour(np.flipud(methods[m][R, C] > 0), [0.5], colors=[COL[m]], linewidths=1.0,
               extent=(0, n * 0.5, 0, n * 0.5))
    ax.set_title("A. NAIP 0.6 m, 100 m x 100 m")
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    ax.grid(False)
    from matplotlib.lines import Line2D

    ax.legend(handles=[Line2D([], [], color=INK, lw=1.2, label="Lidar crowns (>= 0.3 m tall)"),
                       Line2D([], [], color=COL[m], lw=1.2, label=NAME[m])],
              loc="upper left", bbox_to_anchor=(0, -0.07), ncol=1, fontsize=9)

    ax = fig.add_subplot(gs[1])
    r10, c10, n10 = r0 // 20, c0 // 20, n // 20
    v = s2["dry"]["msavi"][r10:r10 + n10, c10:c10 + n10]
    cv = cover["reference"][r10:r10 + n10, c10:c10 + n10]
    im = ax.imshow(v, cmap=_blues(), extent=(0, n * 0.5, 0, n * 0.5), interpolation="nearest")
    for i in range(n10):
        for j in range(n10):
            ax.text(j * 10 + 5, (n10 - 1 - i) * 10 + 5, f"{100 * cv[i, j]:.0f}", ha="center", va="center",
                    fontsize=7.5, color=INK if v[i, j] < np.nanpercentile(v, 70) else "#ffffff")
    ax.set_title("B. Sentinel-2 MSAVI, dry season, same 100 m")
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 50, 100])
    ax.grid(False)
    ax.set_xlabel("numbers: lidar shrub cover in each 10 m pixel, %")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("MSAVI")
    cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[2])
    _separability_axes(ax, out, "msavi_10m", label_limits=True)
    ax.set_title("C. Can a 10 m pixel tell shrubs from bare soil?")
    _save(fig, path)


def _separability_axes(ax, out, key, label_limits=False):
    for season, (label, color) in SEASON.items():
        rows = out["scaling"][season][key]["separability"]
        x = [100 * (r["cover_lo"] + min(r["cover_hi"], 0.4)) / 2 for r in rows]
        y = [r["auc"] for r in rows]
        lo = [r.get("auc_lo", r["auc"]) for r in rows]
        hi = [r.get("auc_hi", r["auc"]) for r in rows]
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, lw=2, marker="o", ms=5, label=label)
        lim = out["scaling"][season][key]["detection_limit_auc0.8"]
        if label_limits and lim is not None:
            ax.annotate(f"{100 * lim:.0f} %", (100 * lim, 0.8), xytext=(4, -14 if season == "dry" else 6),
                        textcoords="offset points", fontsize=9, color=INK2)
    ax.axhline(0.8, color=MUTED, lw=1, ls="--")
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.text(0.5, 0.505, "coin flip", fontsize=8, color=MUTED, va="bottom")
    ax.text(0.5, 0.805, "AUC 0.8", fontsize=8, color=MUTED, va="bottom")
    ax.set_xlabel("Lidar shrub cover, % (class midpoint)")
    ax.set_ylabel("AUC against bare pixels (< 1 % cover)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="upper left", fontsize=9)


def accuracy(out, path, ref_heights):
    acc = out["accuracy_test_blocks"]
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    meths = [m for m in ("sam_points", "baseline", "sam_auto") if f"{m}@{ref_heights[0]:.1f}m" in acc]
    fig, axes = plt.subplots(1, len(ref_heights), figsize=(12, 4.2), sharey=True)
    for ax in np.atleast_1d(axes):
        ax.set_ylim(0, 0.85)
    for ax, h in zip(np.atleast_1d(axes), ref_heights):
        w = 0.8 / len(meths)
        for i, m in enumerate(meths):
            a = acc[f"{m}@{h:.1f}m"]
            y = [a[k]["est"] for k, _ in metrics]
            err = [[a[k]["est"] - a[k]["lo"] for k, _ in metrics], [a[k]["hi"] - a[k]["est"] for k, _ in metrics]]
            xs = np.arange(len(metrics)) + (i - (len(meths) - 1) / 2) * w
            ax.bar(xs, y, w * 0.92, color=COL[m], label=f"{NAME[m]} ({a['test_blocks']} test blocks)",
                   yerr=err, error_kw={"ecolor": INK2, "capsize": 2, "lw": 1})
            for x, v, e in zip(xs, y, err[1]):
                ax.text(x, v + e + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(np.arange(len(metrics)), [l for _, l in metrics], fontsize=9)
        ax.set_title(f"Reference: lidar crowns >= {h} m tall")
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8, loc="upper left", ncol=1)
    np.atleast_1d(axes)[0].set_ylabel("Score on 72 held-out blocks (95 % interval)")
    _save(fig, path)


def size_recall(out, path, h):
    acc = out["accuracy_test_blocks"]
    bins = out["size_bins_m2"]
    labels = [f"{bins[i]:g}-{bins[i + 1]:g}" for i in range(len(bins) - 1)] + [f">= {bins[-1]:g}"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for m in ("sam_points", "baseline", "sam_auto"):
        k = f"{m}@{h:.1f}m"
        if k not in acc:
            continue
        ks = sorted(int(x[len("recall_size"):]) for x in acc[k] if x.startswith("recall_size"))
        ks = [s for s in ks if s >= 1]  # size class 0 (1-2 m2) is below the minimum crown size
        y = [acc[k][f"recall_size{s}"]["est"] for s in ks]
        lo = [acc[k][f"recall_size{s}"]["lo"] for s in ks]
        hi = [acc[k][f"recall_size{s}"]["hi"] for s in ks]
        x = np.arange(len(ks))
        ax.fill_between(x, lo, hi, color=COL[m], alpha=0.15, linewidth=0)
        ax.plot(x, y, color=COL[m], lw=2, marker="o", ms=6, label=NAME[m])
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xticks(np.arange(len(ks)), [labels[s] for s in ks])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Lidar crown area, m2")
    ax.set_ylabel("Share of lidar crowns found (recall)")
    ax.set_title(f"Bigger crowns are found more often (held-out blocks, lidar >= {h} m)")
    _save(fig, path)


def scaling(out, path):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    ax = axes[0]
    for season, (label, color) in SEASON.items():
        rows = out["scaling"][season]["msavi_10m"]["binned"]
        x = [100 * r["cover_mid"] for r in rows]
        ax.fill_between(x, [r["q25"] for r in rows], [r["q75"] for r in rows], color=color, alpha=0.15, linewidth=0)
        ax.plot(x, [r["median"] for r in rows], color=color, lw=2, label=label)
    ax.set_xlabel("Lidar shrub cover in the 10 m pixel, %")
    ax.set_ylabel("Sentinel-2 MSAVI (median, quartiles)")
    ax.set_title("A. MSAVI against cover")
    ax.legend(fontsize=9, loc="upper left")
    _separability_axes(axes[1], out, "msavi_10m", label_limits=True)
    axes[1].set_title("B. MSAVI, 10 m pixels")
    _separability_axes(axes[2], out, "ndvi_10m", label_limits=True)
    axes[2].set_title("C. NDVI, 10 m pixels")
    ax = axes[3]
    steps = {"10m": ("10 m", "#0d366b"), "20m": ("20 m", "#2a78d6"), "30m": ("30 m", "#6da7ec")}  # validated ordinal
    for key, (label, color) in steps.items():
        rows = out["scaling"]["dry"][f"msavi_{key}"]["separability"]
        x = [100 * (r["cover_lo"] + min(r["cover_hi"], 0.4)) / 2 for r in rows]
        lim = out["scaling"]["dry"][f"msavi_{key}"]["detection_limit_auc0.8"]
        ax.plot(x, [r["auc"] for r in rows], color=color, lw=2, marker="o", ms=5,
                label=f"{label} pixels (limit {'none' if lim is None else f'{100 * lim:.0f} %'})")
    ax.legend(loc="lower right", fontsize=9)
    ax.axhline(0.8, color=MUTED, lw=1, ls="--")
    ax.axhline(0.5, color=MUTED, lw=0.8)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Lidar shrub cover, % (class midpoint)")
    ax.set_ylabel("AUC against bare pixels")
    ax.set_title("D. Dry-season MSAVI averaged to coarser pixels")
    _save(fig, path)


def maps(cover, s2, path):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    fig.subplots_adjust(wspace=0.38)
    for ax, (arr, title, lab) in zip(axes, [
        (100 * cover["reference"], "A. Lidar shrub cover per 10 m pixel", "% cover"),
        (s2["dry"]["msavi"], "B. Sentinel-2 MSAVI, dry season 2022", "MSAVI"),
        (s2["monsoon"]["msavi"], "C. Sentinel-2 MSAVI, monsoon 2022", "MSAVI"),
    ]):
        lo, hi = np.nanpercentile(arr, [2, 98])
        im = ax.imshow(arr, cmap=_blues(), vmin=lo, vmax=hi, extent=(0, 3, 0, 3), interpolation="nearest")
        ax.set_title(title)
        ax.grid(False)
        ax.set_xlabel("km")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(lab)
        cb.outline.set_visible(False)
    _save(fig, path)


def aral(out, path):
    a = out["aral"]
    years = sorted(int(y) for y in a["years"])
    groups = list(a["groups"])
    ncol = 4
    nrow = int(np.ceil(len(groups) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.0 * nrow), sharex=True, sharey=True)
    for ax, k in zip(axes.flat, groups):
        ctrl = [a["years"][str(y)]["control"]["msavi"]["median"] for y in years]
        p90 = [a["years"][str(y)]["control"]["msavi"]["p90"] for y in years]
        v = [a["years"][str(y)]["plots"][k]["msavi"] for y in years]
        ax.fill_between(years, ctrl, p90, color=GRID, linewidth=0, label="Seabed > 600 m from plots: median to 90th pct")
        ax.plot(years, ctrl, color=MUTED, lw=1)
        ax.plot(years, v, color=INK, lw=2, marker="o", ms=4, label="Plot (50 m circle)")
        info = a["groups"][k]
        cov = info["survey_cover_2020"]
        ax.set_title(f"Plot {k}, planted {', '.join(str(p) for p in sorted(set(info['planted'])))}", fontsize=10)
        ax.text(0.03, 0.95, f"2020 survey cover: {'n/a' if cov is None else f'{100 * cov:.1f} %'}",
                transform=ax.transAxes, fontsize=8.5, va="top", color=INK2)
        for py in set(info["planted"]):
            if years[0] <= py <= years[-1]:
                ax.axvline(py, color=MUTED, lw=0.8, ls=":")
    for ax in axes.flat[len(groups):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSAVI, Aug-Sep median")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Saxaul plots on the dried Aral Sea bed, Sentinel-2 2018-2025", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    _save(fig, path)


def aral_survey(out, path):
    a = out["aral"]
    s = a["survey_month_sep2020"]["plots"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    lim = out["scaling"]["dry"]["msavi_10m"]["detection_limit_auc0.8"]
    slope = out["scaling"]["dry"]["msavi_10m"]["sensitivity"]["slope_per_1pct"]
    if lim is not None:
        ax.axvspan(0, 100 * lim, color=GRID, linewidth=0)
        ax.text(1, -0.0085, "below the Jornada 10 m" + chr(10) + f"detection limit ({100 * lim:.0f} %)",
                fontsize=8.5, color=INK2, va="bottom")
    xs = np.array([0, 42])
    ax.plot(xs, slope * xs, color=MUTED, lw=1.2, ls="--")
    ax.text(30, slope * 30 + 0.0012, "Jornada dry-season slope" + chr(10) + "(+%.5f MSAVI per 1 %% cover)" % slope,
            fontsize=8.5, color=INK2, ha="center")
    offsets = {"2": (6, -12), "20+21": (6, 4), "19": (6, -13)}
    for k, info in a["groups"].items():
        cov = info["survey_cover_2020"]
        if cov is None:
            continue
        y = s[k]["msavi_minus_control"]
        ax.plot(100 * cov, y, "o", color=INK, ms=8, mec=SURFACE, mew=2)
        ax.annotate(f"plot {k}", (100 * cov, y), xytext=offsets.get(k, (6, 2)), textcoords="offset points",
                    fontsize=9, color=INK2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-1, 43)
    ax.set_ylim(-0.009, 0.026)
    ax.set_xlabel("Saxaul cover in the 2020 field survey, %")
    ax.set_ylabel("Plot MSAVI minus seabed MSAVI, Sep 2020")
    ax.set_title("Sentinel-2 barely sees the surveyed saxaul cover")
    _save(fig, path)


def make_all(out, naip, chm, ref, methods, cover, s2, blocks, test, fig_dir: Path, data_dir: Path):
    fig_dir = Path(fig_dir)
    hs = sorted({float(k.split("@")[1].split("m")[0]) for k in out["accuracy_test_blocks"] if "@" in k})
    hero(out, naip, ref, methods, cover, s2, fig_dir / "hero.png")
    accuracy(out, fig_dir / "accuracy.png", hs)
    size_recall(out, fig_dir / "size_recall.png", out["settings"]["ref_height"])
    scaling(out, fig_dir / "scaling.png")
    maps(cover, s2, fig_dir / "maps.png")
    if "aral" in out:
        aral(out, fig_dir / "aral_series.png")
        aral_survey(out, fig_dir / "aral_survey.png")
