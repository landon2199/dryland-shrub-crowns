"""Derived headline numbers for the README, computed from results/metrics.json only.

    python -m shrubs.summary          # rewrites results/summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(m: dict) -> dict:
    out: dict = {}
    s = m["scaling"]
    slope = s["dry"]["msavi_10m"]["sensitivity"]["slope_per_1pct"]
    out["jornada_dry_msavi_slope_per_1pct"] = slope
    out["expected_msavi_excess_at_cover_pct"] = {c: slope * c for c in (6.3, 11.1, 12.6, 25, 40)}
    out["dry_10m_auc_class_10_15pct"] = s["dry"]["msavi_10m"]["separability"][5]
    out["dry_10m_auc_class_15_20pct"] = s["dry"]["msavi_10m"]["separability"][6]
    a0 = m["accuracy_test_blocks"][f"baseline@{m['settings']['ref_height']:.1f}m"]
    test_ha = a0["test_blocks"] * m["settings"]["block"] ** 2 / 1e4
    out["test_area_ha"] = test_ha
    out["test_lidar_crowns_per_ha"] = a0["n_ref"]["est"] / test_ha
    out["monsoon_msavi_10m_max_auc"] = max(r["auc"] for r in s["monsoon"]["msavi_10m"]["separability"])
    if "aral" not in m:
        return out
    a = m["aral"]
    years = sorted(a["years"])
    groups = list(a["groups"])
    exc = {g: {y: a["years"][y]["plots"][g]["msavi"] - a["years"][y]["control"]["msavi"]["median"] for y in years}
           for g in groups}
    out["aral_msavi_excess_by_year"] = exc
    out["aral_msavi_excess_mean_2018_2025"] = {g: _mean(exc[g].values()) for g in groups}
    out["aral_msavi_excess_sd_2018_2025"] = {g: float(np.std(list(exc[g].values()), ddof=1)) for g in groups}
    ctrl = [a["years"][y]["control"]["msavi"]["median"] for y in years]
    out["aral_control_msavi_range"] = [min(ctrl), max(ctrl)]
    out["aral_n_dates_range"] = [min(a["years"][y]["n_dates"] for y in years), max(a["years"][y]["n_dates"] for y in years)]
    diffs, cdiffs = [], []
    for y in years:
        c = a["years"][y].get("scl_masked_check")
        if c:
            cdiffs.append(abs(c["control_msavi"] - a["years"][y]["control"]["msavi"]["median"]))
            diffs += [abs(c["plots_msavi"][g] - a["years"][y]["plots"][g]["msavi"]) for g in groups]
    out["aral_scl_check_2022_2025"] = {"max_abs_diff_plot_msavi": max(diffs), "median_abs_diff_plot_msavi": float(np.median(diffs)),
                                       "max_abs_diff_control_msavi": max(cdiffs)}
    for idx in ("si_khan", "brightness", "ndsi"):
        out[f"aral_{idx}_excess_mean_2018_2025"] = {
            g: _mean([a["years"][y]["plots"][g][idx] - a["years"][y]["control"][idx]["median"] for y in years]) for g in groups}
        out[f"aral_{idx}_control_range"] = [min(a["years"][y]["control"][idx]["median"] for y in years),
                                            max(a["years"][y]["control"][idx]["median"] for y in years)]
    s1y = [y for y in years if "s1" in a["years"][y]]
    for p in ("vv", "vh"):
        out[f"aral_s1_{p}_excess_db_mean"] = {
            g: _mean([a["years"][y]["s1"]["plots"][g][p] - a["years"][y]["s1"]["control"][p] for y in s1y]) for g in groups}
    out["aral_s1_years"] = s1y
    sv = a["survey_month_sep2020"]["plots"]
    pairs = [(a["groups"][g]["survey_cover_2020"], sv[g]["msavi_minus_control"]) for g in groups
             if a["groups"][g]["survey_cover_2020"] is not None]
    x, y = np.array(pairs).T
    out["aral_sep2020_survey_vs_msavi"] = {"n": len(pairs), "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                                           "max_excess": float(y.max()), "excess_plot1": sv["1"]["msavi_minus_control"]}
    return out


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    m = json.loads((root / "results" / "metrics.json").read_text())
    s = summarize(m)
    (root / "results" / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
