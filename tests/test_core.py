"""Fast offline tests (synthetic data only; no downloads, no GPU)."""
import numpy as np
import pytest

from shrubs import aral as A
from shrubs import detect as Dt
from shrubs import evaluate as E
from shrubs import grid as G
from shrubs import imagery as I
from shrubs import lidar as L
from shrubs import scaling as S


def disk(shape, cy, cx, r):
    yy, xx = np.indices(shape)
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def bump(shape, cy, cx, sigma, h=1.0):
    yy, xx = np.indices(shape)
    return h * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))


# ----------------------------------------------------------------------------- grid
def test_site_grid_nests_in_sentinel2_grid():
    g = G.site_grid()
    g10 = G.site_grid(10.0)
    assert (g.width, g.height) == (6000, 6000)
    assert (g10.width, g10.height) == (300, 300)
    assert g.bounds == g10.bounds
    assert g.x0 % 10 == 0 and g.y1 % 10 == 0


def test_block_ids_and_split():
    b = G.block_ids((8, 8), 0.5, 2.0)  # 4 px blocks -> 2 x 2
    assert b.max() == 3 and b[0, 0] == 0 and b[0, 7] == 1 and b[7, 0] == 2
    dev, test = G.split_blocks(144, 0.5, 42)
    assert len(test) == 72 and len(np.intersect1d(dev, test)) == 0
    assert sorted(np.concatenate([dev, test])) == list(range(144))


def test_aggregate():
    a = np.arange(16, dtype=float).reshape(4, 4)
    assert G.aggregate(a, 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]
    assert G.aggregate(a, 2, "sum")[0, 0] == 10


# ----------------------------------------------------------------------------- lidar
def test_grid_reduce_min_max():
    v = np.array([3.0, 1.0, 5.0, 2.0])
    r = np.array([0, 0, 1, 1])
    c = np.array([0, 0, 1, 1])
    mn = L.grid_reduce(v, r, c, (2, 2), "min")
    mx = L.grid_reduce(v, r, c, (2, 2), "max")
    assert mn[0, 0] == 1 and mx[0, 0] == 3 and mx[1, 1] == 5 and np.isnan(mn[0, 1])


def test_terrain_removes_a_shrub():
    ground = np.full((20, 20), 100.0)
    z = ground.copy()
    z[9:11, 9:11] += 1.5  # a 2 x 2 m shrub with no ground return under it
    dtm = L.terrain_from_zmin(z)
    assert np.abs(dtm - 100).max() < 1e-9


def test_segment_crowns_splits_two_touching_shrubs():
    h = bump((40, 60), 20, 22, 4) + bump((40, 60), 20, 38, 4)
    lab = L.segment_crowns(h, h > 0.3, min_distance_px=3, sigma=0)
    assert lab.max() == 2


def test_reference_crowns_min_area():
    chm = np.zeros((40, 40), np.float32)
    chm[5:7, 5:7] = 1.0  # 4 px = 1 m2 at 0.5 m
    chm[20:24, 20:24] = 1.0  # 16 px = 4 m2
    lab = L.reference_crowns(chm, 0.3, 2.0, 0.5)
    assert lab.max() == 1 and lab[22, 22] > 0 and lab[6, 6] == 0


def test_relabel_consecutive():
    a = np.array([[0, 5], [9, 5]])
    r = L.relabel(a)
    assert sorted(np.unique(r)) == [0, 1, 2]


# ----------------------------------------------------------------------------- indices / imagery
def test_msavi_and_ndvi_values():
    assert I.ndvi(np.array(0.5), np.array(0.1)) == pytest.approx(0.4 / 0.6)
    # MSAVI2 closed form for nir=0.5, red=0.1: (2 - sqrt(4 - 3.2)) / 2
    assert I.msavi(np.array(0.5), np.array(0.1)) == pytest.approx((2 - np.sqrt(0.8)) / 2)
    assert I.msavi(np.array(0.3), np.array(0.3)) == pytest.approx(0.0, abs=1e-12)


def test_coregistration_recovers_known_shift():
    rng = np.random.default_rng(0)
    fixed = np.zeros((256, 256))
    for _ in range(60):
        fixed += bump((256, 256), rng.uniform(20, 236), rng.uniform(20, 236), 2.5)
    moving = np.roll(np.roll(fixed, 3, axis=0), -2, axis=1)  # 3 rows down (south), 2 cols left (west)
    dx, dy = I.coregistration_shift(moving, fixed, res=0.5)
    assert dx == pytest.approx(1.0, abs=0.06) and dy == pytest.approx(1.5, abs=0.06)


def test_s2_indices_salinity():
    refl = {k: np.array([0.2]) for k in ("B02", "B03", "B04", "B08", "B11", "B12")}
    refl["B08"] = np.array([0.3])
    ix = I.s2_indices(refl)
    assert ix["si_khan"][0] == pytest.approx(0.2)
    assert ix["ndsi"][0] == pytest.approx(-0.2)


# ----------------------------------------------------------------------------- detection helpers
def test_tile_starts_cover_edges():
    s = Dt.tile_starts(1000, 256, 192)
    assert s[0] == 0 and s[-1] == 1000 - 256
    assert all(b - a <= 192 for a, b in zip(s, s[1:]))


def test_core_bounds_partition_without_gaps():
    n, size, step = 1000, 256, 192
    cores = [Dt.core_bounds(s, size, n, 32) for s in Dt.tile_starts(n, size, step)]
    assert cores[0][0] == 0 and cores[-1][1] == n
    for (a0, a1), (b0, b1) in zip(cores, cores[1:]):
        assert b0 <= a1  # neighbouring cores meet or overlap


def test_paint_best_score_first_and_core_rule():
    label = np.zeros((20, 20), np.int32)
    big = disk((10, 10), 5, 5, 4)
    small = disk((10, 10), 5, 5, 3)  # covers more than 30 % of `big`
    nid = Dt.paint(label, [big, small], [0.5, 0.9], 0, 0, (0, 20, 0, 20), 1)
    assert nid == 2  # small (better score) painted, big rejected as mostly taken
    assert label[5, 5] == 1 and label[5, 1] == 0
    small = disk((10, 10), 5, 5, 2)
    label2 = np.zeros((20, 20), np.int32)
    Dt.paint(label2, [small], [0.9], 0, 0, (10, 20, 10, 20), 1)  # centroid outside core
    assert label2.max() == 0


def test_index_watershed_and_filter():
    idx = bump((60, 60), 20, 20, 3) + bump((60, 60), 40, 40, 3, h=0.4)
    lab = Dt.index_watershed(idx, 0.2, sigma=0, min_px=4)
    assert lab.max() == 2
    kept = Dt.filter_labels(lab, idx, min_mean_index=0.5, min_px=4)
    assert kept.max() == 1 and kept[20, 20] == 1


def test_stretch_uint8():
    a = np.stack([np.linspace(0, 200, 100).reshape(10, 10)] * 4).astype(np.uint8)
    s = Dt.stretch(a)
    assert s.shape == (10, 10, 3) and s.dtype == np.uint8 and s.max() == 255


# ----------------------------------------------------------------------------- evaluation
def two_scene():
    ref = np.zeros((40, 40), np.int32)
    ref[disk((40, 40), 10, 10, 3)] = 1
    ref[disk((40, 40), 30, 30, 3)] = 2
    pred = np.zeros_like(ref)
    pred[disk((40, 40), 10, 11, 3)] = 1  # shifted 1 px: still a match
    pred[disk((40, 40), 10, 30, 3)] = 2  # false positive
    return ref, pred


def test_match_iou():
    ref, pred = two_scene()
    m = E.match(ref, pred, 0.3)
    assert m["ref_id"].tolist() == [1] and m["pred_id"].tolist() == [1]
    assert 0.5 < m["iou"][0] < 1
    assert len(E.match(ref, ref, 0.99)["ref_id"]) == 2


def test_block_tallies_metrics():
    ref, pred = two_scene()
    blocks = G.block_ids(ref.shape, 0.5, 10.0)  # 20 px blocks -> 4 blocks
    t = E.block_tallies(ref, pred, blocks, 4, 0.5, 0.3)
    m = E.metrics_from_tallies(t, np.arange(4))
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5) and m["count_ratio"] == pytest.approx(1.0)
    assert m["cover_ratio"] == pytest.approx(1.0)


def test_block_bootstrap_interval_brackets_estimate():
    ref, pred = two_scene()
    blocks = G.block_ids(ref.shape, 0.5, 5.0)
    t = E.block_tallies(ref, pred, blocks, int(blocks.max()) + 1, 0.5, 0.3)
    b = E.block_bootstrap(t, np.arange(int(blocks.max()) + 1), n_boot=200, seed=1)
    for k in ("cover_ref", "cover_pred"):
        assert b[k]["lo"] <= b[k]["est"] <= b[k]["hi"]


def test_cell_agreement_perfect():
    a = np.array([0.0, 0.1, 0.2, 0.3])
    r = E.cell_agreement(a, a)
    assert r["r2_1to1"] == pytest.approx(1.0) and r["rmse"] == 0 and r["bias"] == 0


# ----------------------------------------------------------------------------- scaling
def test_auc_known_values():
    assert S.auc([2, 3, 4], [0, 1]) == 1.0
    assert S.auc([0, 1], [2, 3]) == 0.0
    assert S.auc([1, 1], [1, 1]) == 0.5
    rng = np.random.default_rng(0)
    assert abs(S.auc(rng.normal(size=4000), rng.normal(size=4000)) - 0.5) < 0.03


def test_sensitivity_and_detection_limit():
    rng = np.random.default_rng(1)
    cover = rng.uniform(0, 0.4, 20000)
    cover[:5000] = 0.0
    idx = 0.08 + 0.1 * cover + rng.normal(0, 0.002, cover.size)  # +0.001 per 1 % cover, sd 0.002
    s = S.sensitivity(cover, idx)
    assert s["slope_per_1pct"] == pytest.approx(0.001, rel=0.05)
    assert s["lod_3sigma_cover"] == pytest.approx(0.06, rel=0.1)
    rows = S.separability(cover, idx, n_boot=0)
    lim = S.detection_limit(rows)
    assert lim is not None and 0.01 <= lim <= 0.06


def test_planting_arithmetic():
    assert S.planting_cover(1.0, 600) == pytest.approx(0.06)
    assert S.crown_area_for_cover(0.05) == pytest.approx(5.0)


# ----------------------------------------------------------------------------- Aral
def test_plot_cover_reported_or_derived():
    assert A.plot_cover(A.PLOTS[1]) == (0.40, "reported")
    c, src = A.plot_cover(A.PLOTS[17])
    assert c == pytest.approx(0.042) and src.startswith("derived")
    assert A.plot_cover(A.PLOTS[21])[0] is None


def test_plot_masks_radius():
    g = G.Grid(x0=0.0, y1=1000.0, width=100, height=100, res=10.0, crs=A.AREA_CRS)
    masks, ctrl = A.plot_masks(g, {1: (500.0, 500.0)}, radius=50.0, control_min=300.0)
    assert 70 <= masks[1].sum() <= 90  # pi * 5^2 ~ 78 pixels
    assert not ctrl[50, 50] and ctrl[0, 0]
