"""windowcov：矩形窗口覆盖分析单元测试。"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.svggeo import analyze_contours  # noqa: E402
from app.windowcov import compute_window_coverage  # noqa: E402


def _coverage(d, windows, fill_rule="nonzero"):
    contours, _warnings = analyze_contours(d, fill_rule)
    return compute_window_coverage(contours, windows)


def test_window_fully_inside_outer_full_coverage():
    # 窗口完全落在外方 0..10 内：净填充面积 = 窗口面积
    r = _coverage("M0,0h10v10h-10Z", [(2.0, 2.0, 8.0, 8.0)])
    w = r["windows"][0]
    assert w["index"] == 0
    assert w["window"] == {"xmin": 2.0, "ymin": 2.0, "xmax": 8.0, "ymax": 8.0}
    assert w["window_area"] == 36.0
    assert w["covered_area"] == 36.0
    assert w["coverage"] == 1.0
    assert r["window_count"] == 1
    assert r["contributing_contour_count"] == 1
    assert r["skipped_contours"] == []


def test_window_larger_than_contour():
    # 窗口包住整个外方：净填充面积 = 轮廓面积
    r = _coverage("M0,0h10v10h-10Z", [(-5.0, -5.0, 20.0, 20.0)])
    w = r["windows"][0]
    assert w["window_area"] == 625.0
    assert w["covered_area"] == 100.0
    assert w["coverage"] == 0.16


def test_partial_overlap():
    # 窗口 5..15 与外方 0..10 的交集为 5..10：25
    r = _coverage("M0,0h10v10h-10Z", [(5.0, 5.0, 15.0, 15.0)])
    w = r["windows"][0]
    assert w["covered_area"] == 25.0
    assert w["coverage"] == 0.25


def test_outer_minus_hole():
    # 外方 0..10 扣内孔 2..8：100 - 36 = 64
    r = _coverage("M0,0h10v10h-10Z M2,2h6v6h-6Z", [(0.0, 0.0, 10.0, 10.0)])
    w = r["windows"][0]
    assert w["covered_area"] == 64.0
    assert w["coverage"] == 0.64
    assert r["contributing_contour_count"] == 2


def test_hole_partially_clipped_by_window():
    # 窗口 0..5 x 0..10：外环裁剪 50，孔裁剪 2..5 x 2..8 = 18 -> 32
    r = _coverage("M0,0h10v10h-10Z M2,2h6v6h-6Z", [(0.0, 0.0, 5.0, 10.0)])
    w = r["windows"][0]
    assert w["covered_area"] == 32.0
    assert w["coverage"] == 0.64


def test_island_readded():
    # 外 100 - 孔 64 + 岛 4 = 40
    d = "M0,0h10v10h-10Z M1,1h8v8h-8Z M6,6h2v2h-2Z"
    r = _coverage(d, [(0.0, 0.0, 10.0, 10.0)])
    assert r["windows"][0]["covered_area"] == 40.0
    assert r["contributing_contour_count"] == 3


def test_four_level_alternation():
    # 外 100 - 孔 64 + 岛 16 - 岛中孔 4 = 48
    d = ("M0,0h10v10h-10Z M1,1h8v8h-8Z "
         "M3,3h4v4h-4Z M4,4h2v2h-2Z")
    r = _coverage(d, [(0.0, 0.0, 10.0, 10.0)])
    assert r["windows"][0]["covered_area"] == 48.0


def test_contour_enclosing_window_all_vertices_outside():
    # 大环 -10..30 完全包住窗口，四个顶点均在窗外：净填充 = 窗口面积
    r = _coverage("M-10,-10h40v40h-40Z", [(0.0, 0.0, 10.0, 10.0)])
    w = r["windows"][0]
    assert w["covered_area"] == 100.0
    assert w["coverage"] == 1.0


def test_diamond_crossing_window_vertices_outside():
    # 菱形顶点 (5,-2) (12,5) (5,12) (-2,5) 全部在窗口 0..10 之外，
    # 四条边斜穿窗口，切去四个直角边长 3 的角三角形：100 - 4*4.5 = 82
    d = "M5,-2 L12,5 L5,12 L-2,5 Z"
    r = _coverage(d, [(0.0, 0.0, 10.0, 10.0)])
    w = r["windows"][0]
    assert w["covered_area"] == 82.0
    assert w["coverage"] == 0.82


def test_open_and_degenerate_skipped():
    d = "M0,0h10v10h-10Z M5,5 L8,5 L5,5 Z M20,0 L30,0 L30,10"
    r = _coverage(d, [(0.0, 0.0, 10.0, 10.0)])
    assert r["windows"][0]["covered_area"] == 100.0
    assert r["contributing_contour_count"] == 1
    skipped = {(s["original_index"], s["reason"])
               for s in r["skipped_contours"]}
    assert skipped == {(1, "degenerate"), (2, "open")}


def test_empty_path_zero_coverage():
    r = _coverage("", [(0.0, 0.0, 10.0, 10.0)])
    w = r["windows"][0]
    assert w["covered_area"] == 0.0
    assert w["coverage"] == 0.0
    assert w["window_area"] == 100.0
    assert r["contributing_contour_count"] == 0


def test_only_open_contours_zero_coverage():
    r = _coverage("M0,0 L10,0 L10,10", [(0.0, 0.0, 10.0, 10.0)])
    assert r["windows"][0]["covered_area"] == 0.0
    assert r["windows"][0]["coverage"] == 0.0
    assert r["skipped_contours"][0]["reason"] == "open"


def test_window_outside_contour_zero():
    r = _coverage("M0,0h10v10h-10Z", [(20.0, 20.0, 30.0, 30.0)])
    w = r["windows"][0]
    assert w["covered_area"] == 0.0
    assert w["coverage"] == 0.0


def test_window_touching_edge_zero():
    # 窗口仅与外方在 x=10 处相切：交集为零面积
    r = _coverage("M0,0h10v10h-10Z", [(10.0, 0.0, 20.0, 10.0)])
    assert r["windows"][0]["covered_area"] == 0.0


def test_multiple_windows_in_input_order():
    d = "M0,0h10v10h-10Z M2,2h6v6h-6Z"
    r = _coverage(d, [(0.0, 0.0, 10.0, 10.0),
                      (5.0, 5.0, 15.0, 15.0),
                      (20.0, 20.0, 30.0, 30.0)])
    assert r["window_count"] == 3
    assert [w["index"] for w in r["windows"]] == [0, 1, 2]
    # 全窗口：100 - 36 = 64
    assert r["windows"][0]["covered_area"] == 64.0
    # 5..15 窗口：外环裁剪 25，孔裁剪 5..8 见方 9 -> 16
    assert r["windows"][1]["covered_area"] == 16.0
    assert r["windows"][1]["coverage"] == 0.16
    # 窗外：0
    assert r["windows"][2]["covered_area"] == 0.0


def test_curve_flattened_circle():
    # 单位圆细分多边形：窗口包住时面积近似 π 且不超过真圆
    d = "M-1,0 A1,1 0 1,0 1,0 A1,1 0 1,0 -1,0 Z"
    r = _coverage(d, [(-2.0, -2.0, 2.0, 2.0)])
    w = r["windows"][0]
    assert 3.0 < w["covered_area"] < math.pi
    assert math.isclose(w["coverage"], w["covered_area"] / 16.0, abs_tol=1e-9)


def test_circle_clipped_to_inner_window():
    # 窗口完全在圆内：覆盖率 1（裁剪后恰为窗口）
    d = "M-10,0 A10,10 0 1,0 10,0 A10,10 0 1,0 -10,0 Z"
    r = _coverage(d, [(0.0, 0.0, 5.0, 5.0)])
    assert r["windows"][0]["coverage"] == 1.0


def test_evenodd_same_result_as_nonzero():
    d = "M0,0h10v10h-10Z M2,2h6v6h-6Z M3,3h4v4h-4Z"
    windows = [(0.0, 0.0, 10.0, 10.0), (2.0, 0.0, 9.0, 10.0)]
    r_nz = _coverage(d, windows, "nonzero")
    r_eo = _coverage(d, windows, "evenodd")
    assert [w["covered_area"] for w in r_nz["windows"]] == \
           [w["covered_area"] for w in r_eo["windows"]]


def test_orientation_independent():
    # 外环逆时针书写，窗口覆盖面积不变
    d_ccw = "M0,0 L0,10 L10,10 L10,0 Z M2,2h6v6h-6Z"
    r = _coverage(d_ccw, [(0.0, 0.0, 10.0, 10.0)])
    assert r["windows"][0]["covered_area"] == 64.0


def test_duplicate_outer_clamped_to_window_area():
    # 两条全等外环（异常重叠）：交替和 200 被钳制到窗口面积 100
    d = "M0,0h10v10h-10Z M0,0h10v10h-10Z"
    r = _coverage(d, [(0.0, 0.0, 10.0, 10.0)])
    w = r["windows"][0]
    assert w["covered_area"] == 100.0
    assert w["coverage"] == 1.0


def test_negative_coordinates_window():
    r = _coverage("M-10,-10h20v20h-20Z", [(-5.0, -5.0, 5.0, 5.0)])
    w = r["windows"][0]
    assert w["covered_area"] == 100.0
    assert w["coverage"] == 1.0


def test_no_negative_zero_in_output():
    d = "M-1,0 A1,1 0 1,0 1,0 A1,1 0 1,0 -1,0 Z"
    r = _coverage(d, [(-2.0, -2.0, 2.0, 2.0), (5.0, 5.0, 9.0, 9.0)])
    assert "-0.0" not in repr(r)
