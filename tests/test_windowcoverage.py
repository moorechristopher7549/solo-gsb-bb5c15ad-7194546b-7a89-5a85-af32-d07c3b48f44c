"""windowcoverage：矩形窗口覆盖分析单元测试。"""

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.svggeo import analyze_contours  # noqa: E402
from app.windowcoverage import (  # noqa: E402
    MAX_WINDOWS,
    _clip_ring_to_window,
    compute_window_coverage,
)


def _coverage(d, windows, fill_rule="nonzero"):
    contours, _warnings = analyze_contours(d, fill_rule)
    return compute_window_coverage(contours, windows)


def test_window_fully_inside_outer():
    # 窗口 2..8 完全落在外方 0..10 内：净填充 = 窗口面积
    r = _coverage("M0,0h10v10h-10Z", [(2.0, 2.0, 8.0, 8.0)])
    assert r["window_count"] == 1
    w = r["windows"][0]
    assert w["index"] == 0
    assert w["window"] == {"xmin": 2.0, "ymin": 2.0,
                           "xmax": 8.0, "ymax": 8.0}
    assert w["window_area"] == 36.0
    assert w["filled_area"] == 36.0
    assert w["coverage"] == 1.0
    assert w["contributing_contour_count"] == 1
    assert r["skipped_contours"] == []


def test_window_partial_overlap():
    # 窗口 5..15 与外方 0..10 交集 5..10：面积 25
    r = _coverage("M0,0h10v10h-10Z", [(5.0, 5.0, 15.0, 15.0)])
    w = r["windows"][0]
    assert w["window_area"] == 100.0
    assert w["filled_area"] == 25.0
    assert w["coverage"] == 0.25


def test_window_outside_zero():
    r = _coverage("M0,0h10v10h-10Z", [(20.0, 20.0, 30.0, 30.0)])
    w = r["windows"][0]
    assert w["filled_area"] == 0.0
    assert w["coverage"] == 0.0
    assert w["contributing_contour_count"] == 0


def test_window_covers_contour_entirely():
    # 窗口大于轮廓：净填充 = 轮廓面积
    r = _coverage("M0,0h10v10h-10Z", [(-5.0, -5.0, 20.0, 20.0)])
    w = r["windows"][0]
    assert w["filled_area"] == 100.0
    assert w["coverage"] == 100.0 / 625.0


def test_window_equal_to_contour_bounds():
    r = _coverage("M0,0h10v10h-10Z", [(0.0, 0.0, 10.0, 10.0)])
    w = r["windows"][0]
    assert w["filled_area"] == 100.0
    assert w["coverage"] == 1.0


def test_hole_subtracted_in_window():
    # 外方 0..10 含孔 2..8，窗口覆盖整个图形：100 - 36 = 64
    r = _coverage("M0,0h10v10h-10Z M2,2h6v6h-6Z",
                  [(0.0, 0.0, 10.0, 10.0)])
    w = r["windows"][0]
    assert w["filled_area"] == 64.0
    assert w["coverage"] == 0.64
    assert w["contributing_contour_count"] == 2


def test_hole_partially_clipped_by_window():
    # 窗口 5..10：外环交集 25，孔 2..8 交集 3x3=9 -> 16
    r = _coverage("M0,0h10v10h-10Z M2,2h6v6h-6Z",
                  [(5.0, 5.0, 10.0, 10.0)])
    w = r["windows"][0]
    assert w["filled_area"] == 16.0
    assert w["coverage"] == 0.64


def test_island_readded_in_window():
    # 外 100 - 孔 64 + 岛 4 = 40（窗口覆盖全部）
    d = "M0,0h10v10h-10Z M1,1h8v8h-8Z M6,6h2v2h-2Z"
    r = _coverage(d, [(0.0, 0.0, 10.0, 10.0)])
    w = r["windows"][0]
    assert w["filled_area"] == 40.0
    assert w["coverage"] == 0.4
    assert w["contributing_contour_count"] == 3


def test_four_level_alternation_in_window():
    # 外 100 - 孔 64 + 岛 16 - 岛中孔 4 = 48
    d = ("M0,0h10v10h-10Z M1,1h8v8h-8Z "
         "M3,3h4v4h-4Z M4,4h2v2h-2Z")
    r = _coverage(d, [(0.0, 0.0, 10.0, 10.0)])
    assert r["windows"][0]["filled_area"] == 48.0


def test_crossing_contour_with_all_vertices_outside():
    # 菱形顶点均在窗外、边穿越窗口边界：交集 = 窗口 4 - 四角各 0.125 = 3.5
    r = _coverage("M-1.5,0 L0,-1.5 L1.5,0 L0,1.5 Z",
                  [(-1.0, -1.0, 1.0, 1.0)])
    w = r["windows"][0]
    assert w["filled_area"] == 3.5
    assert w["coverage"] == 0.875
    assert w["contributing_contour_count"] == 1


def test_contour_encloses_window_with_vertices_outside():
    # 大外环完全包住窗口（顶点全在窗外、边也不穿越窗口内部边界）
    r = _coverage("M-100,-100h200v200h-200Z", [(0.0, 0.0, 10.0, 10.0)])
    w = r["windows"][0]
    assert w["filled_area"] == 100.0
    assert w["coverage"] == 1.0


def test_open_and_degenerate_ignored():
    d = "M0,0h10v10h-10Z M5,5 L8,5 L5,5 Z M20,0 L30,0 L30,10"
    r = _coverage(d, [(0.0, 0.0, 10.0, 10.0)])
    w = r["windows"][0]
    assert w["filled_area"] == 100.0
    assert w["contributing_contour_count"] == 1
    skipped = {(s["original_index"], s["reason"])
               for s in r["skipped_contours"]}
    assert skipped == {(1, "degenerate"), (2, "open")}


def test_empty_path_zero_fill():
    r = _coverage("", [(0.0, 0.0, 10.0, 10.0)])
    w = r["windows"][0]
    assert w["filled_area"] == 0.0
    assert w["coverage"] == 0.0
    assert w["contributing_contour_count"] == 0
    assert r["skipped_contours"] == []


def test_only_open_path_zero_fill():
    r = _coverage("M0,0 L10,0 L10,10", [(0.0, 0.0, 10.0, 10.0)])
    w = r["windows"][0]
    assert w["filled_area"] == 0.0
    assert w["coverage"] == 0.0
    assert r["skipped_contours"][0]["reason"] == "open"


def test_concave_polygon_clipped_area():
    # L 形凹多边形：0..10 去掉右上 5x5 -> 75
    d = "M0,0 L10,0 L10,5 L5,5 L5,10 L0,10 Z"
    r = _coverage(d, [(0.0, 0.0, 10.0, 10.0),    # 整体 -> 75
                      (0.0, 0.0, 5.0, 10.0),      # 左臂整条 -> 50
                      (5.0, 5.0, 10.0, 10.0)])    # 缺口区域 -> 0
    assert r["windows"][0]["filled_area"] == 75.0
    assert r["windows"][1]["filled_area"] == 50.0
    assert r["windows"][2]["filled_area"] == 0.0


def test_curve_circle_window():
    # 单位圆（细分 32 边形）被窗口 [-2,2]^2 完整覆盖：
    # 净填充为细分多边形面积，介于 (3, π) 之间
    d = "M-1,0 A1,1 0 1,0 1,0 A1,1 0 1,0 -1,0 Z"
    r = _coverage(d, [(-2.0, -2.0, 2.0, 2.0)])
    w = r["windows"][0]
    assert 3.0 < w["filled_area"] < math.pi
    assert 0.19 < w["coverage"] < 0.2


def test_multiple_windows_independent_and_ordered():
    r = _coverage("M0,0h10v10h-10Z",
                  [(0.0, 0.0, 5.0, 5.0),
                   (5.0, 5.0, 10.0, 10.0),
                   (20.0, 20.0, 30.0, 30.0)])
    assert r["window_count"] == 3
    assert [w["index"] for w in r["windows"]] == [0, 1, 2]
    assert [w["filled_area"] for w in r["windows"]] == [25.0, 25.0, 0.0]
    assert [w["coverage"] for w in r["windows"]] == [1.0, 1.0, 0.0]


def test_window_touching_contour_edge_only_zero():
    # 窗口与轮廓仅共享一条边（x=10）：交集面积为 0
    r = _coverage("M0,0h10v10h-10Z", [(10.0, 0.0, 20.0, 10.0)])
    w = r["windows"][0]
    assert w["filled_area"] == 0.0
    assert w["coverage"] == 0.0
    assert w["contributing_contour_count"] == 0


def test_fill_rule_independent_and_deterministic():
    d = "M0,0h10v10h-10Z M2,2h6v6h-6Z"
    windows = [(1.0, 1.0, 9.0, 9.0)]
    nz = _coverage(d, windows, "nonzero")
    eo = _coverage(d, windows, "evenodd")
    assert nz == eo
    again = _coverage(d, windows, "nonzero")
    assert nz == again


def test_clip_helper_directly():
    # 直接对裁剪助手做白盒验证：三角形被窗口切出 (5,5),(10,5),(10,10)
    clipped = _clip_ring_to_window(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
        (5.0, 5.0, 10.0, 10.0))
    # 交集为直角三角形，两条直角边长度 5，面积 12.5
    from app.svggeo import signed_area
    assert abs(abs(signed_area(clipped)) - 12.5) < 1e-9


def test_max_windows_constant():
    assert MAX_WINDOWS == 50


def test_no_negative_zero_in_output():
    r = _coverage("M-1,0 A1,1 0 1,0 1,0 A1,1 0 1,0 -1,0 Z",
                  [(-2.0, -2.0, 2.0, 2.0)])
    assert "-0.0" not in repr(r)
