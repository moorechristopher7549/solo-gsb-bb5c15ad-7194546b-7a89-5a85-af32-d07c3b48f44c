"""矩形窗口覆盖分析（window-coverage，只读即时分析）。

对一份已存 SVG path 即时解析并完成轮廓包含层级分类后（与
metrics / clearance / scanlines 等同一份解析与曲线细分折线，见
:func:`app.svggeo.analyze_contours` 与
:func:`app.svggeo.flatten_subpath`），把每个**有效闭合环**
（``outer`` / ``hole``，闭合且非退化）用 Sutherland–Hodgman 算法
逐半平面裁剪到请求的轴对齐矩形窗口内，再对裁剪结果按鞋带公式求面积，
按**包含层级交替**累加：

- 外环 / 岛（深度偶数）与窗口的交集面积 **+** 计入；
- 孔洞（深度奇数）与窗口的交集面积 **−** 扣减，更深层依次交替；
- 裁剪前先做包围盒剔除：轮廓包围盒与窗口不相交时直接跳过（面积为 0），
  因此**穿越窗口但所有顶点均在窗外**的轮廓（边与窗口相交）仍会被裁剪
  并计入，不会漏算；
- **开放轮廓**（未以 ``Z`` 闭合）与**退化轮廓**（面积为 0 或顶点不足）
  不参与，列入 ``skipped_contours``；
- 窗口内没有任何有效填充（窗口完全在图形外、空路径、仅开放/退化轮廓）
  时净填充面积与覆盖率均为 ``0``。

裁剪输出可能带沿窗口边界的桥接边（凹多边形裁剪的标准现象），
鞋带求和时桥接边成对反向、自动抵消，净面积即填充区域与窗口的交集面积。

整个计算只读、即时完成：不依赖已保存的规范化结果，也不写库。
"""

from __future__ import annotations

import math
from typing import Any

from .svggeo import Contour, dedupe_ring, signed_area

# 单次请求允许的最多窗口数，超限由路由层（Pydantic）返回 422
MAX_WINDOWS = 50
# 面积零值阈值的相对系数（与 svggeo 退化判定 1e-10*diag² 同量级）
_AREA_EPS = 1e-10
# 输出量化位数（仅影响 JSON 展示，不改变判定）
_DECIMALS = 9

# (裁剪轴, 边界, 是否保留“大于等于”一侧)：左 x>=xmin、右 x<=xmax、
# 下 y>=ymin、上 y<=ymax（SVG 屏幕坐标，y 轴向下，不影响面积）
_HALFPLANES = ((0, "xmin", True), (0, "xmax", False),
               (1, "ymin", True), (1, "ymax", False))


def compute_window_coverage(
        contours: list[Contour],
        windows: list[tuple[float, float, float, float]]) -> dict[str, Any]:
    """对已分类的轮廓列表逐窗口计算净填充面积与覆盖率。

    参数:
        contours: :func:`app.svggeo.analyze_contours` 返回的轮廓
            （已带 ``role`` / ``depth``）。
        windows: ``(xmin, ymin, xmax, ymax)`` 窗口列表，坐标均须有限、
            宽高为正（由路由层模型保证），最多 :data:`MAX_WINDOWS` 个。

    返回:
        ``{window_count, windows, skipped_contours}``；``windows`` 按
        输入顺序给出每项 ``{index, window, window_area, filled_area,
        coverage, contributing_contour_count}``。
    """
    valid = [c for c in contours if c.role in ("outer", "hole")]
    valid.sort(key=lambda c: c.original_index)

    scale = 1.0
    for c in valid:
        diag = math.hypot(c.bounds[2] - c.bounds[0],
                          c.bounds[3] - c.bounds[1])
        scale = max(scale, diag)

    skipped = [
        {"original_index": c.original_index, "role": c.role,
         "reason": ("open" if c.role == "open" else "degenerate")}
        for c in sorted(contours, key=lambda c: c.original_index)
        if c.role not in ("outer", "hole")
    ]

    items: list[dict[str, Any]] = []
    for idx, (xmin, ymin, xmax, ymax) in enumerate(windows):
        win_w, win_h = xmax - xmin, ymax - ymin
        win_area = win_w * win_h
        win_diag = math.hypot(win_w, win_h)
        eps = _AREA_EPS * max(1.0, scale, win_diag) ** 2

        net_area = 0.0
        contributing = 0
        for c in valid:
            # 包围盒剔除：不相交（含仅隔一条切线）直接跳过；
            # bbox 与窗口相交即尝试裁剪，保证“顶点全在窗外但穿越窗口”
            # 的轮廓被计入
            if (c.bounds[2] < xmin or c.bounds[0] > xmax
                    or c.bounds[3] < ymin or c.bounds[1] > ymax):
                continue
            clipped = _clip_ring_to_window(c.points,
                                           (xmin, ymin, xmax, ymax))
            piece = abs(signed_area(clipped))
            if piece <= eps:
                continue  # 仅相切于一点/一线，窗口内无面积
            contributing += 1
            net_area += piece if c.depth % 2 == 0 else -piece

        # 几何上嵌套环的交替代数和必落在 [0, 窗口面积]；
        # 钳制仅用于吸收细分/裁剪浮点误差
        if net_area < 0.0:
            net_area = 0.0
        elif net_area > win_area:
            net_area = win_area

        items.append({
            "index": idx,
            "window": {"xmin": _num(xmin), "ymin": _num(ymin),
                       "xmax": _num(xmax), "ymax": _num(ymax)},
            "window_area": _num(win_area),
            "filled_area": _num(net_area),
            "coverage": _num(net_area / win_area),
            "contributing_contour_count": contributing,
        })

    return {
        "window_count": len(windows),
        "windows": items,
        "skipped_contours": skipped,
    }


def _clip_ring_to_window(
        points: list[tuple[float, float]],
        window: tuple[float, float, float, float]
        ) -> list[tuple[float, float]]:
    """把闭合环折线依次裁剪到矩形窗口的四个半平面内（Sutherland–Hodgman）。

    返回裁剪后位于窗口内（含边界）的多边形顶点；无交集时为空/退化列表。
    顶点环绕方向与输入一致，凹多边形裁剪产生的桥接边在鞋带求面积时
    成对抵消。
    """
    xmin, ymin, xmax, ymax = window
    bounds = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    subject = list(points)
    for axis, key, greater in _HALFPLANES:
        subject = _clip_halfplane(subject, axis, bounds[key], greater)
        if not subject:
            return []
    return dedupe_ring(subject)


def _clip_halfplane(subject: list[tuple[float, float]],
                    axis: int,
                    bound: float,
                    greater: bool) -> list[tuple[float, float]]:
    """Sutherland–Hodgman 单步裁剪。

    ``greater=True`` 保留 ``坐标 >= bound`` 一侧（左/下边界），
    否则保留 ``坐标 <= bound`` 一侧（右/上边界）；恰在边界上的点视为在内。
    遍历每条有向边 ``S -> E``，按两端内/外组合决定输出交点与终点。
    """
    if not subject:
        return []
    out: list[tuple[float, float]] = []
    s = subject[-1]
    sv = s[axis]
    s_in = sv >= bound if greater else sv <= bound
    for e in subject:
        ev = e[axis]
        e_in = ev >= bound if greater else ev <= bound
        if e_in:
            if not s_in and ev != sv:
                t = (bound - sv) / (ev - sv)
                out.append((s[0] + t * (e[0] - s[0]),
                            s[1] + t * (e[1] - s[1])))
            out.append(e)
        elif s_in:
            t = (bound - sv) / (ev - sv)
            out.append((s[0] + t * (e[0] - s[0]),
                        s[1] + t * (e[1] - s[1])))
        s, sv, s_in = e, ev, e_in
    return out


def _num(v: float) -> float:
    """量化展示值并消除 -0.0。"""
    r = round(v, _DECIMALS)
    return r + 0.0 if r == 0 else r
