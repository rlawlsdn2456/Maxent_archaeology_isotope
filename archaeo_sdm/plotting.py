"""결과 시각화 (적합도 지도, 반응 곡선, 변수 중요도)."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 파일로 저장
import matplotlib.pyplot as plt
import numpy as np

# 한글 라벨이 깨지지 않도록 (Windows 기본 폰트)
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_suitability(surface, stack, out_png, points_xy=None, title="서식지 적합도 (cloglog)"):
    import rasterio.plot

    fig, ax = plt.subplots(figsize=(8, 7))
    extent = rasterio.plot.plotting_extent(surface, stack.transform)
    im = ax.imshow(surface, extent=extent, cmap="viridis", vmin=0, vmax=1)
    if points_xy is not None:
        ax.scatter(points_xy[0], points_xy[1], s=18, c="red", edgecolors="white",
                   linewidths=0.5, label="출토지점")
        ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    fig.colorbar(im, ax=ax, shrink=0.8, label="적합도")
    return _save(fig, out_png)


def plot_response_curves(model, X_reference, out_png, ncols: int = 3):
    names = model.var_names_
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        g, y = model.response_curve(name, X_reference)
        ax.plot(g, y, lw=2)
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlabel("변수 값"); ax.set_ylabel("적합도")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("반응 곡선 (다른 변수는 평균 고정)")
    fig.tight_layout()
    return _save(fig, out_png)


def plot_importance(importance: dict, out_png, title="변수 중요도 (%)"):
    names = list(importance.keys())[::-1]
    vals = [importance[n] for n in names]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 2))
    ax.barh(names, vals, color="#3b7dd8")
    ax.set_xlabel("%"); ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    return _save(fig, out_png)


def plot_isotope_biplot(d13c, d15n, out_png, labels=None, title="탄소-질소 동위원소 분포"):
    fig, ax = plt.subplots(figsize=(6, 5))
    if labels is None:
        ax.scatter(d13c, d15n, s=35, c="#3b7dd8", edgecolors="k", linewidths=0.3)
    else:
        for g in np.unique(labels):
            m = labels == g
            ax.scatter(np.asarray(d13c)[m], np.asarray(d15n)[m], s=35,
                       edgecolors="k", linewidths=0.3, label=f"집단 {g}")
        ax.legend()
    ax.set_xlabel("δ13C (‰)"); ax.set_ylabel("δ15N (‰)"); ax.set_title(title)
    ax.grid(alpha=0.3)
    return _save(fig, out_png)


# ===================== 시기별 비교용 다중 패널 그림 =====================

def plot_panel_maps(surfaces, stack, out_png, titles=None, suptitle="",
                    cmap="viridis", vmin=None, vmax=None, cbar_label="",
                    points=None, ncols=3, diverging=False, basemap=None):
    """여러 시기(또는 분류군)의 면을 한 장에 나란히 그립니다.

    surfaces : {이름: 2D배열}
    points   : {이름: (x배열, y배열)}  해당 패널에 찍을 유적 위치
    basemap  : 뒤에 옅게 깔 2D 배열(예: 고도나 기온) - 육지 윤곽을 잡아 줍니다
    """
    import rasterio.plot

    keys = list(surfaces.keys())
    n = len(keys)
    if n == 0:
        return None
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    stacked = np.concatenate([np.asarray(s)[np.isfinite(s)].ravel() for s in surfaces.values()]) \
        if any(np.isfinite(np.asarray(s)).any() for s in surfaces.values()) else np.array([0.0])
    if vmin is None or vmax is None:
        lo, hi = np.nanpercentile(stacked, [2, 98])
        if diverging:
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    extent = rasterio.plot.plotting_extent(np.zeros(stack.shape), stack.transform)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.9 * nrows), squeeze=False)
    im = None
    for ax, key in zip(axes.ravel(), keys):
        if basemap is not None:
            ax.imshow(basemap, extent=extent, cmap="Greys", alpha=0.30)
        im = ax.imshow(surfaces[key], extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        if points and key in points:
            px, py = points[key]
            ax.scatter(px, py, s=14, c="crimson", edgecolors="white", linewidths=0.4, zorder=3)
        ax.set_title(titles.get(key, key) if isinstance(titles, dict) else key, fontsize=11)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, ls=":", lw=0.5)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.75, label=cbar_label)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    return _save(fig, out_png)


def plot_isotope_timeseries(summary: "pd.DataFrame", out_png, period_order,
                            value_col="d13C_평균", err_col="d13C_표준편차",
                            group_col="taxon_group", ylabel="δ13C (‰)",
                            title="시기별 동위원소 변화"):
    """시기(x축) x 분류군(선)별 평균±표준편차 변화도."""
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = {k: i for i, k in enumerate(period_order)}
    for grp, sub in summary.groupby(group_col):
        sub = sub[sub["period"].isin(xs)].sort_values("period", key=lambda s: s.map(xs))
        if sub[value_col].notna().sum() == 0:
            continue
        x = sub["period"].map(xs)
        ax.errorbar(x, sub[value_col], yerr=sub.get(err_col),
                    marker="o", capsize=3, lw=1.8, label=f"{grp} (n={int(sub['n'].sum())})")
    ax.set_xticks(range(len(period_order)))
    ax.set_xticklabels(period_order)
    ax.set_xlabel("고고학적 시기"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    return _save(fig, out_png)


def plot_biplot_by_period(df, out_png, period_order, x="d13C_coll", y="d15N_coll",
                          hue="period", title="시기별 δ13C–δ15N 분포"):
    """식이 해석용 이변량 산점도 + C3/C4 기준선."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(period_order)))
    for c, key in zip(colors, period_order):
        sub = df[df[hue] == key]
        if len(sub) == 0:
            continue
        ax.scatter(sub[x], sub[y], s=26, color=c, edgecolors="k", linewidths=0.25,
                   label=f"{key} (n={len(sub)})", alpha=0.85)
    ax.axvline(-18, ls="--", c="gray", lw=1)
    ax.axvline(-12, ls="--", c="gray", lw=1)
    ax.text(-18, ax.get_ylim()[1], " C3", va="top", fontsize=9, color="gray")
    ax.text(-12, ax.get_ylim()[1], " C4", va="top", fontsize=9, color="gray")
    ax.set_xlabel("δ13C (‰, VPDB)"); ax.set_ylabel("δ15N (‰, AIR)")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, out_png)


def plot_zone_trajectories(table, out_png, period_order, value_col="d13C_coll_평균",
                           err_col="d13C_coll_표준편차", n_col="d13C_coll_n",
                           ylabel="δ13C (‰)", title="소지역별 시기 변화",
                           min_n: int = 3):
    """소지역마다 한 선 - 같은 지역 안에서 시기가 어떻게 변하는지 봅니다.

    지역이 다르면 애초에 환경이 다르므로, 시기 비교는 '같은 선 위에서만' 유효합니다.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    xs = {k: i for i, k in enumerate(period_order)}
    zones = [z for z in table["zone"].dropna().unique()]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(zones), 1)))
    for c, zone in zip(colors, zones):
        sub = table[(table["zone"] == zone) & (table.get(n_col, 0) >= min_n)]
        sub = sub[sub["period"].isin(xs)].sort_values("period", key=lambda s: s.map(xs))
        if sub[value_col].notna().sum() == 0:
            continue
        x = sub["period"].map(xs)
        ls = "-" if len(sub) > 1 else "None"
        ax.errorbar(x, sub[value_col], yerr=sub.get(err_col), color=c, marker="o",
                    ls=ls, capsize=3, lw=2, label=f"{zone} ({int(sub['n'].sum())}점)")
    ax.set_xticks(range(len(period_order)))
    ax.set_xticklabels(period_order)
    ax.set_xlabel("고고학적 시기"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    return _save(fig, out_png)


def plot_site_map(df, out_png, stack=None, color_by="zone", title="유적 분포"):
    """자료가 있는 유적의 위치를 소지역별 색으로 표시."""
    fig, ax = plt.subplots(figsize=(10, 7))
    if stack is not None:
        import rasterio.plot
        ext = rasterio.plot.plotting_extent(np.zeros(stack.shape), stack.transform)
        ax.imshow(stack.data[0], extent=ext, cmap="Greys", alpha=0.3)
    sites = df.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["site"])
    groups = sites[color_by].fillna("미분류").unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
    for c, g in zip(colors, groups):
        s = sites[sites[color_by].fillna("미분류") == g]
        ax.scatter(s["lon"], s["lat"], s=40, color=c, edgecolors="k", linewidths=0.4,
                   label=f"{g} ({len(s)})")
    ax.set_xlabel("경도"); ax.set_ylabel("위도"); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, out_png)


def plot_zone_taxon_biplot(df, out_png, zone_col="zone", taxon_col="taxon_group",
                           taxa=("돼지", "개", "소", "말", "사슴", "양·염소"),
                           title="지역·분류군별 δ13C–δ15N 분포"):
    """지역은 색, 분류군은 기호로 구분한 이변량 산점도.

    같은 지역 안에서 분류군이 어떻게 갈라지는지, 지역 간에는 어떻게 다른지를
    한 장에서 봅니다. 세로 점선은 C3/C4 식이 경계입니다.
    """
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    zones = [z for z in df[zone_col].dropna().unique()]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(zones), 1)))
    fig, ax = plt.subplots(figsize=(9.5, 7))
    for c, zone in zip(colors, zones):
        for m, taxon in zip(markers, taxa):
            s = df[(df[zone_col] == zone) & (df[taxon_col] == taxon)]
            if s.empty:
                continue
            ax.scatter(s["d13C_coll"], s["d15N_coll"], s=48, color=c, marker=m,
                       edgecolors="k", linewidths=0.35, alpha=0.85,
                       label=f"{zone} · {taxon} ({len(s)})")
    for v, lab in ((-18, " C3 우세"), (-12, " C4 우세")):
        ax.axvline(v, ls="--", c="gray", lw=1)
        ax.text(v, ax.get_ylim()[1], lab, va="top", fontsize=9, color="gray")
    ax.set_xlabel("δ13C (‰, VPDB)")
    ax.set_ylabel("δ15N (‰, AIR)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    return _save(fig, out_png)
