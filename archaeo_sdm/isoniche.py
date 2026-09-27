"""
동위원소 니치(isotopic niche) 분석 — SIBER 방식 (Jackson et al. 2011).

δ13C–δ15N 평면에서 한 집단(예: '요동 돼지 한대')이 차지하는 영역을 타원으로 요약하고,
집단 간 크기와 겹침을 비교합니다. MaxEnt와 달리 **개체 단위**로 계산하므로
유적이 몇 곳뿐이어도 개체가 10점 이상이면 쓸 수 있습니다.

지표
  SEA   표준타원 면적 (‰²). 자료의 약 40%를 담는 타원
  SEAc  소표본 보정 SEA = SEA·(n-1)/(n-2). 집단 간 비교에는 SEAc를 씁니다
  NR    δ15N 범위 — 영양단계 폭
  CR    δ13C 범위 — 식물 자원(C3/C4) 폭
  TA    볼록껍질 면적 — 전체 니치 공간 (표본 수에 민감)
  겹침  두 표준타원의 교집합 면적과 각 타원 대비 비율

주의
  δ15N 기저값(baseline)은 지역마다 다릅니다(건조도·토양·해양 영향). 서로 다른 지역의
  δ15N 차이를 곧바로 '영양단계 차이'로 읽으면 안 됩니다. 같은 지역의 초식동물(사슴 등)
  값을 기준선으로 함께 보십시오.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_N = 5          # SEAc 계산 최소 개체수 (10 이상 권장)


def standard_ellipse(x, y, n_points: int = 100):
    """표준타원 매개변수와 그리기용 좌표."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    S = np.cov(x, y)
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, 0, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    a, b = np.sqrt(vals)                               # 장축·단축 반지름(1 SD)
    t = np.linspace(0, 2 * np.pi, n_points)
    circ = np.vstack([a * np.cos(t), b * np.sin(t)])
    pts = vecs @ circ
    return {"cx": x.mean(), "cy": y.mean(), "a": a, "b": b, "vecs": vecs,
            "ex": pts[0] + x.mean(), "ey": pts[1] + y.mean()}


def sea(x, y) -> tuple[float, float]:
    n = len(x)
    e = standard_ellipse(x, y)
    s = float(np.pi * e["a"] * e["b"])
    return s, s * (n - 1) / (n - 2) if n > 2 else np.nan


def layman(x, y) -> dict:
    from scipy.spatial import ConvexHull

    x, y = np.asarray(x, float), np.asarray(y, float)
    try:
        ta = float(ConvexHull(np.column_stack([x, y])).volume) if len(x) >= 3 else np.nan
    except Exception:                                  # 점들이 한 직선 위에 있을 때
        ta = 0.0
    return {"NR(δ15N범위)": round(float(np.ptp(y)), 2), "CR(δ13C범위)": round(float(np.ptp(x)), 2),
            "TA(볼록껍질)": round(ta, 2)}


def sea_bootstrap(x, y, n_boot: int = 1000, random_state: int = 0):
    """SEAc의 부트스트랩 95% 신뢰구간."""
    rng = np.random.default_rng(random_state)
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        vals.append(sea(x[i], y[i])[1])
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _inside(e, X, Y):
    dx, dy = X - e["cx"], Y - e["cy"]
    u = e["vecs"][0, 0] * dx + e["vecs"][1, 0] * dy
    v = e["vecs"][0, 1] * dx + e["vecs"][1, 1] * dy
    return (u / max(e["a"], 1e-9)) ** 2 + (v / max(e["b"], 1e-9)) ** 2 <= 1


def ellipse_overlap(g1, g2, res: int = 400) -> dict:
    """두 표준타원의 겹침 (격자 적분)."""
    e1 = standard_ellipse(*g1)
    e2 = standard_ellipse(*g2)
    xs = np.concatenate([e1["ex"], e2["ex"]])
    ys = np.concatenate([e1["ey"], e2["ey"]])
    X, Y = np.meshgrid(np.linspace(xs.min(), xs.max(), res), np.linspace(ys.min(), ys.max(), res))
    cell = (xs.max() - xs.min()) / (res - 1) * (ys.max() - ys.min()) / (res - 1)
    i1, i2 = _inside(e1, X, Y), _inside(e2, X, Y)
    a1, a2, inter = i1.sum() * cell, i2.sum() * cell, (i1 & i2).sum() * cell
    return {"겹침면적": round(float(inter), 2),
            "겹침/A(%)": round(100 * inter / a1, 1) if a1 else np.nan,
            "겹침/B(%)": round(100 * inter / a2, 1) if a2 else np.nan,
            "겹침/합집합(%)": round(100 * inter / (a1 + a2 - inter), 1) if (a1 + a2 - inter) else np.nan}


def niche_table(df: pd.DataFrame, group_cols, x="d13C_coll", y="d15N_coll",
                min_n: int = MIN_N, n_boot: int = 1000) -> pd.DataFrame:
    """집단별 SEAc(신뢰구간)·Layman 지표 표."""
    rows = []
    for key, sub in df.dropna(subset=[x, y]).groupby(list(group_cols)):
        key = key if isinstance(key, tuple) else (key,)
        n = len(sub)
        row = dict(zip(group_cols, key))
        row.update({"n": n, "유적수": sub["site"].nunique() if "site" in sub else np.nan,
                    "δ13C평균": round(sub[x].mean(), 2), "δ15N평균": round(sub[y].mean(), 2)})
        if n >= min_n and np.ptp(sub[x]) > 0 and np.ptp(sub[y]) > 0:
            s, sc = sea(sub[x], sub[y])
            lo, hi = sea_bootstrap(sub[x], sub[y], n_boot)
            row.update({"SEA": round(s, 2), "SEAc": round(sc, 2),
                        "SEAc_95%하한": round(lo, 2), "SEAc_95%상한": round(hi, 2)})
            row.update(layman(sub[x], sub[y]))
            row["비고"] = "" if n >= 10 else "n<10: 해석 주의"
        else:
            row["비고"] = f"n<{min_n} 또는 값이 한 점에 몰림: 계산 생략"
        rows.append(row)
    return pd.DataFrame(rows)


def overlap_matrix(df, group_cols, x="d13C_coll", y="d15N_coll", min_n: int = MIN_N):
    """집단 쌍별 겹침/합집합(%) 행렬."""
    groups = {}
    for key, sub in df.dropna(subset=[x, y]).groupby(list(group_cols)):
        if len(sub) >= min_n and np.ptp(sub[x]) > 0 and np.ptp(sub[y]) > 0:
            name = " · ".join(map(str, key if isinstance(key, tuple) else (key,)))
            groups[name] = (sub[x].to_numpy(), sub[y].to_numpy())
    names = list(groups)
    M = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i:]:
            v = 100.0 if a == b else ellipse_overlap(groups[a], groups[b])["겹침/합집합(%)"]
            M.loc[a, b] = M.loc[b, a] = v
    return M, groups


def plot_niches(groups: dict, out_png, title="동위원소 니치 (표준타원 SEAc)"):
    from .plotting import _save, plt

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
    for c, (name, (x, y)) in zip(colors, groups.items()):
        e = standard_ellipse(x, y)
        ax.scatter(x, y, s=20, color=c, alpha=0.55, edgecolors="none")
        ax.plot(e["ex"], e["ey"], color=c, lw=2.2, label=f"{name} (n={len(x)})")
        ax.plot(e["cx"], e["cy"], marker="+", color=c, ms=12, mew=2)
    for v, lab in ((-18, " C3 우세"), (-12, " C4 우세")):
        ax.axvline(v, ls="--", c="gray", lw=1)
        ax.text(v, ax.get_ylim()[1], lab, va="top", fontsize=9, color="gray")
    ax.set_xlabel("δ13C (‰, VPDB)")
    ax.set_ylabel("δ15N (‰, AIR)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return _save(fig, out_png)
