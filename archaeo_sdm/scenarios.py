"""
기후 시나리오 투영과 분산(이동) 시뮬레이션.

1) 투영(hindcast/forecast)
   한 시점의 기후로 학습한 모델을 다른 시점(중기 홀로세·LGM·현생)의 기후에 적용해
   '그 동물이 살 수 있었던 범위'가 어떻게 달라지는지 봅니다.

2) 범위 변화 지표
   적합/부적합 이진화 후 면적 증감, 유지 범위, 중심점 이동 거리·방향.

3) 분산 시뮬레이션 (MigClim·KISSMig 계열을 단순화한 셀룰러 오토마타)
   '적합한 곳'과 '실제로 도달한 곳'은 다릅니다. 동물이 한 세대에 이동할 수 있는
   거리가 제한되어 있으면, 기후가 좋아져도 곧바로 채워지지 않습니다.
   시점 사이의 적합도를 선형 보간하며 단계별로 확산·소멸을 계산합니다.

주의: 외삽
   학습 시점에 없던 기후 조합(예: LGM의 혹한)에서는 모델이 근거 없이 값을 냅니다.
   mess() 로 외삽 지역을 표시해 해석에서 제외하십시오 (Elith et al. 2010).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ 투영
def project(model, stacks: dict, chunk: int = 200_000) -> dict:
    """{시점이름: RasterStack} 각각에 모델을 적용한 적합도면."""
    out = {}
    for name, st in stacks.items():
        H, W = st.shape
        surf = np.full((H, W), np.nan, dtype="float32")
        rr, cc = np.nonzero(st.valid_mask)
        for s in range(0, len(rr), chunk):
            r, c = rr[s:s + chunk], cc[s:s + chunk]
            surf[r, c] = model.predict(st.data[:, r, c].T).astype("float32")
        out[name] = surf
    return out


def mess(X_ref: np.ndarray, stack) -> np.ndarray:
    """MESS(다변량 환경 유사도, Elith et al. 2010). 음수면 학습 범위 밖(외삽)."""
    ref = np.asarray(X_ref, float)
    H, W = stack.shape
    out = np.full((H, W), np.nan, dtype="float32")
    rr, cc = np.nonzero(stack.valid_mask)
    X = stack.data[:, rr, cc].T
    mins, maxs = ref.min(0), ref.max(0)
    rng = np.where(maxs - mins == 0, 1, maxs - mins)
    sims = np.empty_like(X)
    for j in range(X.shape[1]):
        col = np.sort(ref[:, j])
        f = np.searchsorted(col, X[:, j], side="right") / len(col) * 100
        s = np.where(f == 0, (X[:, j] - mins[j]) / rng[j] * 100,
            np.where(f <= 50, 2 * f,
            np.where(f < 100, 2 * (100 - f), (maxs[j] - X[:, j]) / rng[j] * 100)))
        sims[:, j] = s
    out[rr, cc] = sims.min(axis=1)
    return out


# ------------------------------------------------------------ 이진화·변화
def threshold_p10(model, Xp) -> float:
    """학습 출토지의 90%가 '적합'으로 남는 임계값(P10). 고고학 SDM에서 흔히 씀."""
    return float(np.quantile(model.predict(Xp), 0.10))


def cell_area_km2(stack) -> np.ndarray:
    """격자 한 칸의 면적(km²). 경위도 격자면 위도에 따라 달라집니다."""
    import rasterio

    H, W = stack.shape
    t = stack.transform
    if stack.crs and stack.crs.is_geographic:
        _, lats = rasterio.transform.xy(t, np.arange(H), np.zeros(H, int))
        lats = np.asarray(lats)
        a = (abs(t.a) * 111.32 * np.cos(np.radians(lats))) * (abs(t.e) * 110.57)
        return np.repeat(a[:, None], W, axis=1)
    return np.full((H, W), abs(t.a * t.e) / 1e6)


def _centroid(mask, stack):
    import rasterio

    rr, cc = np.nonzero(mask)
    if len(rr) == 0:
        return np.nan, np.nan
    xs, ys = rasterio.transform.xy(stack.transform, rr, cc)
    return float(np.mean(xs)), float(np.mean(ys))


def range_change(bin_from, bin_to, stack, label_from="이전", label_to="이후") -> dict:
    """두 이진 분포 사이의 면적 증감과 중심 이동."""
    A = cell_area_km2(stack)
    a, b = bin_from.astype(bool), bin_to.astype(bool)
    x0, y0 = _centroid(a, stack)
    x1, y1 = _centroid(b, stack)
    dx = (x1 - x0) * 111.32 * np.cos(np.radians((y0 + y1) / 2))
    dy = (y1 - y0) * 110.57
    bearing = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    dirs = ["북", "북동", "동", "남동", "남", "남서", "서", "북서"]
    return {
        "비교": f"{label_from} → {label_to}",
        "이전_면적km2": round(float(A[a].sum())),
        "이후_면적km2": round(float(A[b].sum())),
        "유지km2": round(float(A[a & b].sum())),
        "상실km2": round(float(A[a & ~b].sum())),
        "확장km2": round(float(A[~a & b].sum())),
        "면적변화(%)": round(100 * (A[b].sum() - A[a].sum()) / max(A[a].sum(), 1), 1),
        "중심이동km": round(float(np.hypot(dx, dy)), 1) if np.isfinite(dx) else np.nan,
        "이동방향": dirs[int(((bearing + 22.5) % 360) // 45)] if np.isfinite(bearing) else "-",
    }


# ------------------------------------------------------------ 분산 시뮬레이션
def dispersal_simulation(suit_series: list, threshold: float, start_mask: np.ndarray,
                         steps_per_interval: int = 10, max_dispersal_cells: int = 1,
                         p_colonize: float = 0.6, p_persist_unsuitable: float = 0.2,
                         random_state: int = 0) -> dict:
    """적합도면 시계열을 따라 점유 범위가 어떻게 퍼지고 줄어드는지 계산합니다.

    suit_series        : [시점1 면, 시점2 면, ...]  (같은 격자)
    threshold          : 이 값 이상이면 '살 수 있는 곳'
    start_mask         : 시작 시점의 점유 셀 (예: 출토지가 있는 셀)
    steps_per_interval : 두 시점 사이를 몇 단계로 나눌지 (한 단계 = 한 세대 묶음)
    max_dispersal_cells: 한 단계에 이동할 수 있는 최대 격자 수
    p_colonize         : 적합한 이웃 셀로 퍼질 확률 (적합도에 비례해 조정)
    p_persist_unsuitable: 부적합해진 셀에서 한 단계 더 버틸 확률
    """
    from scipy.ndimage import binary_dilation

    rng = np.random.default_rng(random_state)
    occ = start_mask.astype(bool) & np.isfinite(suit_series[0])
    r = max_dispersal_cells
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kernel = (xx ** 2 + yy ** 2) <= r ** 2

    frames, areas, suitable_areas = [occ.copy()], [int(occ.sum())], []
    for a, b in zip(suit_series[:-1], suit_series[1:]):
        for s in range(1, steps_per_interval + 1):
            w = s / steps_per_interval
            suit = np.where(np.isfinite(a) & np.isfinite(b), (1 - w) * a + w * b, np.nan)
            good = np.nan_to_num(suit, nan=0) >= threshold
            reach = binary_dilation(occ, structure=kernel) & ~occ
            p = p_colonize * np.clip(np.nan_to_num(suit, nan=0) / max(threshold, 1e-9), 0, 1)
            new = reach & good & (rng.random(occ.shape) < p)
            survive = occ & (good | (rng.random(occ.shape) < p_persist_unsuitable))
            occ = survive | new
            frames.append(occ.copy())
            areas.append(int(occ.sum()))
            suitable_areas.append(int(good.sum()))
    return {"frames": frames, "occupied_cells": areas, "suitable_cells": suitable_areas}


def summarize_series(names, surfaces, threshold, stack) -> pd.DataFrame:
    """시점별 적합 면적과 연속 시점 간 변화표."""
    bins = {n: np.nan_to_num(s, nan=0) >= threshold for n, s in zip(names, surfaces)}
    rows = [range_change(bins[a], bins[b], stack, a, b) for a, b in zip(names[:-1], names[1:])]
    return pd.DataFrame(rows)
