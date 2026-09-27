"""
아이소스케이프(isoscape) 생성 - 동위원소 값의 공간 분포면.

isotope.py가 '점 몇 개를 IDW로 펼치는' 최소 기능이었다면, 이 모듈은 고고학 자료에
실제로 쓸 수 있도록 다음을 추가합니다.

  - 유적 단위 평균화 : 같은 유적의 개체 수십 개가 한 지점을 지배하는 것(유사반복)을 방지
  - 지지도(support)  : 가장 가까운 시료까지의 거리(km)를 함께 만들어, 근거가 없는 곳은
                       아예 결측 처리하거나 반투명하게 표시
  - 변화면(delta)    : 시기 A -> 시기 B의 아이소스케이프 차이
  - 니치 중첩도       : 두 적합도면 사이의 Schoener's D
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial import cKDTree


def site_means(df: pd.DataFrame, value_col: str,
               site_col: str = "site", lat_col: str = "lat", lon_col: str = "lon"
               ) -> pd.DataFrame:
    """유적 단위 평균값 표. (유적, lon, lat, 값, 개체수)"""
    d = df[[site_col, lat_col, lon_col, value_col]].dropna(subset=[value_col, lat_col, lon_col])
    g = d.groupby([site_col, lat_col, lon_col], as_index=False).agg(
        value=(value_col, "mean"), n=(value_col, "size"), sd=(value_col, "std"))
    return g.rename(columns={site_col: "site", lat_col: "lat", lon_col: "lon"})


def _grid_xy(like):
    H, W = like.shape
    rows, cols = np.mgrid[0:H, 0:W]
    gx, gy = rasterio.transform.xy(like.transform, rows.ravel(), cols.ravel())
    return np.column_stack([np.asarray(gx), np.asarray(gy)]), H, W


def _deg_per_km(lat: float) -> tuple[float, float]:
    return 1.0 / (111.32 * max(np.cos(np.radians(lat)), 0.1)), 1.0 / 110.57


def build_isoscape(points: pd.DataFrame, like, power: float = 2.0, k: int = 6,
                   max_distance_km: float | None = 250.0):
    """IDW 아이소스케이프와 지지도(가장 가까운 시료까지의 km)를 만듭니다.

    points : site_means()의 결과 (lon, lat, value 필요)
    like   : RasterStack (격자 기준)
    반환   : (아이소스케이프 2D, 지지거리 2D km)
    """
    x = points["lon"].to_numpy(float)
    y = points["lat"].to_numpy(float)
    v = points["value"].to_numpy(float)
    if len(v) < 2:
        raise ValueError(f"아이소스케이프에는 유적이 2곳 이상 필요합니다 (현재 {len(v)}곳).")

    grid, H, W = _grid_xy(like)
    lat0 = float(np.mean(y))
    kx, ky = _deg_per_km(lat0)          # 1km에 해당하는 경도/위도 각도

    # 거리 계산을 위해 경위도를 대략적인 km 좌표로 환산(등거리 근사)
    P = np.column_stack([x / kx, y / ky])
    G = np.column_stack([grid[:, 0] / kx, grid[:, 1] / ky])

    tree = cKDTree(P)
    kk = min(k, len(v))
    dist, idx = tree.query(G, k=kk)
    if kk == 1:
        dist, idx = dist[:, None], idx[:, None]

    d = np.maximum(dist, 1e-6)
    w = 1.0 / d ** power
    surf = np.sum(w * v[idx], axis=1) / np.sum(w, axis=1)
    support = dist[:, 0]                                    # km

    if max_distance_km is not None:
        surf[support > max_distance_km] = np.nan
    return surf.reshape(H, W).astype("float32"), support.reshape(H, W).astype("float32")


def delta_isoscape(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """시기 A -> B 변화량 (B - A). 둘 중 하나라도 결측이면 결측."""
    a, b = np.asarray(before, float), np.asarray(after, float)
    out = b - a
    out[np.isnan(a) | np.isnan(b)] = np.nan
    return out.astype("float32")


def schoener_d(s1: np.ndarray, s2: np.ndarray) -> float:
    """두 적합도면의 니치 중첩도 (Schoener's D, 0=완전분리 1=동일)."""
    a, b = np.asarray(s1, float), np.asarray(s2, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return float("nan")
    pa, pb = a[m], b[m]
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return float(1.0 - 0.5 * np.abs(pa - pb).sum())


def diet_class(d13c_collagen) -> str:
    """콜라겐 δ13C로 C3/혼합/C4 식이를 구분 (발표문 기준: -18‰, -12‰)."""
    v = float(d13c_collagen)
    if np.isnan(v):
        return "미상"
    if v < -18:
        return "C3 우세"
    if v > -12:
        return "C4 우세"
    return "C3/C4 혼합"
