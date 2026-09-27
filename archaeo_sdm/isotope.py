"""
동위원소 값을 '환경 레이어'로 바꾸는 도구.

고고학 MaxEnt에서 탄소·질소 동위원소는 두 가지 방식으로 쓸 수 있습니다.

1) 예측변수(isoscape)로 사용
   출토 개체의 d13C, d15N 값을 공간 보간(IDW)해서 면(surface)으로 만든 뒤,
   기후·지형 레이어와 함께 MaxEnt에 넣습니다.
   -> "이런 먹이/식생 환경에서 이 동물이 살았다"는 생태권 정보를 모델에 직접 반영.

2) 집단 나누기(niche grouping)에 사용
   d13C/d15N 값으로 개체를 C3 삼림형 / C4·잡식형 등으로 나눠
   집단별로 따로 MaxEnt를 돌려 서식지 적합도를 비교합니다.
"""

from __future__ import annotations

import numpy as np
import rasterio
from scipy.spatial import cKDTree


def idw_surface(x, y, values, like, power: float = 2.0, k: int = 8,
                max_distance: float | None = None) -> np.ndarray:
    """역거리가중(IDW) 보간으로 동위원소 면을 만듭니다.

    x, y   : 시료 위치(래스터와 같은 CRS)
    values : 동위원소 값 (예: d13C)
    like   : RasterStack (격자 기준)
    k      : 참조할 이웃 시료 수
    max_distance : 이 거리보다 먼 곳은 결측(np.nan) 처리 - 외삽 방지
    """
    x, y, v = map(lambda a: np.asarray(a, float), (x, y, values))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[ok], y[ok], v[ok]
    if len(v) < 2:
        raise ValueError("보간에는 최소 2개 이상의 시료가 필요합니다.")

    H, W = like.shape
    rows, cols = np.mgrid[0:H, 0:W]
    gx, gy = rasterio.transform.xy(like.transform, rows.ravel(), cols.ravel())
    grid = np.column_stack([np.asarray(gx), np.asarray(gy)])

    tree = cKDTree(np.column_stack([x, y]))
    kk = min(k, len(v))
    dist, idx = tree.query(grid, k=kk)
    if kk == 1:
        dist, idx = dist[:, None], idx[:, None]

    dist = np.maximum(dist, 1e-12)
    w = 1.0 / dist ** power
    out = np.sum(w * v[idx], axis=1) / np.sum(w, axis=1)
    if max_distance is not None:
        out[dist[:, 0] > max_distance] = np.nan
    return out.reshape(H, W).astype("float32")


def isotope_group(d13c, d15n, n_groups: int = 2, random_state: int = 0):
    """d13C-d15N 산점도를 k-means로 나눠 생태 집단 라벨을 부여합니다."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    XY = np.column_stack([np.asarray(d13c, float), np.asarray(d15n, float)])
    ok = np.all(np.isfinite(XY), axis=1)
    labels = np.full(len(XY), -1, dtype=int)
    Z = StandardScaler().fit_transform(XY[ok])
    km = KMeans(n_clusters=n_groups, n_init=10, random_state=random_state).fit(Z)
    labels[ok] = km.labels_
    return labels
