"""DEM에서 지형 변수(경사, 사면향, 지형기복)를 계산합니다.

QGIS의 '경사도/사면향' 도구와 같은 Horn(1981) 방식이라, 굳이 QGIS에서 미리
만들어 두지 않아도 코드에서 바로 만들 수 있습니다.
"""

from __future__ import annotations

import numpy as np


def _cellsize_m(stack) -> tuple[float, float]:
    """격자 한 칸의 크기를 미터로 환산(경위도 좌표계면 대략 변환)."""
    t = stack.transform
    xres, yres = abs(t.a), abs(t.e)
    if stack.crs and stack.crs.is_geographic:
        H, W = stack.shape
        import rasterio
        _, lat = rasterio.transform.xy(t, H // 2, W // 2)
        xres *= 111320.0 * np.cos(np.radians(lat))
        yres *= 110540.0
    return float(xres), float(yres)


def slope_aspect(dem: np.ndarray, stack, degrees: bool = True):
    """Horn 방식 경사도(도)와 사면향(도, 북=0, 시계방향)."""
    z = np.asarray(dem, dtype=float)
    xres, yres = _cellsize_m(stack)
    # 3x3 이웃을 만들어 가장자리는 가장자리 값으로 채움
    p = np.pad(z, 1, mode="edge")
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, _, f = p[1:-1, :-2], p[1:-1, 1:-1], p[1:-1, 2:]
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]

    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * xres)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * yres)

    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.degrees(np.arctan2(dzdy, -dzdx)) % 360.0
    if degrees:
        slope = np.degrees(slope)
    return slope.astype("float32"), aspect.astype("float32")


def northness_eastness(aspect_deg: np.ndarray):
    """사면향(0~360도)은 순환값이라 모델에 그대로 넣으면 안 됩니다.
    cos/sin으로 나눠 '북향도'와 '동향도' 두 변수로 씁니다."""
    a = np.radians(np.asarray(aspect_deg, dtype=float))
    return np.cos(a).astype("float32"), np.sin(a).astype("float32")


def tri(dem: np.ndarray) -> np.ndarray:
    """지형 기복 지수(Terrain Ruggedness Index, Riley et al. 1999)."""
    z = np.asarray(dem, dtype=float)
    p = np.pad(z, 1, mode="edge")
    diffs = [(p[dy:dy + z.shape[0], dx:dx + z.shape[1]] - z) ** 2
             for dy in (0, 1, 2) for dx in (0, 1, 2) if not (dy == 1 and dx == 1)]
    return np.sqrt(np.sum(diffs, axis=0)).astype("float32")


def add_terrain_layers(stack, dem_name: str, include=("slope", "northness", "tri")):
    """RasterStack에 지형 파생 변수를 추가합니다."""
    dem = stack.data[stack.names.index(dem_name)]
    slope, aspect = slope_aspect(dem, stack)
    north, east = northness_eastness(aspect)
    layers = {"slope": slope, "aspect": aspect, "northness": north,
              "eastness": east, "tri": tri(dem)}
    for key in include:
        stack.add_layer(key, layers[key])
    return stack
