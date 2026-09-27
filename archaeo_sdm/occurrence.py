"""출토지점(occurrence) 자료 읽기 - SHP / CSV / XLSX 모두 지원."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# 자주 쓰이는 경위도 컬럼 이름 후보(한글 포함)
LON_CANDIDATES = ["lon", "long", "longitude", "x", "경도", "위경도_x", "동경"]
LAT_CANDIDATES = ["lat", "latitude", "y", "위도", "위경도_y", "북위"]


def _guess(cols, candidates):
    low = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    for c in cols:  # 부분 일치
        if any(cand in str(c).strip().lower() for cand in candidates):
            return c
    return None


def load_points(path: str, lon_col: str | None = None, lat_col: str | None = None,
                crs: str = "EPSG:4326"):
    """점 자료를 읽어 (DataFrame, lon, lat, crs)를 반환합니다.

    - .shp / .gpkg / .geojson : geopandas로 읽고 기하에서 좌표를 뽑습니다.
    - .csv / .xlsx            : 경위도 컬럼을 찾아 사용합니다(컬럼명 자동 추정).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".shp", ".gpkg", ".geojson", ".json"):
        import geopandas as gpd

        gdf = gpd.read_file(path)
        gdf = gdf[gdf.geometry.notna()].copy()
        pts = gdf.geometry.representative_point()
        df = pd.DataFrame(gdf.drop(columns=gdf.geometry.name))
        return df, pts.x.to_numpy(), pts.y.to_numpy(), str(gdf.crs) if gdf.crs else crs

    if ext in (".xlsx", ".xls"):
        df = pd.read_csv(path) if False else pd.read_excel(path)
    else:
        df = _read_csv_any_encoding(path)

    lon_col = lon_col or _guess(df.columns, LON_CANDIDATES)
    lat_col = lat_col or _guess(df.columns, LAT_CANDIDATES)
    if lon_col is None or lat_col is None:
        raise ValueError(
            f"경위도 컬럼을 찾지 못했습니다. 현재 컬럼: {list(df.columns)}\n"
            "lon_col=, lat_col= 로 직접 지정해 주세요."
        )
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy()
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy()
    keep = np.isfinite(lon) & np.isfinite(lat)
    return df.loc[keep].reset_index(drop=True), lon[keep], lat[keep], crs


def _read_csv_any_encoding(path: str) -> pd.DataFrame:
    """한글 CSV는 cp949/utf-8-sig가 섞여 있어 순서대로 시도합니다."""
    last = None
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, LookupError) as e:  # pragma: no cover
            last = e
    raise last
