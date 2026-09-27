"""
유적 좌표 사전(gazetteer).

직접 정리한 동위원소 표에는 유적 '이름'만 있고 좌표가 없는 경우가 많습니다.
이 모듈은 좌표를 가진 다른 자료(SHP, 대규모 DB)에서 이름을 맞춰 좌표를 채우고,
끝내 못 채운 유적을 목록으로 돌려줍니다.

좌표의 출처와 정확도를 반드시 함께 기록합니다(coord_source, coord_quality).
  exact       : 원자료에 좌표가 들어 있음
  name_match  : 다른 DB에서 같은 이름의 유적을 찾아 가져옴
  manual      : 사람이 직접 입력한 gazetteer.csv 항목
  approx      : 지역 중심점으로 대신함 (아이소스케이프에는 기본적으로 쓰지 않음)

approx 좌표를 아이소스케이프에 넣으면 '있지도 않은 곳의 값'을 만들어내므로,
기본값에서는 제외됩니다(use_approx=True로 명시할 때만 사용).
"""

from __future__ import annotations

import os
import re
import unicodedata

import numpy as np
import pandas as pd

# 한글/러시아어 표기와 영문 표기를 잇는 별칭
ALIASES = {
    "체레파하13": "cherepakha 13",
    "체르냐찌노2": "chernyatino 2",
    "루스키1": "russkiy 1",
    "보스펠로보1": "pospelovo 1",
    "나지모바1": "nazimova 1",
    "russiky-1": "russkiy 1",
    "pospelovo i": "pospelovo 1",
    "nazimova i": "nazimova 1",
    "atsai ii": "atsai 2",
    "i'lmovaya pad'": "ilmovaya pad",
    "ilmovaya pad'": "ilmovaya pad",
    "ivolgin": "ivolga",
    "the bagou": "bagou",
    "the east wuzhuer": "dongwuzhuer",
    "lamadong site": "lamadong",
    "honghe  site": "honghe",
    "honghe site": "honghe",
    "weizili site": "weizili",
    "changshan site": "changshan",
    "troitsky cemeter": "troitsky",
    "troitsky cemetery": "troitsky",
}

# 지역 중심점(대략) - 좌표를 못 찾은 유적의 위치를 '대강' 표시할 때만 사용
REGION_CENTROIDS = {
    "primorye": (43.5, 132.0),
    "amur": (50.3, 127.5),
    "transbaikal": (51.5, 107.5),
    "transbaika": (51.5, 107.5),
    "east mongolia": (47.5, 112.0),
    "north china": (38.0, 114.0),
    "liaoning": (41.5, 122.0),
    "inner mongolia": (43.0, 115.0),
    "jillin province": (43.5, 126.0),
    "jilin": (43.5, 126.0),
}

_SUFFIXES = (" site", " cemetery", " cemeter", " settlement", " tomb", " tombs",
             " burial", " group", " ruins")


def normalize_name(name) -> str:
    """유적 이름을 비교하기 좋게 정규화."""
    s = str(name)
    if s.strip().lower() in ("", "nan", "none"):
        return ""
    s = unicodedata.normalize("NFKC", s).strip().lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = ALIASES.get(s, s)
    s = re.sub(r"[\-_,.()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for suf in _SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
    # 로마숫자 -> 아라비아숫자
    for rom, ar in (("iii", "3"), ("ii", "2"), ("iv", "4"), ("i", "1")):
        s = re.sub(rf"\b{rom}\b", ar, s)
    s = ALIASES.get(s, s)
    return s


def gazetteer_from_frames(frames, source_name: str = "name_match") -> pd.DataFrame:
    """좌표를 가진 표들에서 (정규화 이름 -> 좌표) 사전을 만듭니다."""
    rows = []
    for df in frames:
        d = df[["site", "lat", "lon"]].dropna()
        for site, lat, lon in d.itertuples(index=False):
            key = normalize_name(site)
            if key:
                rows.append({"key": key, "site": site, "lat": float(lat),
                             "lon": float(lon), "coord_source": source_name})
    if not rows:
        return pd.DataFrame(columns=["key", "site", "lat", "lon", "coord_source"])
    g = pd.DataFrame(rows)
    # 같은 이름이 여러 번 나오면 중앙값으로
    agg = g.groupby("key").agg(site=("site", "first"), lat=("lat", "median"),
                               lon=("lon", "median"),
                               coord_source=("coord_source", "first")).reset_index()
    return agg


def gazetteer_from_shapefile(path: str, name_field: str,
                             source_name: str = "shapefile") -> pd.DataFrame:
    """좌표를 가진 SHP에서 유적 사전을 만듭니다(경위도로 변환)."""
    import geopandas as gpd

    g = gpd.read_file(path)
    g = g[g.geometry.notna()].to_crs("EPSG:4326")
    pts = g.geometry.representative_point()
    d = pd.DataFrame({"site": g[name_field].astype(str),
                      "lat": pts.y.to_numpy(), "lon": pts.x.to_numpy()})
    return gazetteer_from_frames([d], source_name)


def merge_gazetteers(*gazes) -> pd.DataFrame:
    """앞에 오는 사전이 우선순위가 높습니다."""
    out = pd.concat([g for g in gazes if g is not None and len(g)], ignore_index=True)
    return out.drop_duplicates(subset="key", keep="first")


def resolve_coordinates(df: pd.DataFrame, gaz: pd.DataFrame,
                        region_col: str = "region", use_approx: bool = False):
    """df의 site 이름으로 좌표를 채웁니다.

    반환: (좌표가 채워진 df, 못 채운 유적 목록 DataFrame)
    """
    out = df.copy()
    if "coord_source" not in out.columns:
        out["coord_source"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    else:
        out["coord_source"] = out["coord_source"].astype(object)
    have = out["lat"].notna() & out["lon"].notna()
    out.loc[have, "coord_source"] = out.loc[have, "coord_source"].fillna("exact")

    key = out["site"].map(normalize_name)
    lut = gaz.set_index("key")[["lat", "lon", "coord_source"]] if len(gaz) else None
    if lut is not None:
        m = key.map(lut["lat"])
        need = ~have & m.notna()
        out.loc[need, "lat"] = m[need]
        out.loc[need, "lon"] = key[need].map(lut["lon"])
        out.loc[need, "coord_source"] = key[need].map(lut["coord_source"]).astype(object)

    still = out["lat"].isna() | out["lon"].isna()
    if use_approx and region_col in out.columns:
        reg = out[region_col].astype(str).str.strip().str.lower()
        cen_lat = reg.map(lambda r: REGION_CENTROIDS.get(r, (np.nan, np.nan))[0])
        cen_lon = reg.map(lambda r: REGION_CENTROIDS.get(r, (np.nan, np.nan))[1])
        fill = still & cen_lat.notna()
        out.loc[fill, "lat"] = cen_lat[fill]
        out.loc[fill, "lon"] = cen_lon[fill]
        out.loc[fill, "coord_source"] = "approx"
        still = out["lat"].isna() | out["lon"].isna()

    unresolved = (out.loc[still, ["site", region_col]]
                  .assign(정규화이름=key[still])
                  .drop_duplicates().reset_index(drop=True)) if still.any() else \
        pd.DataFrame(columns=["site", region_col, "정규화이름"])
    out["coord_quality"] = out["coord_source"].map(
        {"exact": "정확", "shapefile": "정확", "name_match": "이름대조",
         "manual": "수기입력", "approx": "지역근사"}).fillna("없음")
    return out, unresolved


def save_gazetteer(gaz: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    gaz.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_gazetteer(path: str) -> pd.DataFrame:
    """사람이 직접 채운 gazetteer.csv (key, site, lat, lon)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["key", "site", "lat", "lon", "coord_source"])
    g = pd.read_csv(path)
    if "key" not in g.columns:
        g["key"] = g["site"].map(normalize_name)
    g["coord_source"] = g.get("coord_source", "manual")
    return g[["key", "site", "lat", "lon", "coord_source"]]
