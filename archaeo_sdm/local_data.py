"""
연구자가 직접 정리한 표(연해주·아무르·자바이칼 등)를 표준 표로 읽어들입니다.

이런 표에는 흔히 두 가지 문제가 있습니다.
  1) 중간부터 열이 밀려 있음 (좌표 칸에 유적명이 들어가는 등)
  2) 연대가 '1200-400 BCE', '398–494 AD' 같은 문자열
이 모듈은 두 가지를 모두 처리하고, 좌표가 없는 행은 gazetteer로 채우게 넘깁니다.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .isotope_db import STD_COLS, classify_taxon

# "1200-400 BCE", "398–494 AD", "265-420AD", "B.C 2 - A.D 1", "100BC-390AD" 등
_ERA = r"(?:B\.?\s?C\.?E?|BCE|BC)"
_CE = r"(?:A\.?\s?D\.?|CE)"


def parse_date_range(text) -> tuple[float, float]:
    """연대 문자열을 (bp_upper, bp_lower)로 변환. 못 읽으면 (nan, nan)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return (np.nan, np.nan)
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return (np.nan, np.nan)
    s = s.replace("–", "-").replace("~", "-").replace("—", "-")

    # 각 숫자에 붙은 시대 표기를 개별적으로 해석
    tokens = re.findall(rf"({_ERA}|{_CE})?\s*(\d{{1,4}})\s*({_ERA}|{_CE})?", s, flags=re.I)
    years = []
    trailing_era = None
    for pre, num, post in tokens:
        if not num:
            continue
        era = (pre or post or "").upper().replace(".", "").replace(" ", "")
        if era:
            trailing_era = era
        years.append((int(num), era))
    if not years:
        return (np.nan, np.nan)

    # 앞쪽 숫자에 시대 표기가 없으면 뒤쪽 표기를 따름 ("1200-400 BCE")
    cal = []
    for num, era in years:
        e = era or trailing_era or "CE"
        cal.append(-num if e.startswith("B") else num)
    # 'B.C 2 - A.D 1' 처럼 숫자가 모두 작으면 '세기' 표기로 봅니다.
    if all(abs(c) <= 25 for c in cal):
        conv = []
        for c in cal:
            conv += [(c + 1) * 100, c * 100] if c < 0 else [(c - 1) * 100 + 1, c * 100]
        cal = conv
    if len(cal) == 1:
        cal = [cal[0], cal[0]]
    y_old, y_new = min(cal), max(cal)
    return (1950 - y_old, 1950 - y_new)      # BP는 클수록 오래됨


def load_ecology_table(path: str, sheet=0) -> pd.DataFrame:
    """'생태권용 데이터' 형식의 표를 표준 표로 읽습니다.

    두 가지 행 배열이 섞여 있습니다.
      A형: ... date, lat, lon, site, feature, species, sex, age, d13C, d15N
      B형: ... date, site, feature, species, d13C, d15N, %C, %N, C:N   (열이 2칸 밀림)
    'lattitude' 칸이 숫자가 아니면 B형으로 판단합니다.
    """
    raw = pd.read_excel(path, sheet_name=sheet)
    raw = raw.dropna(how="all").reset_index(drop=True)
    cols = list(raw.columns)

    def col(name, default=None):
        for c in cols:
            if str(c).strip().lower() == name:
                return c
        return default

    c_region, c_cult = col("region"), col("culture")
    c_period, c_date = col("period"), col("date")
    c_lat, c_lon = col("lattitude", col("latitude")), col("longtidue", col("longitude"))
    c_site, c_feat = col("site"), col("feature form")
    c_sp = col("spieces", col("species"))
    c_sex, c_age = col("sex"), col("age")
    c_c, c_n = col("δ13c"), col("δ15n")

    rows = []
    for _, r in raw.iterrows():
        lat_raw = r.get(c_lat)
        shifted = isinstance(lat_raw, str) and not _is_number(lat_raw)
        if shifted:                       # B형: 열이 두 칸 밀린 행
            site = r.get(c_lat)
            species = r.get(c_site)
            d13c, d15n = _num(r.get(c_feat)), _num(r.get(c_sp))
            cn = _num(r.get(c_c))
            lat = lon = np.nan
        else:                             # A형
            site = r.get(c_site)
            species = r.get(c_sp)
            d13c, d15n = _num(r.get(c_c)), _num(r.get(c_n))
            cn = np.nan
            lat, lon = _num(lat_raw), _num(r.get(c_lon))
        if pd.isna(d13c) and pd.isna(d15n):
            continue
        rows.append({
            "source": "연구자 정리표",
            "group": "human" if "homo" in str(species).lower() else "animal",
            "site": str(site).strip() if site is not None else np.nan,
            "lat": lat, "lon": lon,
            "region": _norm_region(r.get(c_region)),
            "culture": r.get(c_cult),
            "taxon": species,
            "d13C_coll": d13c, "d15N_coll": d15n, "CN": cn,
            "period_text": r.get(c_period), "date_text": r.get(c_date),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=STD_COLS)

    bp = df["date_text"].map(parse_date_range)
    df["bp_upper"] = [b[0] for b in bp]
    df["bp_lower"] = [b[1] for b in bp]
    df["bp_mid"] = (df["bp_upper"] + df["bp_lower"]) / 2
    df["taxon_group"] = df["taxon"].map(classify_taxon)
    df.loc[df["group"] == "human", "taxon_group"] = "사람"
    for c in STD_COLS:
        if c not in df.columns:
            df[c] = np.nan
    keep = STD_COLS + ["period_text", "date_text"]
    return df[keep].copy()


def load_point_isotopes(shapefile: str, csv_path: str | None = None, join_on: str = "id",
                        site_field: str = "유적명", value_cols=("13C", "15N"),
                        source: str = "연해주 SHP") -> pd.DataFrame:
    """좌표를 가진 SHP(+동위원소 CSV)를 표준 표로 읽습니다."""
    import geopandas as gpd

    g = gpd.read_file(shapefile)
    g = g[g.geometry.notna()].to_crs("EPSG:4326")
    pts = g.geometry.representative_point()
    df = pd.DataFrame(g.drop(columns=g.geometry.name))
    df["lat"], df["lon"] = pts.y.to_numpy(), pts.x.to_numpy()

    if csv_path:
        from .occurrence import _read_csv_any_encoding
        iso = _read_csv_any_encoding(csv_path)
        df = df.merge(iso, on=join_on, how="left")

    out = pd.DataFrame({
        "source": source,
        "group": "animal",
        "site": df[site_field] if site_field in df.columns else np.nan,
        "lat": df["lat"], "lon": df["lon"],
        "region": "Primorye",
        "culture": np.nan,
        "taxon": "Pig",
        "d13C_coll": pd.to_numeric(df.get(value_cols[0]), errors="coerce"),
        "d15N_coll": pd.to_numeric(df.get(value_cols[1]), errors="coerce"),
    })
    out["taxon_group"] = "돼지"
    for c in STD_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[STD_COLS].copy()


REGION_FIX = {"transbaika": "Transbaikal", "transbaikal": "Transbaikal",
              "primorye": "Primorye", "amur": "Amur", "east mongolia": "East Mongolia",
              "north china": "North China", "liaoning": "Liaoning",
              "inner mongolia": "Inner Mongolia", "jillin province": "Jilin",
              "jilin": "Jilin"}


def _norm_region(v) -> str:
    s = str(v).strip().lower()
    return REGION_FIX.get(s, str(v).strip() if s not in ("", "nan") else np.nan)


def _is_number(v) -> bool:
    try:
        float(str(v).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def _num(v):
    return pd.to_numeric(v, errors="coerce")
