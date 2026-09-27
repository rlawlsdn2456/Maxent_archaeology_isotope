"""
대규모 동위원소 데이터베이스를 하나의 표준 표(tidy table)로 정리합니다.

지원하는 원자료
  1) Isotope Dataset for Archaeological Biological Remains in China
     - 시트 4개: Human/Animal x organic(콜라겐) / bioapatite(아파타이트)
  2) IsoMemo/CIMA 계열 유라시아 데이터셋 (몽골·카자흐스탄·러시아)

표준 컬럼
  source, group(human/animal), site, lat, lon, region, culture,
  bp_upper, bp_lower, bp_mid, taxon, taxon_group,
  d13C_coll, d15N_coll, CN, d13C_ap, d18O_ap
"""

from __future__ import annotations

import numpy as np
import pandas as pd

STD_COLS = ["source", "group", "site", "lat", "lon", "region", "culture",
            "bp_upper", "bp_lower", "bp_mid", "taxon", "taxon_group",
            "d13C_coll", "d15N_coll", "CN", "d13C_ap", "d18O_ap"]

# 분석에 쓰기 좋은 큰 분류군으로 묶기 (원자료의 Common Name 기준)
TAXON_GROUPS = {
    "돼지": ("pig", "piglet", "wild boar", "sus"),
    "개": ("dog", "canis", "canis familiaris", "canis sp"),
    "소": ("cattle", "aurochs", "bovini", "water buffalo", "bos", "cattle/water buffalo"),
    "양·염소": ("sheep", "goat", "sheep/goat", "caprine", "ovis", "capra"),
    "말": ("horse", "equus"),
    "사슴": ("deer", "red deer", "sika deer", "roe deer", "water deer", "cervus", "elk", "moose"),
    "사람": ("human", "homo sapien", "homo sapiens"),
}


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def classify_taxon(name) -> str:
    """세부 종명을 큰 분류군으로 묶습니다. 못 찾으면 '기타'."""
    s = str(name).strip().lower()
    if not s or s == "nan":
        return "기타"
    for grp, keys in TAXON_GROUPS.items():
        if any(k == s for k in keys):
            return grp
    for grp, keys in TAXON_GROUPS.items():
        if any(k in s for k in keys):
            return grp
    return "기타"


# --------------------------------------------------------------- 중국 데이터셋
def load_china_isotope_db(path: str) -> pd.DataFrame:
    """중국 동위원소 데이터셋 4개 시트를 읽어 표준 표로 합칩니다.

    같은 개체가 콜라겐 시트와 아파타이트 시트에 나뉘어 있으므로,
    유적·시료ID 기준으로 합쳐서 한 행에 C·N·C·O가 모두 오도록 만듭니다.
    """
    x = pd.ExcelFile(path)
    frames = []

    org_map = {
        "Human organic materials": "human",
        "Animal organic materials": "animal",
    }
    ap_map = {
        "Human bioapatite": "human",
        "Animal bioapatite": "animal",
    }

    def base(d, grp):
        out = pd.DataFrame(index=d.index)
        out["source"] = "China DB"
        out["group"] = grp
        out["site"] = d.get("Site Name")
        out["lat"] = _to_num(d.get("Latitude"))
        out["lon"] = _to_num(d.get("Longitude"))
        out["region"] = d.get("Province")
        out["culture"] = d.get("Culture/Chronology")
        out["bp_upper"] = _to_num(d.get("Estimated Year  Upper (BP)"))
        out["bp_lower"] = _to_num(d.get("Estimated Year Lower (BP)"))
        out["sample_id"] = d.get("Sample ID").astype(str) if "Sample ID" in d else ""
        if grp == "human":
            out["taxon"] = "Human"
        else:
            out["taxon"] = d.get("Common Name").fillna(d.get("Genus&Species"))
        return out

    for sheet, grp in org_map.items():
        d = x.parse(sheet)
        b = base(d, grp)
        b["d13C_coll"] = _to_num(d.get("δ13C (‰, VPDB)"))
        b["d15N_coll"] = _to_num(d.get("δ15N (‰, AIR)"))
        b["CN"] = _to_num(d.get("Atomic C:N Ratio"))
        frames.append(b)

    ap_frames = []
    for sheet, grp in ap_map.items():
        d = x.parse(sheet)
        b = base(d, grp)
        c13 = d.get("δ13CCarbonate (‰, VPDB)")
        if c13 is None:
            c13 = d.get("δ13C Carbonate (‰, VPDB)")
        o18 = d.get("δ18OCarbonate (‰, VPDB)")
        if o18 is None:
            o18 = d.get("δ18O Carbonate (‰, VPDB)")
        b["d13C_ap"] = _to_num(c13)
        b["d18O_ap"] = _to_num(o18)
        ap_frames.append(b)

    org = pd.concat(frames, ignore_index=True)
    ap = pd.concat(ap_frames, ignore_index=True)

    # 유적+시료ID로 아파타이트 값을 붙이고, 짝이 없는 아파타이트는 별도 행으로 추가
    key = ["site", "sample_id", "group"]
    ap_small = ap[key + ["d13C_ap", "d18O_ap"]].dropna(subset=["d13C_ap", "d18O_ap"], how="all")
    # 시료ID가 비어 있거나 중복이면 결합 기준이 될 수 없습니다.
    # (그대로 두면 'nan' 끼리 결합되어 행이 폭증합니다)
    valid_id = ap_small["sample_id"].notna() & ~ap_small["sample_id"].isin(["", "nan", "None"])
    ap_small = ap_small[valid_id].drop_duplicates(subset=key, keep="first")
    merged = org.merge(ap_small, on=key, how="left", validate="many_to_one")
    matched = set(map(tuple, merged.loc[merged["d13C_ap"].notna() | merged["d18O_ap"].notna(), key]
                      .astype(str).values))
    unmatched = ap[~ap[key].astype(str).apply(tuple, axis=1).isin(matched)]

    out = pd.concat([merged, unmatched], ignore_index=True)
    out["taxon_group"] = out["taxon"].map(classify_taxon)
    out.loc[out["group"] == "human", "taxon_group"] = "사람"
    return _finalize(out)


# ------------------------------------------------------- IsoMemo/CIMA 데이터셋
def load_isomemo_db(path: str, sheet=0) -> pd.DataFrame:
    """유라시아(몽골·카자흐스탄·러시아 등) IsoMemo 계열 데이터셋."""
    d = pd.ExcelFile(path).parse(sheet)
    out = pd.DataFrame(index=d.index)
    out["source"] = "IsoMemo/CIMA"
    out["site"] = d.get("Site name")
    out["lat"] = _to_num(d.get("Latitude"))
    out["lon"] = _to_num(d.get("Longitude"))
    out["region"] = d.get("Country")
    out["culture"] = d.get("Archae. Culture")

    # 이 데이터셋의 연대는 달력연대(BCE는 음수). BP로 변환.
    ymin = _to_num(d.get("Min age (95%)"))
    ymax = _to_num(d.get("Max age (95%)"))
    out["bp_upper"] = 1950 - pd.concat([ymin, ymax], axis=1).min(axis=1)
    out["bp_lower"] = 1950 - pd.concat([ymin, ymax], axis=1).max(axis=1)

    out["taxon"] = d.get("Common name").fillna(d.get("Species"))
    out["d13C_coll"] = _to_num(d.get("delta 13C coll"))
    out["d15N_coll"] = _to_num(d.get("delta 15N coll"))
    out["CN"] = _to_num(d.get("C/N"))
    out["d13C_ap"] = _to_num(d.get("delta 13C carb bioap"))
    out["d18O_ap"] = _to_num(d.get("delta 18O carb bioap"))
    out["taxon_group"] = out["taxon"].map(classify_taxon)
    sp = d.get("Species").astype(str).str.lower()
    out["group"] = np.where(sp.str.contains("homo"), "human", "animal")
    out.loc[out["group"] == "human", "taxon_group"] = "사람"
    return _finalize(out)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df["bp_mid"] = (df["bp_upper"] + df["bp_lower"]) / 2
    for c in STD_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[STD_COLS].copy()


# ------------------------------------------------------------------- 필터들
def filter_region(df: pd.DataFrame, bbox=None, regions=None) -> pd.DataFrame:
    """bbox=(lon_min, lat_min, lon_max, lat_max) 또는 지역명 목록으로 자릅니다."""
    out = df[df["lat"].notna() & df["lon"].notna()].copy()
    if bbox:
        x0, y0, x1, y1 = bbox
        out = out[(out.lon >= x0) & (out.lon <= x1) & (out.lat >= y0) & (out.lat <= y1)]
    if regions:
        pat = "|".join(regions)
        out = out[out["region"].astype(str).str.contains(pat, case=False, na=False)]
    return out


def quality_filter(df: pd.DataFrame, cn_range=(2.9, 3.6), require: str = "any") -> pd.DataFrame:
    """콜라겐 보존 상태(C:N 원자비)로 거릅니다.

    DeNiro(1985) 기준 2.9~3.6을 벗어나면 속성변화(diagenesis) 가능성이 큽니다.
    C:N이 기록되지 않은 자료는 남깁니다(결측을 탈락으로 처리하면 손실이 큼).
    """
    out = df.copy()
    bad = out["CN"].notna() & ((out["CN"] < cn_range[0]) | (out["CN"] > cn_range[1]))
    out = out[~bad]
    if require == "collagen":
        out = out[out["d13C_coll"].notna() & out["d15N_coll"].notna()]
    elif require == "apatite":
        out = out[out["d13C_ap"].notna() | out["d18O_ap"].notna()]
    return out


def site_summary(df: pd.DataFrame, by=("period", "taxon_group")) -> pd.DataFrame:
    """시기 x 분류군별 시료 수·유적 수·동위원소 평균 요약표."""
    by = [b for b in by if b in df.columns]
    g = df.groupby(by, dropna=False)
    out = g.agg(
        n=("d13C_coll", "size"),
        유적수=("site", pd.Series.nunique),
        d13C_평균=("d13C_coll", "mean"),
        d13C_표준편차=("d13C_coll", "std"),
        d15N_평균=("d15N_coll", "mean"),
        d15N_표준편차=("d15N_coll", "std"),
        d13Cap_평균=("d13C_ap", "mean"),
        d18Oap_평균=("d18O_ap", "mean"),
    ).reset_index()
    return out.round(2)
