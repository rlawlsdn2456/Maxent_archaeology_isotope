"""
'같은 지역 안에서의 시기 비교' 도구.

전체 자료를 시기별로만 비교하면, 시기마다 조사된 지역이 달라서 생기는 차이를
시간 변화로 오해하게 됩니다(공간 교락, spatial confounding).
이 모듈은 연구지역을 여러 소지역(zone)으로 나눈 뒤,
  - 각 소지역 안에서 시기가 연속으로 이어지는 구간만 골라 비교하고
  - 지역 효과를 뺀 순수한 시간 변화(지역 내 편차)를 계산합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (이름, lon_min, lat_min, lon_max, lat_max) - 앞에 있는 것이 우선 적용됩니다.
DEFAULT_ZONES = [
    ("연해주", 130.0, 42.0, 140.0, 49.0),
    ("아무르·하바롭스크", 125.0, 49.0, 141.0, 56.0),
    ("길림·흑룡강", 122.0, 43.0, 135.0, 49.0),
    ("몽골·자바이칼", 100.0, 44.0, 122.0, 56.0),
    ("요서·내몽골동부", 115.0, 40.0, 122.0, 45.0),
    ("요동·요동반도", 120.0, 38.0, 126.0, 43.0),
    ("화북", 105.0, 33.0, 120.0, 40.0),
]


def assign_zone(df: pd.DataFrame, zones=None, lon_col="lon", lat_col="lat") -> pd.Series:
    """좌표로 소지역(zone)을 배정합니다. 어디에도 안 들면 NaN."""
    zones = zones or DEFAULT_ZONES
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    out = pd.Series([None] * len(df), index=df.index, dtype=object)
    for name, x0, y0, x1, y1 in zones:
        m = out.isna() & lon.between(x0, x1) & lat.between(y0, y1)
        out[m] = name
    return out


def zone_period_table(df: pd.DataFrame, value_cols=("d13C_coll", "d15N_coll"),
                      zone_col="zone", period_col="period", min_n: int = 3) -> pd.DataFrame:
    """소지역 x 시기별 요약표(개체수·유적수·평균·표준편차)."""
    rows = []
    for (zone, period), sub in df.groupby([zone_col, period_col], dropna=True):
        row = {"zone": zone, "period": period, "n": len(sub),
               "유적수": sub["site"].nunique()}
        for c in value_cols:
            v = pd.to_numeric(sub[c], errors="coerce").dropna()
            row[f"{c}_n"] = len(v)
            row[f"{c}_평균"] = round(v.mean(), 2) if len(v) else np.nan
            row[f"{c}_표준편차"] = round(v.std(), 2) if len(v) > 1 else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    return out[out["n"] >= min_n].reset_index(drop=True) if len(out) else out


def continuous_sequences(table: pd.DataFrame, period_order, min_periods: int = 2,
                         zone_col="zone") -> dict:
    """소지역별로 '연속으로 이어지는 시기'의 구간을 찾습니다.

    반환: {소지역: [[시기1, 시기2, ...], ...]}  (연속 구간이 여러 개일 수 있음)
    """
    idx = {k: i for i, k in enumerate(period_order)}
    out = {}
    for zone, sub in table.groupby(zone_col):
        keys = sorted([k for k in sub["period"] if k in idx], key=lambda k: idx[k])
        runs, cur = [], []
        for k in keys:
            if not cur or idx[k] == idx[cur[-1]] + 1:
                cur.append(k)
            else:
                runs.append(cur)
                cur = [k]
        if cur:
            runs.append(cur)
        runs = [r for r in runs if len(r) >= min_periods]
        if runs:
            out[zone] = runs
    return out


def within_zone_change(df: pd.DataFrame, value_col: str, period_order,
                       zone_col="zone", period_col="period", min_n: int = 3) -> pd.DataFrame:
    """같은 소지역 안에서 이어지는 시기 사이의 변화량(Δ)과 검정 결과."""
    from scipy.stats import mannwhitneyu

    idx = {k: i for i, k in enumerate(period_order)}
    rows = []
    for zone, sub in df.groupby(zone_col):
        avail = [k for k in period_order if k in set(sub[period_col].dropna())]
        for a, b in zip(avail[:-1], avail[1:]):
            if idx[b] - idx[a] != 1:
                continue                      # 시기가 건너뛰면 비교하지 않음
            va = pd.to_numeric(sub.loc[sub[period_col] == a, value_col], errors="coerce").dropna()
            vb = pd.to_numeric(sub.loc[sub[period_col] == b, value_col], errors="coerce").dropna()
            if len(va) < min_n or len(vb) < min_n:
                continue
            try:
                stat, p = mannwhitneyu(va, vb, alternative="two-sided")
            except ValueError:
                stat, p = np.nan, np.nan
            rows.append({
                "소지역": zone, "변화": f"{a}→{b}", "변수": value_col,
                f"{a}_n": len(va), f"{a}_평균": round(va.mean(), 2),
                f"{b}_n": len(vb), f"{b}_평균": round(vb.mean(), 2),
                "Δ": round(vb.mean() - va.mean(), 2),
                "p(Mann-Whitney)": round(float(p), 5) if p == p else np.nan,
            })
    return pd.DataFrame(rows)


def detrend_by_zone(df: pd.DataFrame, value_col: str, zone_col="zone") -> pd.Series:
    """지역 평균을 뺀 값(지역 내 편차). 공간 차이를 제거하고 시간 변화만 보게 합니다."""
    v = pd.to_numeric(df[value_col], errors="coerce")
    return v - v.groupby(df[zone_col]).transform("mean")


def zone_period_stats(df: pd.DataFrame, value_col: str, zone_col="zone",
                      period_col="period", min_n: int = 4) -> pd.DataFrame:
    """소지역별로 '시기 사이에 차이가 있는가'를 Kruskal-Wallis로 검정."""
    from scipy.stats import kruskal

    rows = []
    for zone, sub in df.groupby(zone_col):
        groups = []
        labels = []
        for period, s in sub.groupby(period_col):
            v = pd.to_numeric(s[value_col], errors="coerce").dropna()
            if len(v) >= min_n:
                groups.append(v.to_numpy())
                labels.append(period)
        if len(groups) < 2:
            continue
        H, p = kruskal(*groups)
        rows.append({"소지역": zone, "변수": value_col, "비교시기": ", ".join(labels),
                     "집단수": len(groups), "H": round(float(H), 2),
                     "p": float(f"{p:.3g}"),
                     "유의(α=0.05)": "예" if p < 0.05 else "아니오"})
    return pd.DataFrame(rows)
