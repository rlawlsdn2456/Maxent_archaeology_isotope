"""
고고학적 편년(chronology) 정의와 시기 배정.

동위원소 자료는 '문화명'(예: Lower Xiajiadian Culture)과 '추정 연대(BP)'가 섞여 있습니다.
이 모듈은 두 가지를 모두 받아서 하나의 시기(period)로 정리합니다.

  1) 절대연대(BP 상한/하한)의 중앙값이 어느 구간에 들어가는지로 우선 배정
  2) 연대가 없으면 문화/왕조 이름의 키워드로 배정

BP는 1950년 기준 '몇 년 전'이며, 값이 클수록 오래된 것입니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Period:
    key: str                     # 짧은 식별자 (파일명에 사용)
    label_ko: str                # 한글 이름
    label_en: str
    bp_start: float              # 오래된 쪽 (큰 값)
    bp_end: float                # 최근 쪽 (작은 값)
    keywords: tuple = field(default_factory=tuple)   # 문화/왕조명 키워드

    @property
    def bp_mid(self) -> float:
        return (self.bp_start + self.bp_end) / 2

    def calendar(self) -> str:
        """BP를 BCE/CE 표기로."""
        def conv(bp):
            y = 1950 - bp
            return f"BCE {int(abs(y))}" if y < 0 else f"CE {int(y)}"
        return f"{conv(self.bp_start)}~{conv(self.bp_end)}"


# 만주·북중국 기준 편년 (청동기 ~ 요금)
MANCHURIA_PERIODS = [
    Period("EBA", "청동기 전기(하가점하층기)", "Early Bronze Age", 4000, 3500,
           ("lower xiajiadian", "xiajiadian (lower", "erlitou", "xia dynasty", "proto-shang")),
    Period("LBA", "청동기 후기(하가점상층·상주)", "Late Bronze Age", 3500, 2800,
           ("upper xiajiadian", "shang dynasty", "western zhou", "shang to zhou",
            "bronze age (late)", "weiyingzi")),
    Period("EIA", "초기철기(춘추전국)", "Early Iron Age", 2800, 2200,
           ("spring and autumn", "warring states", "eastern zhou", "zhou dynasty",
            "early iron age", "yankovskaia", "paleometal", "baijinbao", "백금보")),
    Period("HAN", "한·원삼국(부여·한사군)", "Han / Proto-Three-Kingdoms", 2200, 1700,
           ("han dynasty", "qin to han", "western han", "eastern han", "qin dynasty",
            "fuyu", "buyeo", "부여", "han commandery", "xiongnu", "흉노", "polsetskaia", "폴체")),
    Period("GOG", "삼연·선비·고구려", "Sanyan / Xianbei / Goguryeo", 1700, 1300,
           ("xianbei", "선비", "wei to jin", "northern wei", "jin dynasty (265", "sanyan",
            "goguryeo", "koguryo", "고구려", "sixteen kingdoms", "tuoba")),
    Period("BHL", "발해·요금", "Bohai / Liao-Jin", 1300, 700,
           ("bohai", "balhae", "발해", "liao to jin", "liao dynasty", "liao jin",
            "jin dynasties", "tang dynasty", "khitan", "jurchen", "여진", "malgal",
            "말갈", "mongol empire", "몽골")),
]

PERIOD_ORDER = [p.key for p in MANCHURIA_PERIODS]


def assign_period(df: pd.DataFrame, periods=None,
                  bp_upper_col: str = "bp_upper", bp_lower_col: str = "bp_lower",
                  culture_col: str = "culture") -> pd.Series:
    """각 행에 시기 key를 배정합니다. 어디에도 안 들어가면 NaN."""
    periods = periods or MANCHURIA_PERIODS
    up = pd.to_numeric(df.get(bp_upper_col), errors="coerce")
    lo = pd.to_numeric(df.get(bp_lower_col), errors="coerce")
    mid = (up + lo) / 2
    mid = mid.fillna(up).fillna(lo)

    out = pd.Series(np.nan, index=df.index, dtype=object)

    # 1) 절대연대 기준
    for p in periods:
        m = out.isna() & mid.notna() & (mid <= p.bp_start) & (mid > p.bp_end)
        out[m] = p.key

    # 2) 문화명 키워드 기준 (연대가 없는 자료만)
    if culture_col in df.columns:
        cult = df[culture_col].fillna("").astype(str).str.lower()
        for p in periods:
            if not p.keywords:
                continue
            hit = cult.apply(lambda s, kw=p.keywords: any(k in s for k in kw))
            out[out.isna() & hit] = p.key
    return out


def period_table(periods=None) -> pd.DataFrame:
    periods = periods or MANCHURIA_PERIODS
    return pd.DataFrame([{
        "key": p.key, "시기": p.label_ko, "Period": p.label_en,
        "BP": f"{int(p.bp_start)}–{int(p.bp_end)}", "연대": p.calendar(),
    } for p in periods])


def infer_period_from_site(df: pd.DataFrame, period_col: str = "period",
                           site_col: str = "site"):
    """연대가 없는 행에, 같은 유적의 다른 시료가 가진 시기를 물려줍니다.

    좌표만 있고 연대가 없는 자료(예: 유적명·동위원소만 있는 SHP)를 살리기 위한 것입니다.
    같은 유적이면 층위가 다를 수 있으므로, 물려받은 행은 period_source='유적상속'으로
    표시해 나중에 걸러낼 수 있게 합니다.

    반환: (시기 Series, 물려받았는지 여부 Series)
    """
    from .gazetteer import normalize_name

    key = df[site_col].map(normalize_name)
    have = df[period_col].notna()
    if not have.any():
        return df[period_col], pd.Series(False, index=df.index)

    # 유적별 최빈 시기
    known = (pd.DataFrame({"k": key[have], "p": df.loc[have, period_col]})
             .groupby("k")["p"].agg(lambda s: s.mode().iat[0]))
    filled = df[period_col].copy()
    cand = key.map(known)
    need = filled.isna() & cand.notna()
    filled[need] = cand[need]
    return filled, need
