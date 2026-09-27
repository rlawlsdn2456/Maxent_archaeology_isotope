# -*- coding: utf-8 -*-
"""
북중국·만주 지역, 청동기~요금 시기의 사람·동물 동위원소 통합 분석.

들어가는 자료
  - 중국 고고학 생물유체 동위원소 데이터셋 (2026.3 업데이트, 4개 시트)
  - IsoMemo/CIMA 유라시아 데이터셋 (몽골·카자흐스탄·러시아)
  - WorldClim 생물기후 (고기후 자료로 교체 가능: paleoenv.paleoclim_layers)

나오는 것
  - 시기별 아이소스케이프(콜라겐 δ13C·δ15N, 아파타이트 δ13C·δ18O) GeoTIFF + 비교 패널
  - 시기 간 변화면(Δ)
  - 시기별 MaxEnt 서식지 적합도 + 니치 중첩도
  - 시기별 동위원소 산점도·시계열, 요약표

실행:
    .venv\\Scripts\\python.exe examples\\run_manchuria.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from archaeo_sdm.chronology import PERIOD_ORDER, assign_period, period_table
from archaeo_sdm.isotope_db import (filter_region, load_china_isotope_db,
                                    load_isomemo_db, quality_filter, site_summary)
from archaeo_sdm.paleoenv import worldclim_layers
from archaeo_sdm.plotting import plot_biplot_by_period, plot_isotope_timeseries
from archaeo_sdm.temporal import run_temporal_study

# ------------------------------------------------------------------- 입력
from archaeo_sdm import paths

CHINA_DB = paths.CHINA_DB
ISOMEMO_DB = paths.ISOMEMO_DB
BIO_DIR = paths.WORLDCLIM_DIR

OUT = paths.out("manchuria")

# 연구 범위: 북중국 ~ 만주 (동경 110~136, 북위 35~52)
BBOX = (110, 35, 136, 52)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    # 1) 두 데이터베이스를 표준 표로 합치고, 지역·품질·시기로 거르기
    df = pd.concat([load_china_isotope_db(CHINA_DB), load_isomemo_db(ISOMEMO_DB)],
                   ignore_index=True)
    df = quality_filter(filter_region(df, bbox=BBOX))
    df["period"] = assign_period(df)
    df = df[df["period"].notna()].copy()
    print(f"[data] 분석 대상 {len(df)}개 시료 / 유적 {df['site'].nunique()}곳")
    print(period_table().to_string(index=False))

    df.to_csv(os.path.join(OUT, "harmonized_dataset.csv"), index=False, encoding="utf-8-sig")

    # 2) 요약표와 기본 그림
    summ = site_summary(df, by=("period", "taxon_group"))
    summ.to_csv(os.path.join(OUT, "summary_by_period_taxon.csv"), index=False,
                encoding="utf-8-sig")
    print(summ.to_string(index=False))

    plot_biplot_by_period(df[df.taxon_group == "사람"], os.path.join(OUT, "biplot_human.png"),
                          PERIOD_ORDER, title="사람 — 시기별 δ13C–δ15N 분포")
    plot_biplot_by_period(df[df.taxon_group == "돼지"], os.path.join(OUT, "biplot_pig.png"),
                          PERIOD_ORDER, title="돼지 — 시기별 δ13C–δ15N 분포")
    for col, err, lab, fn in [("d13C_평균", "d13C_표준편차", "δ13C (‰)", "timeseries_d13C.png"),
                              ("d15N_평균", "d15N_표준편차", "δ15N (‰)", "timeseries_d15N.png")]:
        plot_isotope_timeseries(summ, os.path.join(OUT, fn), PERIOD_ORDER,
                                value_col=col, err_col=err, ylabel=lab,
                                title=f"시기별 {lab} 변화 (분류군별)")

    # 3) 시기별 아이소스케이프 + MaxEnt (고환경 결합)
    env = worldclim_layers(BIO_DIR, bio_numbers=(1, 4, 12, 15))
    print(f"[env] 고환경 레이어: {list(env)}")

    res = run_temporal_study(
        df, env, OUT, bbox=BBOX,
        proxies=("d13C_coll", "d15N_coll", "d13C_ap", "d18O_ap"),
        iso_taxa=("사람", "돼지", "소", "양·염소"),
        sdm_taxa=("돼지", "사람", "소"),
        couple_isoscape_to_sdm=True,
        min_sites_iso=3, min_sites_sdm=4,
        max_distance_km=300, beta_multiplier=3.0,
    )
    print("\n완료:", OUT)
