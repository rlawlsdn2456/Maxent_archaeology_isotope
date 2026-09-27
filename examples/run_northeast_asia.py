# -*- coding: utf-8 -*-
"""
동북아시아 통합 분석 — 북중국·만주·연해주·아무르·자바이칼, 청동기~요금.

앞선 run_manchuria.py에서 다음이 추가되었습니다.
  1) 자료 확대   : 연해주·아무르·자바이칼 자료 결합 + 유적 좌표 사전(gazetteer)
  2) 고기후 교체 : 4.2 ka 이전 시기는 중기 홀로세(6 ka) 고기후로 자동 교체
  3) 같은 지역 비교 : 소지역(zone) 안에서만 시기 변화를 비교 (공간 교락 제거)
  4) 공간 블록 교차검증 : 무작위 CV의 성능 부풀림을 피한 정직한 평가
  5) 탄소·질소 아이소스케이프를 각각 별도 지도로 생성

실행:
    .venv\\Scripts\\python.exe examples\\run_northeast_asia.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from archaeo_sdm.chronology import MANCHURIA_PERIODS, PERIOD_ORDER, assign_period, period_table
from archaeo_sdm.gazetteer import (gazetteer_from_frames, gazetteer_from_shapefile,
                                   load_gazetteer, merge_gazetteers, resolve_coordinates,
                                   save_gazetteer)
from archaeo_sdm.isotope_db import (filter_region, load_china_isotope_db, load_isomemo_db,
                                    quality_filter, site_summary)
from archaeo_sdm.local_data import load_ecology_table, load_point_isotopes
from archaeo_sdm.paleoenv import climate_for_period
from archaeo_sdm.plotting import (plot_biplot_by_period, plot_isotope_timeseries,
                                  plot_site_map, plot_zone_trajectories)
from archaeo_sdm.raster import RasterStack
from archaeo_sdm.regional import (DEFAULT_ZONES, assign_zone, continuous_sequences,
                                  within_zone_change, zone_period_stats, zone_period_table)
from archaeo_sdm.temporal import temporal_isoscapes, temporal_sdm
from archaeo_sdm.validation import null_model_auc, spatial_block_cv

# --------------------------------------------------------------------- 입력
# 경로는 archaeo_sdm.paths 가 알아서 찾습니다(내장 드라이브 사본 우선, 없으면 외장하드).
from archaeo_sdm import paths

CHINA_DB = paths.CHINA_DB
ISOMEMO_DB = paths.ISOMEMO_DB
ECOLOGY_XLSX = paths.ECOLOGY_XLSX
PRIMORYE_SHP = paths.PRIMORYE_SHP
PRIMORYE_CSV = paths.PRIMORYE_CSV
BIO_DIR = paths.WORLDCLIM_DIR
PALEO_DIR = paths.PALEOCLIM_DIR
GAZ_MANUAL = paths.GAZETTEER_MANUAL

OUT = paths.out("northeast_asia")
BBOX = (105, 33, 141, 56)          # 화북 ~ 아무르·자바이칼
BIOS = (1, 4, 12, 15)


def build_dataset():
    """네 가지 자료를 합치고 좌표·시기·소지역을 붙입니다."""
    china = load_china_isotope_db(CHINA_DB)
    isomemo = load_isomemo_db(ISOMEMO_DB)
    eco = load_ecology_table(ECOLOGY_XLSX)
    primorye = load_point_isotopes(PRIMORYE_SHP, PRIMORYE_CSV)

    # 좌표 사전: 연해주 SHP > 직접 입력 > 중국DB/IsoMemo 이름 대조
    gaz = merge_gazetteers(
        gazetteer_from_shapefile(PRIMORYE_SHP, "유적명"),
        load_gazetteer(GAZ_MANUAL),
        gazetteer_from_frames([china], "name_match"),
        gazetteer_from_frames([isomemo], "name_match"),
    )
    save_gazetteer(gaz, os.path.join(OUT, "gazetteer_used.csv"))

    eco_fixed, unresolved = resolve_coordinates(eco, gaz)
    if len(unresolved):
        tmpl = unresolved.copy()
        tmpl["lat"] = ""
        tmpl["lon"] = ""
        tmpl.to_csv(os.path.join(OUT, "gazetteer_TODO.csv"), index=False, encoding="utf-8-sig")
        print(f"[좌표] 미해결 유적 {len(unresolved)}곳 -> gazetteer_TODO.csv 에 목록 저장")
        print(f"       lat/lon을 채워 {GAZ_MANUAL} 로 저장하면 다음 실행부터 반영됩니다.")

    df = pd.concat([china, isomemo, eco_fixed, primorye], ignore_index=True)
    df = quality_filter(filter_region(df, bbox=BBOX))
    df["period"] = assign_period(df)
    df = df[df["period"].notna()].copy()
    df["zone"] = assign_zone(df)
    return df


def build_period_stacks(periods, bbox, bios=BIOS):
    """시기마다 알맞은 기후 자료로 스택을 만듭니다(격자는 모두 동일)."""
    from archaeo_sdm.paleoenv import worldclim_layers

    grid_ref = worldclim_layers(BIO_DIR, bios)[f"bio{bios[0]}"]
    stacks, used = {}, {}
    for p in periods:
        layers, scales, label = climate_for_period(p, PALEO_DIR, BIO_DIR, bios)
        names = list(layers)
        stacks[p.key] = RasterStack.from_files(
            [layers[n] for n in names], names, bounds=bbox, scales=scales, grid_from=grid_ref)
        used[p.key] = label
    return stacks, used


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    # ------------------------------------------------------------ 1) 자료
    df = build_dataset()
    df.to_csv(os.path.join(OUT, "dataset_northeast_asia.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[1] 통합 자료: {len(df)}개 시료 / 유적 {df['site'].nunique()}곳")
    print(df.groupby("source").size().to_string())
    print("\n소지역 x 시기 시료수:")
    print(pd.crosstab(df["zone"], df["period"]).reindex(columns=PERIOD_ORDER, fill_value=0).to_string())

    # ------------------------------------------------- 2) 시기별 기후 스택
    stacks, used_clim = build_period_stacks(MANCHURIA_PERIODS, BBOX)
    print("\n[2] 시기별 고기후 자료:")
    for k, v in used_clim.items():
        print(f"    {k}: {v}")
    ref_stack = stacks[PERIOD_ORDER[0]]

    plot_site_map(df, os.path.join(OUT, "map_sites_by_zone.png"), ref_stack,
                  title="자료가 있는 유적의 분포 (소지역별)")

    # --------------------------------------- 3) 같은 소지역 안에서의 시기 비교
    print("\n[3] 같은 소지역 안에서의 시기 비교")
    for taxon, tag in (("사람", "human"), ("돼지", "pig")):
        sub = df[df["taxon_group"] == taxon]
        tbl = zone_period_table(sub)
        if tbl.empty:
            continue
        tbl.to_csv(os.path.join(OUT, f"zone_period_{tag}.csv"), index=False, encoding="utf-8-sig")
        runs = continuous_sequences(tbl, PERIOD_ORDER)
        print(f"  · {taxon}: 시기가 이어지는 소지역 -> "
              + (", ".join(f"{z}({'→'.join(r[0])})" for z, r in runs.items()) or "없음"))
        for col, lab in (("d13C_coll", "δ13C"), ("d15N_coll", "δ15N")):
            plot_zone_trajectories(
                tbl, os.path.join(OUT, f"zone_traj_{tag}_{col}.png"), PERIOD_ORDER,
                value_col=f"{col}_평균", err_col=f"{col}_표준편차", n_col=f"{col}_n",
                ylabel=f"{lab} (‰)", title=f"{taxon} — 같은 소지역 안에서의 {lab} 변화")
            chg = within_zone_change(sub, col, PERIOD_ORDER)
            if len(chg):
                chg.to_csv(os.path.join(OUT, f"zone_change_{tag}_{col}.csv"),
                           index=False, encoding="utf-8-sig")
                print(chg.to_string(index=False))
            st = zone_period_stats(sub, col)
            if len(st):
                st.to_csv(os.path.join(OUT, f"zone_stats_{tag}_{col}.csv"),
                          index=False, encoding="utf-8-sig")

    # ------------------------------- 4) 탄소·질소 아이소스케이프 (각각 별도)
    print("\n[4] 아이소스케이프 (탄소·질소 각각)")
    iso = temporal_isoscapes(df, ref_stack, OUT,
                             proxies=("d13C_coll", "d15N_coll", "d13C_ap", "d18O_ap"),
                             taxa=("사람", "돼지", "소", "양·염소", "개"),
                             min_sites=3, max_distance_km=350)

    # ------------------------------------------- 5) 시기별 SDM (고기후 교체)
    print("\n[5] 시기별 MaxEnt (시기마다 다른 고기후)")
    sdm = temporal_sdm(df, ref_stack, OUT, taxa=("사람", "돼지", "소"),
                       stacks=stacks, isoscape_layers=iso, min_sites=4,
                       beta_multiplier=3.0)

    # ------------------------------------------- 6) 공간 블록 교차검증
    print("\n[6] 공간 블록 교차검증")
    rows = []
    for taxon in ("사람", "돼지", "소"):
        for p in MANCHURIA_PERIODS:
            sub = df[(df["taxon_group"] == taxon) & (df["period"] == p.key)]
            sites = sub[["site", "lat", "lon"]].dropna().drop_duplicates()
            if len(sites) < 8:            # 블록을 나누려면 최소한의 유적 수가 필요
                continue
            s = stacks[p.key]
            x, y = s.to_crs_xy(sites["lon"].to_numpy(), sites["lat"].to_numpy())
            Xp, keep = s.extract(x, y)
            if len(Xp) < 8:
                continue
            Xb, bx, by = s.sample_background(5000, 0)
            res = spatial_block_cv(Xp, (x[keep], y[keep]), Xb, (bx, by),
                                   var_names=s.names, n_folds=4, method="kmeans",
                                   beta_multiplier=3.0)
            null = null_model_auc(len(Xp), len(Xb))
            rows.append({"분류군": taxon, "period": p.key, "유적수": len(Xp),
                         "공간CV_AUC": res["AUC_평균"], "AUC_표준편차": res["AUC_표준편차"],
                         "공간CV_TSS": res["TSS_평균"], "무작위기준선_AUC": round(null, 3),
                         "유효fold": res["유효_fold수"]})
            print(f"    {taxon}/{p.key}: 유적 {len(Xp)}곳, 공간CV AUC={res['AUC_평균']}, "
                  f"TSS={res['TSS_평균']} (무작위 기준선 {null:.3f})")
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "spatial_block_cv.csv"),
                                  index=False, encoding="utf-8-sig")

    # ------------------------------------------------------- 7) 기본 그림들
    summ = site_summary(df, by=("period", "taxon_group"))
    summ.to_csv(os.path.join(OUT, "summary_by_period_taxon.csv"), index=False,
                encoding="utf-8-sig")
    for col, err, lab, fn in [("d13C_평균", "d13C_표준편차", "δ13C (‰)", "timeseries_d13C.png"),
                              ("d15N_평균", "d15N_표준편차", "δ15N (‰)", "timeseries_d15N.png")]:
        plot_isotope_timeseries(summ, os.path.join(OUT, fn), PERIOD_ORDER,
                                value_col=col, err_col=err, ylabel=lab,
                                title=f"시기별 {lab} 변화 (전체·분류군별)")
    plot_biplot_by_period(df[df.taxon_group == "사람"], os.path.join(OUT, "biplot_human.png"),
                          PERIOD_ORDER, title="사람 — 시기별 δ13C–δ15N 분포")
    plot_biplot_by_period(df[df.taxon_group == "돼지"], os.path.join(OUT, "biplot_pig.png"),
                          PERIOD_ORDER, title="돼지 — 시기별 δ13C–δ15N 분포")

    print("\n완료:", OUT)
