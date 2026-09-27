# -*- coding: utf-8 -*-
"""
동물유체 전용 분석 — 만주·요서·연해주·아무르 (사람 제외).

두 가지 축으로 결과를 냅니다.

  A. 통합(integrated) : 위 네 권역을 하나로 묶어 분석
  B. 지역별(regional) : 요서 / 만주 / 연해주·아무르 를 각각 독립적으로 분석
                        (배경점도 그 지역 안에서만 뽑으므로 통합본과 직접 비교됩니다)

각 축에서 다음을 수행합니다.
  1) 시기 x 분류군 MaxEnt            (표본이 되는 칸만)
  2) 분류군별 시기통합 MaxEnt        (시기로 쪼개면 표본이 모자랄 때의 대안 층위)
  3) 탄소·질소 아이소스케이프와 시기간 변화면
  4) 같은 지역 안에서의 시기 비교(통계)
  5) 공간 블록 교차검증
  6) 통합본 vs 지역본 비교표 + 니치 중첩도(Schoener's D)

실행:
    .venv\\Scripts\\python.exe examples\\run_animals_regional.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from archaeo_sdm import paths
from archaeo_sdm.chronology import MANCHURIA_PERIODS, PERIOD_ORDER
from archaeo_sdm.cli import _load_all
from archaeo_sdm.isoscape import schoener_d
from archaeo_sdm.maxent import MaxentModel
from archaeo_sdm.paleoenv import climate_for_period, worldclim_layers
from archaeo_sdm.plotting import (plot_panel_maps, plot_site_map, plot_zone_trajectories)
from archaeo_sdm.raster import RasterStack
from archaeo_sdm.regional import within_zone_change, zone_period_stats, zone_period_table
from archaeo_sdm.temporal import temporal_isoscapes, temporal_sdm
from archaeo_sdm.validation import null_model_auc, spatial_block_cv

BIOS = (1, 4, 12, 15)

# 분석 대상 권역 (화북·몽골/자바이칼 제외)
TARGET_ZONES = ["요서·내몽골동부", "요동·요동반도", "길림·흑룡강", "연해주", "아무르·하바롭스크"]

# 지역별 독립 분석 단위
SCOPES = {
    "통합": TARGET_ZONES,
    "요서": ["요서·내몽골동부"],
    "만주": ["요동·요동반도", "길림·흑룡강"],
    "연해주·아무르": ["연해주", "아무르·하바롭스크"],
}

ANIMAL_TAXA = ["돼지", "개", "소", "양·염소", "말", "사슴", "전체동물"]
BUFFER_DEG = 2.5          # 지역 bbox 여유 (배경점을 뽑을 공간 확보)
MIN_SITES_SDM = 4
MIN_SITES_ISO = 3
MIN_SITES_CV = 8


def load_animals() -> pd.DataFrame:
    """표준 표를 만들고 동물만 남긴 뒤, 대상 권역으로 자릅니다."""
    cfg = {
        "출력폴더": paths.out("animals_regional"),
        "자료": {"china_db": paths.CHINA_DB, "isomemo_db": paths.ISOMEMO_DB,
                 "ecology_xlsx": paths.ECOLOGY_XLSX,
                 "point_shapefile": paths.PRIMORYE_SHP, "point_csv": paths.PRIMORYE_CSV,
                 "gazetteer_manual": paths.GAZETTEER_MANUAL},
        "범위": {"bbox": [105, 33, 145, 56]},
    }
    df = _load_all(cfg)
    df = df[(df["group"] == "animal") & (df["taxon_group"] != "사람")]
    df = df[df["zone"].isin(TARGET_ZONES)].copy()

    # '전체동물' — 분류군을 쪼개면 표본이 모자랄 때 쓰는 통합 분류군
    allan = df.copy()
    allan["taxon_group"] = "전체동물"
    return pd.concat([df, allan], ignore_index=True)


def bbox_for(df: pd.DataFrame, buffer=BUFFER_DEG):
    lon, lat = df["lon"].dropna(), df["lat"].dropna()
    return (float(lon.min() - buffer), float(lat.min() - buffer),
            float(lon.max() + buffer), float(lat.max() + buffer))


def period_stacks(bbox):
    grid_ref = worldclim_layers(paths.WORLDCLIM_DIR, BIOS)[f"bio{BIOS[0]}"]
    stacks = {}
    for p in MANCHURIA_PERIODS:
        layers, scales, _ = climate_for_period(p, paths.PALEOCLIM_DIR,
                                               paths.WORLDCLIM_DIR, BIOS)
        names = list(layers)
        stacks[p.key] = RasterStack.from_files([layers[n] for n in names], names,
                                               bounds=bbox, scales=scales, grid_from=grid_ref)
    return stacks


def pooled_sdm(df, stack, out_dir, taxa, scope_name, min_sites=MIN_SITES_SDM):
    """시기를 합쳐 분류군별로 한 장씩 만드는 MaxEnt (시기별로는 표본이 모자랄 때)."""
    surfaces, rows = {}, []
    for taxon in taxa:
        sites = df[df.taxon_group == taxon][["site", "lat", "lon"]].dropna().drop_duplicates()
        if len(sites) < min_sites:
            rows.append({"범위": scope_name, "분류군": taxon, "유적수": len(sites),
                         "결과": f"유적 부족(최소 {min_sites})"})
            continue
        x, y = stack.to_crs_xy(sites["lon"].to_numpy(), sites["lat"].to_numpy())
        Xp, keep = stack.extract(x, y)
        idx = stack.thin_by_cell(x[keep], y[keep])
        keep, Xp = keep[idx], Xp[idx]
        if len(Xp) < min_sites:
            rows.append({"범위": scope_name, "분류군": taxon, "유적수": int(len(Xp)),
                         "결과": "격자 정리 후 부족"})
            continue
        Xb, bx, by = stack.sample_background(5000, 0)
        m = MaxentModel(("linear", "quadratic", "hinge"), beta_multiplier=3.0)
        m.fit(Xp, Xb, var_names=stack.names)
        surf = stack.predict_surface(m, "cloglog")
        surfaces[taxon] = surf
        stack.write(os.path.join(out_dir, f"sdm_pooled_{taxon.replace('·', '_')}.tif"), surf)
        imp = m.permutation_importance(Xp, Xb, n_repeats=3)
        top = max(imp, key=imp.get) if imp else "-"

        cv = auc_cv = tss_cv = null = np.nan
        if len(Xp) >= MIN_SITES_CV:
            cv = spatial_block_cv(Xp, (x[keep], y[keep]), Xb, (bx, by),
                                  var_names=stack.names, n_folds=4, beta_multiplier=3.0)
            auc_cv, tss_cv = cv["AUC_평균"], cv["TSS_평균"]
            null = round(null_model_auc(len(Xp), len(Xb)), 3)
        rows.append({"범위": scope_name, "분류군": taxon, "유적수": int(len(Xp)),
                     "AUC": round(m.auc(Xp, Xb), 3), "최대기여변수": top,
                     "기여도(%)": round(imp.get(top, 0), 1),
                     "공간CV_AUC": auc_cv, "공간CV_TSS": tss_cv, "무작위기준선": null,
                     "결과": "성공"})
    if surfaces:
        plot_panel_maps(surfaces, stack,
                        os.path.join(out_dir, "panel_sdm_pooled.png"),
                        suptitle=f"[{scope_name}] 분류군별 서식지 적합도 (전 시기 통합)",
                        cmap="viridis", vmin=0, vmax=1, cbar_label="적합도 (cloglog)",
                        basemap=stack.data[0])
    return surfaces, pd.DataFrame(rows)


def run_scope(df_all, scope_name, zones, root_out):
    """한 범위(통합 또는 지역)에 대한 전체 분석."""
    out_dir = os.path.join(root_out, scope_name.replace("·", "_"))
    os.makedirs(out_dir, exist_ok=True)
    df = df_all[df_all["zone"].isin(zones)].copy()
    real = df[df.taxon_group != "전체동물"]
    if real.empty:
        print(f"\n### [{scope_name}] 자료 없음 — 건너뜁니다")
        return None

    print(f"\n{'=' * 70}\n### [{scope_name}]  시료 {len(real)}개 / 유적 {real.site.nunique()}곳"
          f" / 권역 {sorted(set(real.zone))}\n{'=' * 70}")

    bbox = bbox_for(real)
    stacks = period_stacks(bbox)
    ref = stacks[PERIOD_ORDER[0]]
    print(f"  분석 범위 bbox={tuple(round(v, 2) for v in bbox)}  격자={ref.shape}")

    plot_site_map(real, os.path.join(out_dir, "map_sites.png"), ref,
                  title=f"[{scope_name}] 동물유체 동위원소 유적 분포")

    # 1) 시기 x 분류군 MaxEnt
    print("  · 시기 x 분류군 MaxEnt")
    iso = temporal_isoscapes(df, ref, out_dir, proxies=("d13C_coll", "d15N_coll"),
                             taxa=tuple(ANIMAL_TAXA), min_sites=MIN_SITES_ISO,
                             max_distance_km=300, verbose=True)
    sdm = temporal_sdm(df, ref, out_dir, taxa=tuple(ANIMAL_TAXA), stacks=stacks,
                       isoscape_layers=iso, min_sites=MIN_SITES_SDM,
                       beta_multiplier=3.0, verbose=True)

    # 2) 분류군별 시기통합 MaxEnt
    print("  · 분류군별 시기통합 MaxEnt")
    pooled, pooled_tbl = pooled_sdm(df, ref, out_dir, ANIMAL_TAXA, scope_name)
    if len(pooled_tbl):
        pooled_tbl.to_csv(os.path.join(out_dir, "sdm_pooled_summary.csv"),
                          index=False, encoding="utf-8-sig")
        print(pooled_tbl.to_string(index=False))

    # 3) 같은 지역 안에서의 시기 비교
    changes = []
    for taxon in ("돼지", "개", "소", "전체동물"):
        sub = real[real.taxon_group == taxon] if taxon != "전체동물" else real
        if sub.empty:
            continue
        tbl = zone_period_table(sub, min_n=2)
        if tbl.empty:
            continue
        tbl.insert(0, "분류군", taxon)
        tbl.to_csv(os.path.join(out_dir, f"zone_period_{taxon.replace('·', '_')}.csv"),
                   index=False, encoding="utf-8-sig")
        for col, lab in (("d13C_coll", "δ13C"), ("d15N_coll", "δ15N")):
            plot_zone_trajectories(
                tbl, os.path.join(out_dir, f"zone_traj_{taxon.replace('·', '_')}_{col}.png"),
                PERIOD_ORDER, value_col=f"{col}_평균", err_col=f"{col}_표준편차",
                n_col=f"{col}_n", ylabel=f"{lab} (‰)", min_n=2,
                title=f"[{scope_name}] {taxon} — 소지역별 {lab} 변화")
            c = within_zone_change(sub, col, PERIOD_ORDER, min_n=2)
            if len(c):
                c.insert(0, "분류군", taxon)
                c.insert(0, "범위", scope_name)
                changes.append(c)
        st = zone_period_stats(sub, "d13C_coll", min_n=3)
        if len(st):
            st.to_csv(os.path.join(out_dir, f"zone_stats_{taxon.replace('·', '_')}.csv"),
                      index=False, encoding="utf-8-sig")
    change_df = pd.concat(changes, ignore_index=True) if changes else pd.DataFrame()
    if len(change_df):
        change_df.to_csv(os.path.join(out_dir, "zone_changes.csv"),
                         index=False, encoding="utf-8-sig")

    return {"scope": scope_name, "out_dir": out_dir, "df": real, "stack": ref,
            "bbox": bbox, "iso": iso, "sdm": sdm, "pooled": pooled,
            "pooled_table": pooled_tbl, "changes": change_df}


if __name__ == "__main__":
    ROOT = paths.out("animals_regional")
    df_all = load_animals()
    real = df_all[df_all.taxon_group != "전체동물"]
    real.to_csv(os.path.join(ROOT, "dataset_animals.csv"), index=False, encoding="utf-8-sig")

    print(f"[자료] 동물 시료 {len(real)}개 / 유적 {real.site.nunique()}곳")
    print("\n== 권역 x 분류군 (유적수) ==")
    print(real.groupby(["zone", "taxon_group"]).site.nunique().unstack(fill_value=0).to_string())
    print("\n== 분류군 x 시기 (유적수) ==")
    print(real.groupby(["taxon_group", "period"]).site.nunique()
          .unstack(fill_value=0).reindex(columns=PERIOD_ORDER, fill_value=0).to_string())

    results = {}
    for scope, zones in SCOPES.items():
        r = run_scope(df_all, scope, zones, ROOT)
        if r:
            results[scope] = r

    # ------------------------------------------------- 통합본 vs 지역본 비교
    print(f"\n{'=' * 70}\n### 통합본 vs 지역본 비교\n{'=' * 70}")
    tables = [r["pooled_table"] for r in results.values() if len(r["pooled_table"])]
    if tables:
        comp = pd.concat(tables, ignore_index=True)
        comp.to_csv(os.path.join(ROOT, "compare_integrated_vs_regional.csv"),
                    index=False, encoding="utf-8-sig")
        print(comp.to_string(index=False))

    # 같은 분류군에 대해 통합 예측면과 지역 예측면이 얼마나 닮았는지
    rows = []
    if "통합" in results:
        base = results["통합"]
        for scope, r in results.items():
            if scope == "통합":
                continue
            for taxon, surf in r["pooled"].items():
                if taxon not in base["pooled"]:
                    continue
                # 지역 범위에 맞춰 통합 예측면을 잘라 비교
                sub_bbox = r["bbox"]
                b_stack, r_stack = base["stack"], r["stack"]
                import rasterio
                win = rasterio.windows.from_bounds(*sub_bbox, transform=b_stack.transform)
                win = win.round_offsets().round_lengths().intersection(
                    rasterio.windows.Window(0, 0, b_stack.shape[1], b_stack.shape[0]))
                cut = base["pooled"][taxon][int(win.row_off):int(win.row_off + win.height),
                                            int(win.col_off):int(win.col_off + win.width)]
                h = min(cut.shape[0], surf.shape[0])
                w = min(cut.shape[1], surf.shape[1])
                d = schoener_d(cut[:h, :w], surf[:h, :w])
                rows.append({"지역": scope, "분류군": taxon,
                             "통합-지역 니치중첩도(D)": round(d, 3)})
    if rows:
        ov = pd.DataFrame(rows)
        ov.to_csv(os.path.join(ROOT, "overlap_integrated_vs_regional.csv"),
                  index=False, encoding="utf-8-sig")
        print("\n== 통합 예측면 vs 지역 예측면 니치 중첩도 ==")
        print(ov.to_string(index=False))

    all_changes = [r["changes"] for r in results.values() if len(r["changes"])]
    if all_changes:
        pd.concat(all_changes, ignore_index=True).to_csv(
            os.path.join(ROOT, "all_zone_changes.csv"), index=False, encoding="utf-8-sig")

    print(f"\n완료: {ROOT}")
