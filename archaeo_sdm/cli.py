"""
명령줄 실행기 (JSON 설정 파일 하나로 전체 분석 실행).

    python -m archaeo_sdm.cli run config.json
    python -m archaeo_sdm.cli template > config.json     # 설정 파일 뼈대 만들기
    python -m archaeo_sdm.cli download-paleo mid ./data/paleoclim

QGIS 플러그인과 브라우저 UI도 내부적으로 이 실행기를 호출합니다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

TEMPLATE = {
    "출력폴더": "outputs/my_study",
    "자료": {
        "china_db": "",
        "isomemo_db": "",
        "ecology_xlsx": "",
        "point_shapefile": "",
        "point_csv": "",
        "gazetteer_manual": ""
    },
    "고환경": {
        "worldclim_dir": "",
        "paleoclim_dir": "",
        "bio": [1, 4, 12, 15]
    },
    "범위": {"bbox": [105, 33, 141, 56]},
    "분석": {
        "프록시": ["d13C_coll", "d15N_coll"],
        "아이소스케이프_분류군": ["사람", "돼지"],
        "SDM_분류군": ["돼지"],
        "최소유적수_아이소스케이프": 3,
        "최소유적수_SDM": 4,
        "지지거리_km": 350,
        "beta_multiplier": 3.0,
        "공간블록교차검증": True
    }
}


def _load_all(cfg: dict):
    import numpy as np
    import pandas as pd

    from .chronology import assign_period, infer_period_from_site
    from .gazetteer import (gazetteer_from_frames, gazetteer_from_shapefile,
                            load_gazetteer, merge_gazetteers, resolve_coordinates)
    from .isotope_db import filter_region, load_china_isotope_db, load_isomemo_db, quality_filter
    from .local_data import load_ecology_table, load_point_isotopes
    from .regional import assign_zone

    d = cfg.get("자료", {})
    frames, gaz_parts = [], []
    china = isomemo = None
    if d.get("china_db"):
        china = load_china_isotope_db(d["china_db"])
        frames.append(china)
    if d.get("isomemo_db"):
        isomemo = load_isomemo_db(d["isomemo_db"])
        frames.append(isomemo)
    if d.get("point_shapefile"):
        frames.append(load_point_isotopes(d["point_shapefile"], d.get("point_csv") or None))
        gaz_parts.append(gazetteer_from_shapefile(d["point_shapefile"], "유적명"))
    if d.get("gazetteer_manual"):
        gaz_parts.append(load_gazetteer(d["gazetteer_manual"]))
    for f in (china, isomemo):
        if f is not None:
            gaz_parts.append(gazetteer_from_frames([f], "name_match"))

    if d.get("ecology_xlsx"):
        eco = load_ecology_table(d["ecology_xlsx"])
        gaz = merge_gazetteers(*gaz_parts) if gaz_parts else None
        if gaz is not None:
            eco, unresolved = resolve_coordinates(eco, gaz)
            cfg["_미해결유적"] = unresolved.to_dict("records")
        frames.append(eco)

    if not frames:
        raise SystemExit("자료가 하나도 지정되지 않았습니다. config의 '자료' 항목을 확인하세요.")

    df = pd.concat(frames, ignore_index=True)
    df = quality_filter(filter_region(df, bbox=tuple(cfg["범위"]["bbox"])))
    df["period"] = assign_period(df)
    # 연대가 없는 행은 같은 유적의 다른 시료에서 시기를 물려받습니다
    df["period"], inherited = infer_period_from_site(df)
    df["period_source"] = np.where(inherited, "유적상속", "자료")
    cfg["_시기상속_행수"] = int(inherited.sum())
    df = df[df["period"].notna()].copy()
    df["zone"] = assign_zone(df)
    return df


def run(config_path: str, progress=print) -> dict:
    """설정 파일대로 분석을 수행하고 결과 요약을 돌려줍니다."""
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    return run_config(cfg, progress)


def run_config(cfg: dict, progress=print) -> dict:
    import pandas as pd

    from .chronology import MANCHURIA_PERIODS, PERIOD_ORDER
    from .paleoenv import climate_for_period, worldclim_layers
    from .plotting import plot_biplot_by_period, plot_site_map, plot_zone_trajectories
    from .raster import RasterStack
    from .regional import within_zone_change, zone_period_table
    from .temporal import temporal_isoscapes, temporal_sdm
    from .validation import null_model_auc, spatial_block_cv

    out_dir = cfg["출력폴더"]
    os.makedirs(out_dir, exist_ok=True)
    an = cfg.get("분석", {})
    env = cfg.get("고환경", {})
    bios = tuple(env.get("bio", [1, 4, 12, 15]))
    bbox = tuple(cfg["범위"]["bbox"])

    progress("자료를 읽는 중...")
    df = _load_all(cfg)
    df.to_csv(os.path.join(out_dir, "dataset.csv"), index=False, encoding="utf-8-sig")
    progress(f"시료 {len(df)}개 / 유적 {df['site'].nunique()}곳")

    progress("고환경 레이어 준비 중...")
    grid_ref = worldclim_layers(env["worldclim_dir"], bios)[f"bio{bios[0]}"]
    stacks, used = {}, {}
    for p in MANCHURIA_PERIODS:
        layers, scales, label = climate_for_period(
            p, env.get("paleoclim_dir") or None, env["worldclim_dir"], bios)
        names = list(layers)
        stacks[p.key] = RasterStack.from_files([layers[n] for n in names], names,
                                               bounds=bbox, scales=scales, grid_from=grid_ref)
        used[p.key] = label
        progress(f"  {p.key}: {label}")
    ref = stacks[PERIOD_ORDER[0]]

    plot_site_map(df, os.path.join(out_dir, "map_sites_by_zone.png"), ref)

    progress("같은 소지역 안에서의 시기 비교...")
    changes = []
    for taxon in an.get("아이소스케이프_분류군", ["사람"]):
        sub = df[df["taxon_group"] == taxon]
        tbl = zone_period_table(sub)
        if tbl.empty:
            continue
        tbl.to_csv(os.path.join(out_dir, f"zone_period_{taxon}.csv"),
                   index=False, encoding="utf-8-sig")
        for col, lab in (("d13C_coll", "δ13C"), ("d15N_coll", "δ15N")):
            plot_zone_trajectories(tbl, os.path.join(out_dir, f"zone_traj_{taxon}_{col}.png"),
                                   PERIOD_ORDER, value_col=f"{col}_평균",
                                   err_col=f"{col}_표준편차", n_col=f"{col}_n",
                                   ylabel=f"{lab} (‰)",
                                   title=f"{taxon} — 같은 소지역 안에서의 {lab} 변화")
            c = within_zone_change(sub, col, PERIOD_ORDER)
            if len(c):
                c.insert(0, "분류군", taxon)
                changes.append(c)
    if changes:
        pd.concat(changes, ignore_index=True).to_csv(
            os.path.join(out_dir, "zone_changes.csv"), index=False, encoding="utf-8-sig")

    progress("아이소스케이프 생성 중...")
    iso = temporal_isoscapes(df, ref, out_dir,
                             proxies=tuple(an.get("프록시", ["d13C_coll", "d15N_coll"])),
                             taxa=tuple(an.get("아이소스케이프_분류군", ["사람"])),
                             min_sites=int(an.get("최소유적수_아이소스케이프", 3)),
                             max_distance_km=float(an.get("지지거리_km", 350)),
                             verbose=False)
    progress(f"  아이소스케이프 {len(iso)}종 생성")

    progress("시기별 MaxEnt 학습 중...")
    sdm = temporal_sdm(df, ref, out_dir, taxa=tuple(an.get("SDM_분류군", ["돼지"])),
                       stacks=stacks, isoscape_layers=iso,
                       min_sites=int(an.get("최소유적수_SDM", 4)),
                       beta_multiplier=float(an.get("beta_multiplier", 3.0)),
                       verbose=False)

    cv_rows = []
    if an.get("공간블록교차검증", True):
        progress("공간 블록 교차검증...")
        for taxon in an.get("SDM_분류군", ["돼지"]):
            for p in MANCHURIA_PERIODS:
                sub = df[(df.taxon_group == taxon) & (df.period == p.key)]
                sites = sub[["site", "lat", "lon"]].dropna().drop_duplicates()
                if len(sites) < 8:
                    continue
                s = stacks[p.key]
                x, y = s.to_crs_xy(sites["lon"].to_numpy(), sites["lat"].to_numpy())
                Xp, keep = s.extract(x, y)
                if len(Xp) < 8:
                    continue
                Xb, bx, by = s.sample_background(5000, 0)
                r = spatial_block_cv(Xp, (x[keep], y[keep]), Xb, (bx, by),
                                     var_names=s.names, n_folds=4)
                cv_rows.append({"분류군": taxon, "period": p.key, "유적수": len(Xp),
                                "공간CV_AUC": r["AUC_평균"], "공간CV_TSS": r["TSS_평균"],
                                "무작위기준선": round(null_model_auc(len(Xp), len(Xb)), 3)})
        if cv_rows:
            pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, "spatial_block_cv.csv"),
                                         index=False, encoding="utf-8-sig")

    for taxon in ("사람", "돼지"):
        sub = df[df.taxon_group == taxon]
        if len(sub) > 5:
            plot_biplot_by_period(sub, os.path.join(out_dir, f"biplot_{taxon}.png"),
                                  PERIOD_ORDER, title=f"{taxon} — 시기별 δ13C–δ15N 분포")

    summary = {
        "시료수": int(len(df)), "유적수": int(df["site"].nunique()),
        "시기별_기후": used,
        "아이소스케이프": [f"{t}/{p}" for (t, p) in iso.keys()],
        "SDM": list(sdm.keys()),
        "공간CV": cv_rows,
        "미해결유적": cfg.get("_미해결유적", []),
        "출력폴더": os.path.abspath(out_dir),
        "파일": sorted(os.listdir(out_dir)),
    }
    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    progress("완료")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="archaeo_sdm", description="고고학 종분포모델 실행기")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="설정 파일대로 분석 실행")
    r.add_argument("config")

    sub.add_parser("template", help="설정 파일 뼈대를 표준출력으로")

    d = sub.add_parser("download-paleo", help="고기후 자료 내려받기")
    d.add_argument("slice", choices=["mid", "lgm"])
    d.add_argument("out_dir")

    w = sub.add_parser("web", help="브라우저 UI 실행")
    w.add_argument("--port", type=int, default=8765)

    st = sub.add_parser("studio", help="SDM Studio (다중 모델·시나리오·니치) 실행")
    st.add_argument("--port", type=int, default=8766)

    a = ap.parse_args(argv)
    if a.cmd == "studio":
        from .studio.app import serve as serve_studio
        serve_studio(port=a.port)
        return 0
    if a.cmd == "template":
        json.dump(TEMPLATE, sys.stdout, ensure_ascii=False, indent=2)
        return 0
    if a.cmd == "run":
        s = run(a.config)
        print(json.dumps({k: v for k, v in s.items() if k != "파일"},
                         ensure_ascii=False, indent=2))
        return 0
    if a.cmd == "download-paleo":
        from .paleoenv import download_worldclim_paleo
        print(download_worldclim_paleo(a.slice, a.out_dir, confirm=True))
        return 0
    if a.cmd == "web":
        from .webapp.app import serve
        serve(port=a.port)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
