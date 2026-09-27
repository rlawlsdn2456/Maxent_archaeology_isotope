"""
'데이터만 넣으면 돌아가는' 파이프라인.

run_sdm() 하나만 호출하면
  점 자료 읽기 -> 환경 레이어 정렬 -> (선택) 동위원소 면 생성 -> 배경점 추출
  -> MaxEnt 학습 -> 적합도 지도 GeoTIFF -> 그림/표 저장
까지 한 번에 처리합니다. 결과물은 QGIS에서 바로 열 수 있습니다.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .isotope import idw_surface
from .maxent import MaxentModel
from .occurrence import load_points
from .plotting import plot_importance, plot_isotope_biplot, plot_response_curves, plot_suitability
from .raster import RasterStack


def _read_table(path: str) -> pd.DataFrame:
    """동위원소 표(CSV/XLSX)를 한글 인코딩까지 고려해 읽습니다."""
    from .occurrence import _read_csv_any_encoding
    if os.path.splitext(path)[1].lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return _read_csv_any_encoding(path)


def run_sdm(
    occurrence_path: str,
    env_layers: dict,                  # {"레이어이름": "경로.tif", ...}
    out_dir: str,
    lon_col: str | None = None,
    lat_col: str | None = None,
    occurrence_crs: str = "EPSG:4326",  # csv/xlsx일 때의 좌표계
    isotope_cols: dict | None = None,   # {"d13C": "컬럼명", "d15N": "컬럼명"}
    isotope_table: str | None = None,   # 동위원소가 별도 CSV/XLSX에 있을 때 그 경로
    join_on: str | None = None,         # 점 자료와 동위원소 표를 이어붙일 공통 컬럼(예: "id")
    terrain_from: str | None = None,    # DEM 레이어 이름 -> 경사/기복 등 자동 생성
    terrain_include=("slope", "northness", "tri"),
    buffer_ratio: float = 0.25,         # 연구지역 범위 = 점 분포 범위 + 여유
    feature_types=("linear", "quadratic", "hinge"),
    beta_multiplier: float = 1.0,
    n_background: int = 10000,
    thin_points: bool = True,
    reference_layer: str | None = None,
    random_state: int = 0,
    verbose: bool = True,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    # ---------------------------------------------------------- 1) 점 자료
    df, px, py, pcrs = load_points(occurrence_path, lon_col, lat_col, occurrence_crs)
    log(f"[1] 출토지점 {len(px)}개 읽음  (좌표계: {pcrs})")

    # 동위원소 값이 별도 표에 있으면 공통 컬럼으로 붙입니다
    if isotope_table and join_on:
        iso_df = _read_table(isotope_table)
        df = df.merge(iso_df, on=join_on, how="left", suffixes=("", "_iso"))
        log(f"    동위원소 표 결합: {os.path.basename(isotope_table)} (기준 컬럼 '{join_on}')")

    # ------------------------------------------------- 2) 환경 레이어 정렬
    names = list(env_layers.keys())
    paths = [env_layers[n] for n in names]
    ref_idx = names.index(reference_layer) if reference_layer else 0

    # 점 범위 -> 기준 래스터 좌표계로 변환해서 자를 범위 계산
    import rasterio
    from pyproj import Transformer

    with rasterio.open(paths[ref_idx]) as ref:
        ref_crs = ref.crs
    if str(ref_crs) != str(pcrs):
        tf = Transformer.from_crs(pcrs, ref_crs, always_xy=True)
        rx, ry = tf.transform(px, py)
    else:
        rx, ry = px, py
    rx, ry = np.asarray(rx, float), np.asarray(ry, float)
    dx = max(rx.max() - rx.min(), 1e-9) * buffer_ratio
    dy = max(ry.max() - ry.min(), 1e-9) * buffer_ratio
    bounds = (rx.min() - dx, ry.min() - dy, rx.max() + dx, ry.max() + dy)

    stack = RasterStack.from_files(paths, names, reference=ref_idx, bounds=bounds)
    log(f"[2] 환경 레이어 {len(names)}개 정렬 완료  격자={stack.shape}  CRS={stack.crs}")

    if terrain_from:
        from .terrain import add_terrain_layers
        add_terrain_layers(stack, terrain_from, terrain_include)
        log(f"    지형 변수 추가: {list(terrain_include)}")

    # --------------------------------------------- 3) 동위원소 면(선택사항)
    sx, sy = stack.to_crs_xy(px, py, pcrs)
    iso_used = {}
    if isotope_cols:
        for label, col in isotope_cols.items():
            if col not in df.columns:
                log(f"    ! 동위원소 컬럼 '{col}' 없음 - 건너뜀")
                continue
            v = pd.to_numeric(df[col], errors="coerce").to_numpy()
            surf = idw_surface(sx, sy, v, like=stack)
            stack.add_layer(label, surf)
            stack.write(os.path.join(out_dir, f"isoscape_{label}.tif"), surf)
            iso_used[label] = v
            log(f"[3] 동위원소 면 생성: {label} (유효 시료 {np.isfinite(v).sum()}개)")

    # -------------------------------------------------- 4) 값 추출 / 배경점
    Xp, keep = stack.extract(sx, sy)
    if thin_points and len(keep) > 0:
        sub = stack.thin_by_cell(sx[keep], sy[keep])
        keep, Xp = keep[sub], Xp[sub]
    log(f"[4] 모델에 쓰인 유효 출토지점: {len(Xp)}개")
    if len(Xp) < 5:
        raise ValueError("유효 지점이 너무 적습니다. 좌표계/레이어 범위를 확인하세요.")

    Xb, bx, by = stack.sample_background(n_background, random_state)
    log(f"    배경점: {len(Xb)}개")

    # ------------------------------------------------------- 5) MaxEnt 학습
    model = MaxentModel(feature_types, beta_multiplier, random_state=random_state)
    model.fit(Xp, Xb, var_names=stack.names)
    auc = model.auc(Xp, Xb)
    log(f"[5] 학습 완료  AUC={auc:.3f}  (특징 {len(model.lambdas_)}개, "
        f"0이 아닌 계수 {(model.lambdas_ != 0).sum()}개)")

    # ------------------------------------------------------ 6) 예측 / 저장
    surface = stack.predict_surface(model, "cloglog")
    tif = stack.write(os.path.join(out_dir, "suitability_cloglog.tif"), surface)

    imp = model.permutation_importance(Xp, Xb)
    contrib = model.variable_contributions()

    plot_suitability(surface, stack, os.path.join(out_dir, "map_suitability.png"),
                     points_xy=(sx[keep], sy[keep]))
    plot_response_curves(model, Xb, os.path.join(out_dir, "response_curves.png"))
    plot_importance(imp, os.path.join(out_dir, "variable_importance.png"))
    if len(iso_used) == 2:
        (a, va), (b, vb) = iso_used.items()
        plot_isotope_biplot(va, vb, os.path.join(out_dir, "isotope_biplot.png"),
                            title=f"{a} - {b} 분포")

    pd.DataFrame({
        "변수": stack.names,
        "치환중요도(%)": [imp.get(n, 0) for n in stack.names],
        "계수기여도(%)": [contrib.get(n, 0) for n in stack.names],
    }).sort_values("치환중요도(%)", ascending=False).to_csv(
        os.path.join(out_dir, "variable_importance.csv"), index=False, encoding="utf-8-sig")

    summary = {
        "출토지점_사용수": int(len(Xp)),
        "배경점수": int(len(Xb)),
        "AUC": round(float(auc), 4),
        "엔트로피": round(float(model.entropy_), 4),
        "특징종류": list(feature_types),
        "beta_multiplier": beta_multiplier,
        "변수": stack.names,
        "치환중요도_퍼센트": {k: round(v, 2) for k, v in imp.items()},
        "적합도_GeoTIFF": tif,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"[6] 결과 저장 완료 -> {out_dir}")

    return {"model": model, "stack": stack, "surface": surface,
            "summary": summary, "points_xy": (sx[keep], sy[keep])}
