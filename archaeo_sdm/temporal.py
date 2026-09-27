"""
시기별(고고학적 편년별) 통합 분석 엔진.

하나의 표준 표(isotope_db가 만든)와 고환경 레이어를 넣으면 다음을 한꺼번에 만듭니다.

  1) 시기 x 분류군 x 프록시별 아이소스케이프
     - 콜라겐 δ13C / δ15N  (식이·영양단계)
     - 아파타이트 δ13C / δ18O (전체 식이·물/기온)
  2) 시기 간 아이소스케이프 변화면 (Δ)
  3) 시기 x 분류군별 MaxEnt 서식지 적합도면
  4) 시기 간 니치 중첩도(Schoener's D) 행렬
  5) 요약표(csv) + 비교 그림(png)

모든 래스터는 GeoTIFF로 저장되어 QGIS에서 바로 열립니다.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .chronology import MANCHURIA_PERIODS, PERIOD_ORDER
from .isoscape import build_isoscape, delta_isoscape, schoener_d, site_means
from .maxent import MaxentModel
from .plotting import plot_panel_maps
from .raster import RasterStack

PROXIES = {
    "d13C_coll": ("콜라겐 δ13C", "‰ VPDB", "RdYlBu_r"),
    "d15N_coll": ("콜라겐 δ15N", "‰ AIR", "YlGnBu"),
    "d13C_ap": ("아파타이트 δ13C", "‰ VPDB", "RdYlBu_r"),
    "d18O_ap": ("아파타이트 δ18O", "‰ VPDB", "PuOr"),
}


def _safe(name: str) -> str:
    for ch in r'\/:*?"<>| ':
        name = name.replace(ch, "_")
    return name


# ------------------------------------------------------------- 아이소스케이프
def temporal_isoscapes(df: pd.DataFrame, stack: RasterStack, out_dir: str,
                       proxies=("d13C_coll", "d15N_coll"),
                       taxa=("사람", "돼지"),
                       periods=None, min_sites: int = 3,
                       max_distance_km: float = 250.0,
                       verbose: bool = True) -> dict:
    """시기 x 분류군 x 프록시 아이소스케이프 + 시기간 변화면."""
    periods = periods or MANCHURIA_PERIODS
    order = [p.key for p in periods]
    labels = {p.key: f"{p.key} {p.label_ko}\n({p.calendar()})" for p in periods}
    log = print if verbose else (lambda *a, **k: None)
    os.makedirs(out_dir, exist_ok=True)

    results, records = {}, []
    for proxy in proxies:
        if proxy not in df.columns:
            continue
        pretty, unit, cmap = PROXIES[proxy]
        for taxon in taxa:
            sub_t = df[df["taxon_group"] == taxon]
            surfaces, pts, skipped = {}, {}, []
            for key in order:
                sub = sub_t[sub_t["period"] == key]
                sm = site_means(sub, proxy)
                if len(sm) < min_sites:
                    skipped.append(f"{key}({len(sm)}곳)")
                    continue
                surf, support = build_isoscape(sm, stack, max_distance_km=max_distance_km)
                surfaces[key] = surf
                pts[key] = (sm["lon"].to_numpy(), sm["lat"].to_numpy())
                tag = f"{_safe(taxon)}_{proxy}_{key}"
                stack.write(os.path.join(out_dir, f"isoscape_{tag}.tif"), surf)
                records.append({
                    "분류군": taxon, "프록시": proxy, "period": key,
                    "유적수": len(sm), "개체수": int(sm["n"].sum()),
                    "평균": round(float(np.nanmean(sm["value"])), 2),
                    "최소": round(float(np.nanmin(sm["value"])), 2),
                    "최대": round(float(np.nanmax(sm["value"])), 2),
                })
            if skipped:
                log(f"    · {taxon}/{pretty}: 유적 부족으로 제외 -> {', '.join(skipped)}")
            if not surfaces:
                continue
            results[(taxon, proxy)] = surfaces

            plot_panel_maps(
                surfaces, stack,
                os.path.join(out_dir, f"panel_{_safe(taxon)}_{proxy}.png"),
                titles=labels, suptitle=f"{taxon} — {pretty} 아이소스케이프의 시기별 변화",
                cmap=cmap, cbar_label=f"{pretty} ({unit})", points=pts,
                basemap=stack.data[0])

            # 연속한 시기 사이의 변화면
            keys = [k for k in order if k in surfaces]
            deltas = {}
            for a, b in zip(keys[:-1], keys[1:]):
                d = delta_isoscape(surfaces[a], surfaces[b])
                deltas[f"{a}→{b}"] = d
                stack.write(os.path.join(out_dir, f"delta_{_safe(taxon)}_{proxy}_{a}_to_{b}.tif"), d)
            if deltas:
                plot_panel_maps(
                    deltas, stack,
                    os.path.join(out_dir, f"panel_delta_{_safe(taxon)}_{proxy}.png"),
                    suptitle=f"{taxon} — {pretty} 시기간 변화량 (Δ)",
                    cmap="RdBu_r", diverging=True, cbar_label=f"Δ{pretty} ({unit})",
                    basemap=stack.data[0])
            log(f"[iso] {taxon}/{pretty}: {len(surfaces)}개 시기 생성")

    if records:
        pd.DataFrame(records).to_csv(os.path.join(out_dir, "isoscape_summary.csv"),
                                     index=False, encoding="utf-8-sig")
    return results


# --------------------------------------------------------------- 시기별 SDM
def temporal_sdm(df: pd.DataFrame, stack: RasterStack, out_dir: str,
                 taxa=("돼지",), periods=None, min_sites: int = 4,
                 feature_types=("linear", "quadratic", "hinge"),
                 beta_multiplier: float = 3.0, n_background: int = 5000,
                 isoscape_layers: dict | None = None,
                 stacks: dict | None = None,
                 sdm_iso_proxies=("d13C_coll", "d15N_coll"),
                 max_point_loss: float = 0.4,
                 random_state: int = 0, verbose: bool = True) -> dict:
    """시기 x 분류군별 MaxEnt 적합도면과 시기 간 니치 중첩도.

    두 가지 모델을 항상 함께 만듭니다.
      env     : 고환경(고기후·지형) 변수만 사용 - 연구지역 전체를 예측
      env+iso : 여기에 그 시기의 아이소스케이프를 추가 - 동위원소 지지범위 안만 예측
    주의: 두 모델의 AUC를 직접 비교하면 안 됩니다. env+iso는 예측 범위가 아이소스케이프
    지지범위로 좁아져 배경점이 더 균질해지므로 AUC가 구조적으로 낮게 나옵니다.
    같은 범위에서 비교하려면 env 모델도 같은 마스크로 잘라 다시 학습해야 합니다.

    sdm_iso_proxies : 결합에 쓸 프록시. 시료가 희박한 아파타이트를 그대로 넣으면
                      지지범위 밖이 전부 결측이 되어 유적 자체가 탈락합니다.
    max_point_loss  : 어떤 아이소스케이프 레이어 때문에 이 비율 이상의 유적이
                      탈락하면 그 레이어는 자동으로 빼고 진행합니다.
    stacks          : {시기: RasterStack} - 시기마다 다른 고기후를 쓸 때 넘깁니다.
                      (모두 같은 격자여야 시기 간 비교가 가능합니다)
    """
    periods = periods or MANCHURIA_PERIODS
    order = [p.key for p in periods]
    labels = {p.key: f"{p.key} {p.label_ko}" for p in periods}
    log = print if verbose else (lambda *a, **k: None)
    os.makedirs(out_dir, exist_ok=True)

    base_names = list(stack.names)
    out, rows = {}, []

    def _fit(s, x, y, tag):
        """주어진 스택에서 MaxEnt를 학습하고 (면, 정보) 반환. 실패하면 None."""
        Xp, keep = s.extract(x, y)
        if len(keep) == 0:
            return None
        idx = s.thin_by_cell(x[keep], y[keep])
        keep, Xp = keep[idx], Xp[idx]
        if len(Xp) < min_sites:
            log(f"    · {tag}: 격자 정리 후 {len(Xp)}곳 -> 생략")
            return None
        Xb, _, _ = s.sample_background(n_background, random_state)
        model = MaxentModel(feature_types, beta_multiplier, random_state=random_state)
        model.fit(Xp, Xb, var_names=s.names)
        surf = s.predict_surface(model, "cloglog")
        imp = model.permutation_importance(Xp, Xb, n_repeats=3)
        top = max(imp, key=imp.get) if imp else "-"
        return surf, {"유적수": int(len(Xp)), "AUC": round(model.auc(Xp, Xb), 3),
                      "최대기여변수": top, "기여도(%)": round(imp.get(top, 0), 1),
                      "변수": ", ".join(s.names)}, (x[keep], y[keep])

    for taxon in taxa:
        sub_t = df[df["taxon_group"] == taxon]
        surf_env, surf_iso, pts = {}, {}, {}
        for key in order:
            sub = sub_t[sub_t["period"] == key]
            sites = sub[["site", "lat", "lon"]].dropna().drop_duplicates()
            if len(sites) < min_sites:
                log(f"    · {taxon}/{key}: 유적 {len(sites)}곳 -> 모델 생략(최소 {min_sites})")
                continue

            src = (stacks or {}).get(key, stack)
            base = RasterStack(src.data.copy(), list(src.names), src.profile.copy())
            x, y = base.to_crs_xy(sites["lon"].to_numpy(), sites["lat"].to_numpy())

            # (1) 고환경 단독 모델
            r = _fit(base, x, y, f"{taxon}/{key}/env")
            if r is None:
                continue
            surf, info, kept = r
            surf_env[key] = surf
            pts[key] = kept
            base.write(os.path.join(out_dir, f"sdm_{_safe(taxon)}_{key}_env.tif"), surf)
            rows.append({"분류군": taxon, "period": key, "모델": "env", **info})
            log(f"[sdm] {taxon}/{key} env    : 유적 {info['유적수']}곳, "
                f"AUC={info['AUC']}, 주요변수={info['최대기여변수']}")

            # (2) 동위원소 결합 모델
            if not isoscape_layers:
                continue
            s2 = RasterStack(base.data.copy(), list(base.names), base.profile.copy())
            n_base = len(s2.extract(x, y)[1])
            added = []
            for (tx, proxy), by_period in isoscape_layers.items():
                if tx != taxon or key not in by_period or proxy not in sdm_iso_proxies:
                    continue
                s2.add_layer(f"iso_{proxy}", by_period[key])
                n_now = len(s2.extract(x, y)[1])
                if n_base and (n_base - n_now) / n_base > max_point_loss:
                    log(f"    · {taxon}/{key}: iso_{proxy} 결합 시 유적 "
                        f"{n_base}->{n_now}곳으로 줄어 제외")
                    s2.data = s2.data[:-1]
                    s2.names.pop()
                else:
                    added.append(proxy)
            if not added:
                continue
            r2 = _fit(s2, x, y, f"{taxon}/{key}/env+iso")
            if r2 is None:
                continue
            surf2, info2, _ = r2
            surf_iso[key] = surf2
            s2.write(os.path.join(out_dir, f"sdm_{_safe(taxon)}_{key}_env_iso.tif"), surf2)
            rows.append({"분류군": taxon, "period": key, "모델": "env+iso", **info2})
            log(f"[sdm] {taxon}/{key} env+iso: 유적 {info2['유적수']}곳, "
                f"AUC={info2['AUC']}, 주요변수={info2['최대기여변수']}")

        if surf_env:
            out[taxon] = {"env": surf_env, "env+iso": surf_iso}
            plot_panel_maps(surf_env, stack,
                            os.path.join(out_dir, f"panel_sdm_{_safe(taxon)}_env.png"),
                            titles=labels,
                            suptitle=f"{taxon} — 시기별 서식지 적합도 (고환경 변수만)",
                            cmap="viridis", vmin=0, vmax=1, cbar_label="적합도 (cloglog)",
                            points=pts, basemap=stack.data[0])
            if surf_iso:
                plot_panel_maps(surf_iso, stack,
                                os.path.join(out_dir, f"panel_sdm_{_safe(taxon)}_env_iso.png"),
                                titles=labels,
                                suptitle=f"{taxon} — 시기별 서식지 적합도 (고환경 + 동위원소)",
                                cmap="viridis", vmin=0, vmax=1, cbar_label="적합도 (cloglog)",
                                points=pts, basemap=stack.data[0])
            keys = [k for k in order if k in surf_env]
            D = pd.DataFrame(index=keys, columns=keys, dtype=float)
            for a in keys:
                for b in keys:
                    D.loc[a, b] = round(schoener_d(surf_env[a], surf_env[b]), 3)
            D.to_csv(os.path.join(out_dir, f"niche_overlap_{_safe(taxon)}.csv"),
                     encoding="utf-8-sig")

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "sdm_summary.csv"),
                                  index=False, encoding="utf-8-sig")
    return out


# ------------------------------------------------------------- 전체 실행기
def run_temporal_study(df: pd.DataFrame, env_layers: dict, out_dir: str,
                       bbox=None, proxies=("d13C_coll", "d15N_coll"),
                       iso_taxa=("사람", "돼지"), sdm_taxa=("돼지",),
                       periods=None, couple_isoscape_to_sdm: bool = True,
                 stacks: dict | None = None,
                 sdm_iso_proxies=("d13C_coll", "d15N_coll"),
                       min_sites_iso: int = 3, min_sites_sdm: int = 4,
                       max_distance_km: float = 250.0,
                       beta_multiplier: float = 3.0,
                       verbose: bool = True) -> dict:
    """아이소스케이프 -> SDM -> 요약까지 한 번에."""
    periods = periods or MANCHURIA_PERIODS
    log = print if verbose else (lambda *a, **k: None)
    os.makedirs(out_dir, exist_ok=True)

    names = list(env_layers)
    stack = RasterStack.from_files([env_layers[n] for n in names], names, bounds=bbox)
    log(f"[env] 고환경 레이어 {names} / 격자 {stack.shape} / CRS {stack.crs}")

    iso = temporal_isoscapes(df, stack, out_dir, proxies, iso_taxa, periods,
                             min_sites_iso, max_distance_km, verbose)
    sdm = temporal_sdm(df, stack, out_dir, sdm_taxa, periods, min_sites_sdm,
                       beta_multiplier=beta_multiplier,
                       isoscape_layers=iso if couple_isoscape_to_sdm else None,
                       stacks=stacks, sdm_iso_proxies=sdm_iso_proxies,
                       verbose=verbose)

    meta = {
        "격자": list(stack.shape), "CRS": str(stack.crs),
        "고환경레이어": names, "프록시": list(proxies),
        "아이소스케이프_분류군": list(iso_taxa), "SDM_분류군": list(sdm_taxa),
        "시기": [{"key": p.key, "이름": p.label_ko, "연대": p.calendar()} for p in periods],
        "총_시료수": int(len(df)), "총_유적수": int(df["site"].nunique()),
    }
    with open(os.path.join(out_dir, "study_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return {"stack": stack, "isoscapes": iso, "sdm": sdm, "meta": meta}
