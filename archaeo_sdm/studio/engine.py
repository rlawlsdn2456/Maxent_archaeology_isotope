"""
SDM Studio 계산부. 화면(app.py)과 분리되어 있어 스크립트에서도 그대로 쓸 수 있습니다.

    from archaeo_sdm.studio import engine
    df = engine.load_dataset()
    engine.run_compare(df, {"taxon": "사람", "periods": ["EIA"]}, "outputs/studio/test")
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .. import paths
from ..chronology import PERIOD_ORDER
from ..isoniche import niche_table, overlap_matrix, plot_niches
from ..models import MODEL_INFO, Ensemble, compare_models, make_model
from ..paleoenv import (worldclim14_scales, worldclim_layers, worldclim_paleo_layers)
from ..plotting import _save, plot_panel_maps, plt
from ..raster import RasterStack
from ..scenarios import (dispersal_simulation, mess, project, summarize_series,
                         threshold_p10)

SLICES = {"lgm": "LGM (약 21 ka)", "mid": "중기 홀로세 (약 6 ka)", "current": "현생 (1970–2000)"}
SLICE_ORDER = ["lgm", "mid", "current"]          # 오래된 순
DEFAULT_BIOS = (1, 4, 12, 15)

DATASET_CANDIDATES = [
    os.path.join(paths.OUTPUTS, "northeast_asia", "dataset_northeast_asia.csv"),
    os.path.join(paths.OUTPUTS, "animals_regional", "dataset_animals.csv"),
    os.path.join(paths.OUTPUTS, "manchuria", "harmonized_dataset.csv"),
]


# ------------------------------------------------------------------ 자료
def list_datasets():
    return [p for p in DATASET_CANDIDATES if os.path.exists(p)]


def load_dataset(path: str | None = None) -> pd.DataFrame:
    path = path or (list_datasets() or [None])[0]
    if not path or not os.path.exists(path):
        raise FileNotFoundError("표준화된 자료(csv)가 없습니다. 먼저 examples/run_northeast_asia.py 를 실행하세요.")
    df = pd.read_csv(path)
    df.attrs["path"] = path
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    def ct(a, b, count="site"):
        t = df.groupby([a, b])[count].nunique().unstack(fill_value=0)
        if b == "period":
            t = t.reindex(columns=[p for p in PERIOD_ORDER if p in t.columns])
        return t

    return {
        "path": df.attrs.get("path", ""),
        "n": int(len(df)), "sites": int(df["site"].nunique()),
        "taxa": sorted(df["taxon_group"].dropna().unique().tolist()),
        "zones": sorted(df["zone"].dropna().unique().tolist()) if "zone" in df else [],
        "periods": [p for p in PERIOD_ORDER if p in set(df["period"].dropna())],
        "taxon_period_sites": ct("taxon_group", "period"),
        "zone_period_sites": ct("zone", "period") if "zone" in df else pd.DataFrame(),
    }


def select(df, taxon=None, periods=None, zones=None) -> pd.DataFrame:
    d = df
    if taxon and taxon != "전체":
        d = d[d["taxon_group"] == taxon] if taxon != "전체동물" else d[d["group"] == "animal"]
    if periods:
        d = d[d["period"].isin(periods)]
    if zones:
        d = d[d["zone"].isin(zones)]
    return d


# ------------------------------------------------------------------ 기후
def slice_layers(key, bios=DEFAULT_BIOS):
    if key == "current":
        return worldclim_layers(paths.WORLDCLIM_DIR, bios), {}
    d = os.path.join(paths.PALEOCLIM_DIR, key)
    return worldclim_paleo_layers(d, bios, slice_key=key), worldclim14_scales(bios)


def slice_stacks(bbox, bios=DEFAULT_BIOS, keys=SLICE_ORDER) -> dict:
    """같은 격자에 맞춘 시점별 기후 스택."""
    grid_ref = worldclim_layers(paths.WORLDCLIM_DIR, bios)[f"bio{bios[0]}"]
    out = {}
    for k in keys:
        layers, scales = slice_layers(k, bios)
        names = list(layers)
        out[k] = RasterStack.from_files([layers[n] for n in names], names, bounds=bbox,
                                        scales=scales, grid_from=grid_ref)
    return out


def auto_bbox(sites: pd.DataFrame, buffer=3.0):
    return (float(sites.lon.min() - buffer), float(sites.lat.min() - buffer),
            float(sites.lon.max() + buffer), float(sites.lat.max() + buffer))


def presence_matrix(stack, sites):
    x, y = sites["lon"].to_numpy(), sites["lat"].to_numpy()
    Xp, keep = stack.extract(x, y)
    idx = stack.thin_by_cell(x[keep], y[keep])
    keep = keep[idx]
    return Xp[idx], (x[keep], y[keep])


def _prepare(df, cfg):
    sub = select(df, cfg.get("taxon"), cfg.get("periods"), cfg.get("zones"))
    sites = sub[["site", "lat", "lon"]].dropna().drop_duplicates()
    if len(sites) < 3:
        raise ValueError(f"조건에 맞는 유적이 {len(sites)}곳뿐입니다(최소 3곳).")
    bbox = tuple(cfg["bbox"]) if cfg.get("bbox") else auto_bbox(sites, cfg.get("buffer", 3.0))
    bios = tuple(cfg.get("bios", DEFAULT_BIOS))
    train = cfg.get("train_slice", "current")
    keys = cfg.get("slices") or [train]
    stacks = slice_stacks(bbox, bios, list(dict.fromkeys(keys + [train])))
    st = stacks[train]
    Xp, xy_p = presence_matrix(st, sites)
    Xb, bx, by = st.sample_background(int(cfg.get("n_background", 5000)), 0)
    return sub, sites, bbox, stacks, st, Xp, xy_p, Xb, (bx, by)


def _label(cfg):
    p = ",".join(cfg.get("periods") or ["전시기"])
    z = ",".join(cfg.get("zones") or ["전지역"])
    return f"{cfg.get('taxon', '전체')} | {p} | {z}"


# ============================================================ 1. 모델 비교
def run_compare(df, cfg, out_dir, log=print) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    sub, sites, bbox, stacks, st, Xp, xy_p, Xb, xy_b = _prepare(df, cfg)
    log(f"유적 {len(sites)}곳 → 격자 정리 후 {len(Xp)}곳, 배경점 {len(Xb)}개")
    keys = cfg.get("models") or list(MODEL_INFO)
    table = compare_models(Xp, Xb, xy_p, xy_b, st.names, keys, n_folds=int(cfg.get("folds", 4)))
    from ..validation import null_model_auc
    table["무작위기준선"] = round(null_model_auc(len(Xp), min(len(Xb), 2000)), 3)
    table.to_csv(os.path.join(out_dir, "model_comparison.csv"), index=False, encoding="utf-8-sig")
    log("모델 비교표 완료")

    surfaces = {}
    for _, r in table.dropna(subset=["학습AUC"]).iterrows():
        m = make_model(r["model"]).fit(Xp, Xb, st.names)
        surfaces[MODEL_INFO[r["model"]][0]] = project(m, {"x": st})["x"]
    ens = Ensemble(float(cfg.get("min_auc", 0.6))).fit(Xp, Xb, st.names, table)
    ens_note = "기준을 넘은 모델 없음 — 앙상블 생략"
    if not ens.empty:
        surfaces["★ 앙상블"] = project(ens, {"x": st})["x"]
        st.write(os.path.join(out_dir, "ensemble.tif"), surfaces["★ 앙상블"])
        ens_note = "앙상블 구성: " + ", ".join(f"{k}(가중 {w:.2f})" for k, w in ens.weights_.items())
    log(ens_note)

    pts = {k: xy_p for k in surfaces}
    img = plot_panel_maps(surfaces, st, os.path.join(out_dir, "models_panel.png"),
                          suptitle=f"알고리즘별 적합도 — {_label(cfg)}", cmap="viridis",
                          vmin=0, vmax=1, cbar_label="적합도 (0–1)", points=pts,
                          basemap=st.data[0], ncols=4)
    return {"table": table, "images": [os.path.basename(img)], "note": ens_note,
            "n_sites": int(len(Xp))}


# ============================================================ 2. 기후 시나리오
def run_scenario(df, cfg, out_dir, log=print) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cfg = {**cfg, "slices": SLICE_ORDER}
    sub, sites, bbox, stacks, st, Xp, xy_p, Xb, xy_b = _prepare(df, cfg)
    key = cfg.get("model", "glm")
    if key == "ensemble":
        keys = cfg.get("models") or list(MODEL_INFO)
        table = compare_models(Xp, Xb, xy_p, xy_b, st.names, keys)
        model = Ensemble(float(cfg.get("min_auc", 0.6))).fit(Xp, Xb, st.names, table)
        if model.empty:
            raise ValueError("공간 교차검증 기준을 넘은 모델이 없어 앙상블로 투영할 수 없습니다. "
                             "단일 모델(BIOCLIM·GLM 등)을 고르세요.")
    else:
        model = make_model(key).fit(Xp, Xb, st.names)
    thr = threshold_p10(model, Xp)
    log(f"{key} 모델 학습(기준 시점: {SLICES[cfg.get('train_slice', 'current')]}), 임계값 P10={thr:.3f}")

    surf = project(model, stacks)
    # 외삽 기준은 모델이 학습한 전체 환경 범위(출토지 + 배경점)입니다.
    mess_maps = {k: mess(np.vstack([Xp, Xb]), stacks[k]) for k in SLICE_ORDER}
    names = [SLICES[k] for k in SLICE_ORDER]
    panels = {SLICES[k]: surf[k] for k in SLICE_ORDER}
    for k in SLICE_ORDER:
        st.write(os.path.join(out_dir, f"suit_{k}.tif"), surf[k])
    img1 = plot_panel_maps(panels, st, os.path.join(out_dir, "scenario_panel.png"),
                           suptitle=f"기후 시나리오별 적합도 ({key}) — {_label(cfg)}",
                           cmap="viridis", vmin=0, vmax=1, cbar_label="적합도",
                           points={SLICES[cfg.get('train_slice', 'current')]: xy_p},
                           basemap=st.data[0], ncols=3)
    img2 = plot_panel_maps({SLICES[k]: np.clip(mess_maps[k], -50, 50) for k in SLICE_ORDER}, st,
                           os.path.join(out_dir, "scenario_mess.png"),
                           suptitle="외삽 위험(MESS) — 음수(빨강)는 학습 범위 밖 기후",
                           cmap="RdBu", diverging=True, cbar_label="MESS",
                           basemap=st.data[0], ncols=3)
    table = summarize_series(names, [surf[k] for k in SLICE_ORDER], thr, st)
    extrap = {SLICES[k]: round(100 * float(np.nanmean(mess_maps[k] < 0)), 1) for k in SLICE_ORDER}
    table["외삽셀비율(%)_이후"] = [extrap[n] for n in names[1:]]
    table.to_csv(os.path.join(out_dir, "range_change.csv"), index=False, encoding="utf-8-sig")
    return {"table": table, "images": [os.path.basename(img1), os.path.basename(img2)],
            "note": f"임계값 P10={thr:.3f} | 시점별 외삽 셀 비율: {extrap}",
            "threshold": thr, "surfaces": surf, "stack": st, "xy_p": xy_p}


# ============================================================ 3. 분산 시뮬레이션
def run_dispersal(df, cfg, out_dir, log=print) -> dict:
    res = run_scenario(df, cfg, out_dir, log)
    st, surf, thr = res["stack"], res["surfaces"], res["threshold"]
    start_slice = cfg.get("start_slice", "lgm")
    order = SLICE_ORDER[SLICE_ORDER.index(start_slice):]
    series = [surf[k] for k in order]

    # 시작 범위: 시작 시점에 적합한 셀 중 출토지에서 가까운 곳(피난처 가정)
    start = np.nan_to_num(series[0], nan=0) >= thr
    if cfg.get("start_from", "refugia") == "sites":
        import rasterio
        start = np.zeros(st.shape, bool)
        r, c = rasterio.transform.rowcol(st.transform, res["xy_p"][0], res["xy_p"][1])
        start[np.asarray(r), np.asarray(c)] = True
    sim = dispersal_simulation(series, thr, start,
                               steps_per_interval=int(cfg.get("steps", 10)),
                               max_dispersal_cells=int(cfg.get("dispersal_cells", 1)),
                               p_colonize=float(cfg.get("p_colonize", 0.6)),
                               random_state=int(cfg.get("seed", 0)))
    log(f"분산 시뮬레이션 {len(sim['frames']) - 1}단계 완료")

    import rasterio.plot
    from PIL import Image

    ext = rasterio.plot.plotting_extent(np.zeros(st.shape), st.transform)
    frames_png = []
    steps = int(cfg.get("steps", 10))
    for i, fr in enumerate(sim["frames"]):
        seg = min((i - 1) // steps, len(order) - 2) if i else 0
        w = 0 if i == 0 else ((i - 1) % steps + 1) / steps
        a, b = series[seg], series[min(seg + 1, len(series) - 1)]
        suit = (1 - w) * a + w * b
        fig, ax = plt.subplots(figsize=(6, 4.6))
        ax.imshow(st.data[0], extent=ext, cmap="Greys", alpha=0.3)
        ax.imshow(np.where(np.nan_to_num(suit, nan=0) >= thr, 1, np.nan), extent=ext,
                  cmap="Greens", vmin=0, vmax=2, alpha=0.6)
        ax.imshow(np.where(fr, 1, np.nan), extent=ext, cmap="autumn", vmin=0, vmax=1)
        ax.scatter(*res["xy_p"], s=10, c="k")
        ax.set_title(f"{SLICES[order[seg]]} → {SLICES[order[min(seg + 1, len(order) - 1)]]}"
                     f"  ({int(w * 100)}%)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.canvas.draw()
        frames_png.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    gif = os.path.join(out_dir, "dispersal.gif")
    frames_png[0].save(gif, save_all=True, append_images=frames_png[1:], duration=350, loop=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    cell_km2 = float(np.nanmean(np.abs(st.transform.a * 111.32 * np.cos(np.radians(np.mean(res["xy_p"][1]))))
                                * abs(st.transform.e) * 110.57))
    ax.plot(np.array(sim["occupied_cells"]) * cell_km2, lw=2.2, label="실제 점유 (분산 제한)")
    ax.plot(np.r_[np.nan, sim["suitable_cells"]] * cell_km2, lw=1.6, ls="--", label="기후상 적합 면적")
    for j in range(1, len(order)):
        ax.axvline(j * steps, c="gray", lw=0.8)
        ax.text(j * steps, ax.get_ylim()[1], f" {SLICES[order[j]]}", va="top", fontsize=8)
    ax.set_xlabel("단계"); ax.set_ylabel("면적 (km², 근사)")
    ax.set_title("분산 제한 하의 점유 범위 변화"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    curve = _save(fig, os.path.join(out_dir, "dispersal_curve.png"))

    lag = [(o, s) for o, s in zip(sim["occupied_cells"][1:], sim["suitable_cells"])]
    fill = round(100 * lag[-1][0] / max(lag[-1][1], 1), 1) if lag else np.nan
    note = (f"최종 단계에서 기후상 적합한 곳의 {fill}%만 실제로 점유 — "
            f"나머지는 이동 거리 제한 때문에 아직 도달하지 못한 곳")
    return {"table": res["table"], "note": note,
            "images": [os.path.basename(gif), os.path.basename(curve)] + res["images"]}


# ============================================================ 4. 동위원소 니치
def run_niche(df, cfg, out_dir, log=print) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    sub = select(df, cfg.get("taxon"), cfg.get("periods"), cfg.get("zones"))
    group_cols = cfg.get("group_by") or ["zone", "period"]
    tbl = niche_table(sub, group_cols, min_n=int(cfg.get("min_n", 5)))
    order = {p: i for i, p in enumerate(PERIOD_ORDER)}
    if "period" in tbl:
        tbl = tbl.sort_values([c for c in group_cols if c != "period"] + ["period"],
                              key=lambda s: s.map(order) if s.name == "period" else s)
    tbl.to_csv(os.path.join(out_dir, "niche_table.csv"), index=False, encoding="utf-8-sig")
    M, groups = overlap_matrix(sub, group_cols, min_n=int(cfg.get("min_n", 5)))
    M.round(1).to_csv(os.path.join(out_dir, "niche_overlap.csv"), encoding="utf-8-sig")
    log(f"집단 {len(tbl)}개 중 {len(groups)}개에서 타원 계산")
    imgs = []
    if groups:
        imgs.append(os.path.basename(plot_niches(groups, os.path.join(out_dir, "niche_ellipses.png"),
                                                 title=f"동위원소 니치 — {_label(cfg)}")))
        if len(groups) > 1:
            fig, ax = plt.subplots(figsize=(0.7 * len(M) + 3, 0.6 * len(M) + 2.5))
            im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=100)
            ax.set_xticks(range(len(M))); ax.set_xticklabels(M.columns, rotation=60, ha="right", fontsize=8)
            ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index, fontsize=8)
            for i in range(len(M)):
                for j in range(len(M)):
                    ax.text(j, i, f"{M.values[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if M.values[i, j] > 60 else "black")
            fig.colorbar(im, ax=ax, label="겹침/합집합 (%)")
            ax.set_title("표준타원 겹침")
            fig.tight_layout()
            imgs.append(os.path.basename(_save(fig, os.path.join(out_dir, "niche_overlap.png"))))
    return {"table": tbl, "images": imgs,
            "note": "δ15N 기저값은 지역마다 다르므로, 지역 간 δ15N 차이를 곧바로 영양단계 차이로 읽지 마십시오."}


def to_json(res: dict) -> dict:
    out = {k: v for k, v in res.items() if k in ("images", "note", "n_sites", "threshold")}
    t = res.get("table")
    if isinstance(t, pd.DataFrame):
        t = t.astype(object).where(t.notna(), None)
        out["table"] = {"columns": [str(c) for c in t.columns], "rows": t.values.tolist()}
    return json.loads(json.dumps(out, default=float))
