"""
공간 블록 교차검증(spatial block cross-validation).

보통의 무작위 교차검증은 종분포모델에서 성능을 크게 부풀립니다.
가까운 지점끼리 학습셋과 검증셋에 나뉘어 들어가면, 모델은 '환경을 배웠다'가 아니라
'그 근처를 외웠다'로도 높은 점수를 받기 때문입니다(공간 자기상관).

공간 블록 교차검증은 연구지역을 덩어리(블록)로 나누고 블록 단위로 접기(fold)를
만들어, 학습에 쓰지 않은 '다른 지역'에서 성능을 잽니다.
(Roberts et al. 2017; Valavi et al. 2019)

제공하는 방식
  checkerboard : 격자 바둑판을 블록으로 (블록 크기 지정)
  kmeans       : 좌표를 k-means로 묶어 공간 덩어리 생성
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .maxent import MaxentModel


def checkerboard_blocks(x, y, block_size: float, n_folds: int = 4) -> np.ndarray:
    """좌표를 block_size 격자로 나누고, 격자 칸을 순번대로 fold에 배정."""
    bx = np.floor(np.asarray(x, float) / block_size).astype(int)
    by = np.floor(np.asarray(y, float) / block_size).astype(int)
    return ((bx + by) % n_folds).astype(int)


def kmeans_blocks(x, y, n_folds: int = 4, random_state: int = 0) -> np.ndarray:
    """좌표를 k-means로 묶어 공간 덩어리 fold를 만듭니다."""
    from sklearn.cluster import KMeans

    P = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    k = min(n_folds, len(P))
    return KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(P)


def tss(presence_scores, background_scores) -> tuple[float, float]:
    """최대 TSS(True Skill Statistic)와 그때의 임계값."""
    p = np.asarray(presence_scores, float)
    b = np.asarray(background_scores, float)
    thresholds = np.unique(np.concatenate([p, b]))
    if len(thresholds) > 500:
        thresholds = np.quantile(thresholds, np.linspace(0, 1, 500))
    best, best_t = -1.0, np.nan
    for t in thresholds:
        sens = float((p >= t).mean()) if len(p) else 0.0          # 민감도
        spec = float((b < t).mean()) if len(b) else 0.0           # 특이도
        s = sens + spec - 1.0
        if s > best:
            best, best_t = s, float(t)
    return best, best_t


def spatial_block_cv(X_presence, xy_presence, X_background, xy_background,
                     var_names=None, n_folds: int = 4, method: str = "kmeans",
                     block_size: float = 2.0, feature_types=("linear", "quadratic", "hinge"),
                     beta_multiplier: float = 3.0, random_state: int = 0,
                     min_train: int = 4, verbose: bool = False) -> dict:
    """공간 블록 교차검증으로 MaxEnt 성능을 평가합니다.

    xy_presence / xy_background : (x배열, y배열)  블록을 나눌 좌표
    반환 : {"folds": DataFrame, "AUC_평균": float, "TSS_평균": float, ...}
    """
    Xp = np.asarray(X_presence, float)
    Xb = np.asarray(X_background, float)
    px, py = np.asarray(xy_presence[0], float), np.asarray(xy_presence[1], float)
    bx, by = np.asarray(xy_background[0], float), np.asarray(xy_background[1], float)

    if method == "checkerboard":
        fp = checkerboard_blocks(px, py, block_size, n_folds)
        fb = checkerboard_blocks(bx, by, block_size, n_folds)
    else:
        from sklearn.cluster import KMeans

        k = min(n_folds, len(px))
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(
            np.column_stack([px, py]))
        fp = km.labels_
        fb = km.predict(np.column_stack([bx, by]))

    rows = []
    for f in sorted(set(fp)):
        tr_p, te_p = fp != f, fp == f
        tr_b, te_b = fb != f, fb == f
        if tr_p.sum() < min_train or te_p.sum() < 1 or tr_b.sum() < 10 or te_b.sum() < 10:
            rows.append({"fold": int(f), "학습_유적": int(tr_p.sum()),
                         "검증_유적": int(te_p.sum()), "AUC": np.nan, "TSS": np.nan,
                         "비고": "표본 부족으로 건너뜀"})
            continue
        model = MaxentModel(feature_types, beta_multiplier, random_state=random_state)
        model.fit(Xp[tr_p], Xb[tr_b], var_names=var_names)
        auc = model.auc(Xp[te_p], Xb[te_b])
        t, thr = tss(model.predict(Xp[te_p], "cloglog"), model.predict(Xb[te_b], "cloglog"))
        rows.append({"fold": int(f), "학습_유적": int(tr_p.sum()), "검증_유적": int(te_p.sum()),
                     "AUC": round(float(auc), 3), "TSS": round(float(t), 3),
                     "임계값": round(float(thr), 3), "비고": ""})
        if verbose:
            print(f"    fold {f}: 학습 {tr_p.sum()} / 검증 {te_p.sum()} -> AUC {auc:.3f}, TSS {t:.3f}")

    folds = pd.DataFrame(rows)
    valid = folds["AUC"].notna()
    return {
        "folds": folds,
        "방식": method,
        "AUC_평균": round(float(folds.loc[valid, "AUC"].mean()), 3) if valid.any() else np.nan,
        "AUC_표준편차": round(float(folds.loc[valid, "AUC"].std()), 3) if valid.sum() > 1 else np.nan,
        "TSS_평균": round(float(folds.loc[valid, "TSS"].mean()), 3) if valid.any() else np.nan,
        "유효_fold수": int(valid.sum()),
    }


def null_model_auc(n_presence: int, n_background: int, n_iter: int = 99,
                   random_state: int = 0) -> float:
    """무작위 예측의 AUC 분포 상위 5% 값(성능 판단의 최소 기준선)."""
    rng = np.random.default_rng(random_state)
    aucs = []
    for _ in range(n_iter):
        s = rng.random(n_presence + n_background)
        p, b = s[:n_presence], s[n_presence:]
        aucs.append(float((p[:, None] > b[None, :]).mean()))
    return float(np.quantile(aucs, 0.95))
