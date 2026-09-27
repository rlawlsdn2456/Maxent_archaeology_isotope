"""
MaxEnt 외의 종분포모델(SDM) 알고리즘 모음 + 공간 교차검증 + 앙상블.

모든 모델은 같은 방식으로 씁니다.

    m = make_model("bioclim").fit(Xp, Xb, var_names)
    suit = m.predict(X)          # 0~1 적합도

알고리즘 계열과 최소 표본(유적 수) 기준
------------------------------------------------------------------
  계열          모델        최소 유적  특징
  환경 포락선   BIOCLIM         3     출토지 환경의 분위수 범위. 외삽을 가장 보수적으로 막음
                Mahalanobis     4     출토지 환경 평균에서의 통계적 거리. 변수 간 상관 반영
  회귀          GLM             5     선형+2차항 로지스틱. 반응 모양 해석이 쉬움
                GAM             8     스플라인 가법 모델. 곡선 반응을 자유롭게
  기계학습      RandomForest   10     비선형·상호작용. 표본이 적으면 과적합
                BRT(GBM)       15     부스팅 회귀나무. 가장 표본을 많이 요구
  최대엔트로피  MaxEnt          5     기존 archaeo_sdm.maxent (점과정 모델과 동등)
------------------------------------------------------------------
유적이 기준보다 적으면 그 모델은 자동으로 건너뛰고, 이유를 기록합니다.
고고학 자료처럼 유적이 4~20곳인 경우에는 포락선·GLM 계열이 현실적인 선택입니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

MODEL_INFO = {
    "bioclim":     ("BIOCLIM (환경 포락선)", 3),
    "mahalanobis": ("Mahalanobis 거리", 4),
    "glm":         ("GLM (로지스틱+2차항)", 5),
    "gam":         ("GAM (스플라인)", 8),
    "rf":          ("Random Forest", 10),
    "brt":         ("BRT (부스팅 회귀나무)", 15),
    "maxent":      ("MaxEnt", 5),
}


# ============================================================ 개별 모델
class _Base:
    key = ""

    def fit(self, Xp, Xb, var_names=None):
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    @property
    def label(self):
        return MODEL_INFO[self.key][0]


class Bioclim(_Base):
    """Busby(1991) BIOCLIM. 각 변수의 출토지 누적분포에서 가운데에 가까울수록 높은 점수.

    점수 = 모든 변수 중 가장 나쁜 값(최소값) → 하나라도 범위를 벗어나면 0.
    """
    key = "bioclim"

    def fit(self, Xp, Xb=None, var_names=None):
        self.sorted_ = np.sort(np.asarray(Xp, float), axis=0)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        n = self.sorted_.shape[0]
        scores = np.empty_like(X)
        for j in range(X.shape[1]):
            col = self.sorted_[:, j]
            lo = np.searchsorted(col, X[:, j], side="left")
            hi = np.searchsorted(col, X[:, j], side="right")
            F = (lo + hi) / (2.0 * n)                  # 경험적 분위수 (동점 보정)
            scores[:, j] = 2.0 * np.minimum(F, 1.0 - F)
        return np.clip(scores.min(axis=1), 0, 1)


class Mahalanobis(_Base):
    """Farber & Kadmon(2003). 출토지 환경 평균에서의 마할라노비스 거리.

    표본이 적어 공분산이 불안정하므로 Ledoit-Wolf 수축 추정을 씁니다.
    적합도 = 카이제곱 분포에서 그 거리보다 멀 확률 (1 - CDF).
    """
    key = "mahalanobis"

    def fit(self, Xp, Xb=None, var_names=None):
        from sklearn.covariance import LedoitWolf

        Xp = np.asarray(Xp, float)
        self.mu_ = Xp.mean(axis=0)
        self.sd_ = Xp.std(axis=0)
        self.sd_[self.sd_ == 0] = 1.0
        Z = (Xp - self.mu_) / self.sd_
        self.prec_ = np.linalg.pinv(LedoitWolf().fit(Z).covariance_)
        self.p_ = Xp.shape[1]
        return self

    def predict(self, X):
        Z = (np.asarray(X, float) - self.mu_) / self.sd_
        d2 = np.einsum("ij,jk,ik->i", Z, self.prec_, Z)
        return 1.0 - chi2.cdf(d2, df=self.p_)


class _SklearnPB(_Base):
    """출토지(1) vs 배경점(0) 분류기를 두 집단 가중치를 맞춰 학습."""

    def _weights(self, n1, n0):
        return np.concatenate([np.full(n1, 0.5 / n1), np.full(n0, 0.5 / n0)]) * (n1 + n0)

    def _xy(self, Xp, Xb):
        Xp, Xb = np.asarray(Xp, float), np.asarray(Xb, float)
        X = np.vstack([Xp, Xb])
        y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xb))])
        return X, y, self._weights(len(Xp), len(Xb))

    def predict(self, X):
        return self.pipe_.predict_proba(np.asarray(X, float))[:, 1]


class GLM(_SklearnPB):
    key = "glm"

    def fit(self, Xp, Xb, var_names=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler

        square = FunctionTransformer(lambda Z: np.hstack([Z, Z ** 2]))
        self.pipe_ = make_pipeline(StandardScaler(), square,
                                   LogisticRegression(C=1.0, max_iter=2000))
        X, y, w = self._xy(Xp, Xb)
        self.pipe_.fit(X, y, logisticregression__sample_weight=w)
        return self


class GAM(_SklearnPB):
    key = "gam"

    def fit(self, Xp, Xb, var_names=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer, StandardScaler

        self.pipe_ = make_pipeline(StandardScaler(),
                                   SplineTransformer(n_knots=4, degree=3, extrapolation="constant"),
                                   LogisticRegression(C=0.5, max_iter=3000))
        X, y, w = self._xy(Xp, Xb)
        self.pipe_.fit(X, y, logisticregression__sample_weight=w)
        return self


class RandomForest(_SklearnPB):
    """배경점이 훨씬 많으므로 트리마다 두 집단을 같은 수로 뽑는 방식(down-sampled RF,
    Valavi et al. 2021)에 가깝게 class_weight='balanced_subsample'을 씁니다."""
    key = "rf"

    def fit(self, Xp, Xb, var_names=None):
        from sklearn.ensemble import RandomForestClassifier

        self.pipe_ = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                            max_features="sqrt",
                                            class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=0)
        X, y, _ = self._xy(Xp, Xb)
        self.pipe_.fit(X, y)
        return self


class BRT(_SklearnPB):
    key = "brt"

    def fit(self, Xp, Xb, var_names=None):
        from sklearn.ensemble import GradientBoostingClassifier

        self.pipe_ = GradientBoostingClassifier(n_estimators=300, learning_rate=0.02,
                                                max_depth=2, subsample=0.75, random_state=0)
        X, y, w = self._xy(Xp, Xb)
        self.pipe_.fit(X, y, sample_weight=w)
        return self


class Maxent(_Base):
    key = "maxent"

    def __init__(self, beta_multiplier: float = 3.0):
        self.beta = beta_multiplier

    def fit(self, Xp, Xb, var_names=None):
        from .maxent import MaxentModel

        self.m_ = MaxentModel(("linear", "quadratic", "hinge"), self.beta)
        self.m_.fit(Xp, Xb, var_names=var_names)
        return self

    def predict(self, X):
        return self.m_.predict(np.asarray(X, float), "cloglog")


_REGISTRY = {"bioclim": Bioclim, "mahalanobis": Mahalanobis, "glm": GLM, "gam": GAM,
             "rf": RandomForest, "brt": BRT, "maxent": Maxent}


def make_model(key: str) -> _Base:
    return _REGISTRY[key]()


def available_models(n_sites: int, keys=None) -> tuple[list, dict]:
    """유적 수로 쓸 수 있는 모델과, 못 쓰는 모델의 사유를 돌려줍니다."""
    keys = keys or list(_REGISTRY)
    ok, skipped = [], {}
    for k in keys:
        need = MODEL_INFO[k][1]
        if n_sites >= need:
            ok.append(k)
        else:
            skipped[k] = f"유적 {n_sites}곳 < 최소 {need}곳"
    return ok, skipped


# ============================================================ 평가
def auc(scores_p, scores_b) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones(len(scores_p)), np.zeros(len(scores_b))])
    s = np.concatenate([scores_p, scores_b])
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def max_tss(scores_p, scores_b) -> float:
    from .validation import tss

    return float(tss(scores_p, scores_b)[0])


def spatial_folds(xy_p, xy_b, n_folds: int = 4, random_state: int = 0):
    """출토지 좌표를 k-means로 묶어 공간 블록을 만들고, 배경점은 가장 가까운 블록에 배정."""
    from sklearn.cluster import KMeans

    P = np.column_stack(xy_p)
    k = max(2, min(n_folds, len(P)))
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(P)
    return km.labels_, km.predict(np.column_stack(xy_b))


def compare_models(Xp, Xb, xy_p, xy_b, var_names, keys=None, n_folds: int = 4,
                   min_train: int = 3) -> pd.DataFrame:
    """모델별 공간 블록 교차검증 성능표."""
    n = len(Xp)
    ok, skipped = available_models(n, keys)
    fp, fb = spatial_folds(xy_p, xy_b, n_folds)
    rows = []
    for k in ok:
        aucs, tsss = [], []
        for f in np.unique(fp):
            tr_p, te_p = fp != f, fp == f
            tr_b, te_b = fb != f, fb == f
            if tr_p.sum() < max(min_train, MODEL_INFO[k][1] - 1) or te_p.sum() < 1 or te_b.sum() < 20:
                continue
            try:
                m = make_model(k).fit(Xp[tr_p], Xb[tr_b], var_names)
                sp, sb = m.predict(Xp[te_p]), m.predict(Xb[te_b])
                aucs.append(auc(sp, sb))
                tsss.append(max_tss(sp, sb))
            except Exception:                   # 표본이 너무 편향되면 개별 fold가 실패할 수 있음
                continue
        m_all = make_model(k).fit(Xp, Xb, var_names)
        rows.append({"model": k, "모델": MODEL_INFO[k][0], "유적수": n,
                     "학습AUC": round(auc(m_all.predict(Xp), m_all.predict(Xb)), 3),
                     "공간CV_AUC": round(float(np.nanmean(aucs)), 3) if aucs else np.nan,
                     "공간CV_TSS": round(float(np.nanmean(tsss)), 3) if tsss else np.nan,
                     "유효fold": len(aucs), "비고": ""})
    for k, why in skipped.items():
        rows.append({"model": k, "모델": MODEL_INFO[k][0], "유적수": n, "학습AUC": np.nan,
                     "공간CV_AUC": np.nan, "공간CV_TSS": np.nan, "유효fold": 0, "비고": why})
    return pd.DataFrame(rows)


# ============================================================ 앙상블
def rank_normalize(pred, reference) -> np.ndarray:
    """예측값을 배경점 예측값 분포 속 백분위로 바꿔, 모델 간 척도를 통일합니다."""
    ref = np.sort(np.asarray(reference, float))
    return np.searchsorted(ref, np.asarray(pred, float), side="right") / max(len(ref), 1)


class Ensemble:
    """공간 교차검증 AUC가 기준(min_auc)을 넘는 모델만, (AUC-0.5)로 가중 평균.

    기준을 넘는 모델이 하나도 없으면 앙상블을 만들지 않습니다 — 약한 모델을
    평균해도 좋은 모델이 되지 않기 때문입니다.
    """

    def __init__(self, min_auc: float = 0.6):
        self.min_auc = min_auc

    def fit(self, Xp, Xb, var_names, table: pd.DataFrame):
        good = table[(table["공간CV_AUC"] >= self.min_auc)]
        self.members_ = {}
        self.weights_ = {}
        self.ref_ = {}
        for _, r in good.iterrows():
            m = make_model(r["model"]).fit(Xp, Xb, var_names)
            self.members_[r["model"]] = m
            self.weights_[r["model"]] = float(r["공간CV_AUC"]) - 0.5
            self.ref_[r["model"]] = m.predict(Xb)
        return self

    @property
    def empty(self) -> bool:
        return not getattr(self, "members_", None)

    def predict(self, X):
        if self.empty:
            raise ValueError("기준을 넘은 모델이 없어 앙상블이 없습니다.")
        tot = sum(self.weights_.values())
        out = np.zeros(len(X))
        for k, m in self.members_.items():
            out += self.weights_[k] / tot * rank_normalize(m.predict(X), self.ref_[k])
        return out
