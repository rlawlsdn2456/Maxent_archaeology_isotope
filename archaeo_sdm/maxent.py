"""
MaxEnt (최대 엔트로피) 종분포모델 - 파이썬 구현.

핵심 아이디어
------------
출토지점(presence)과 연구지역 전체에서 무작위로 뽑은 배경점(background)을 비교해서,
"어떤 환경 조건에서 유적/유체가 더 자주 나타나는가"를 학습합니다.

수학적으로 MaxEnt 모델은

    p(x) = exp(lambda · f(x)) / Z          (f = 특징, Z = 정규화 상수)

형태이고, 이는 presence/background 로지스틱 회귀에 배경점 가중치를 크게 주는 것과
동일합니다(Fithian & Hastie 2013; Renner et al. 2015). 여기서는 그 방식(IWLR)에
L1(라쏘) 벌점을 걸어 원래 MaxEnt 프로그램과 같은 정규화를 재현합니다.

출력 형식
--------
  raw      : 배경점 위에서 합이 1이 되는 확률(원 논문의 raw output)
  cloglog  : 0~1 서식지 적합도. cloglog = 1 - exp(-exp(H) * raw)   (Phillips et al. 2017)
  logistic : 구 버전 MaxEnt의 logistic output
  link     : lambda · f(x) (선형 예측값, 로그 스케일)
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .features import FeatureTransformer


class MaxentModel:
    def __init__(
        self,
        feature_types=("linear", "quadratic", "hinge"),
        beta_multiplier: float = 1.0,
        n_knots: int = 10,
        clamp: bool = True,
        background_weight: float = 100.0,
        random_state: int = 0,
    ):
        self.feature_types = tuple(feature_types)
        self.beta_multiplier = float(beta_multiplier)
        self.n_knots = int(n_knots)
        self.clamp = bool(clamp)
        self.background_weight = float(background_weight)
        self.random_state = int(random_state)

    # ------------------------------------------------------------------ fit
    def fit(self, X_presence: np.ndarray, X_background: np.ndarray, var_names=None):
        """X_presence: (m x p) 출토지점 환경값, X_background: (n x p) 배경점 환경값."""
        Xp = np.asarray(X_presence, dtype=float)
        Xb = np.asarray(X_background, dtype=float)
        if Xp.shape[1] != Xb.shape[1]:
            raise ValueError("presence와 background의 변수 개수가 다릅니다.")
        self.var_names_ = list(var_names) if var_names is not None else [f"x{i}" for i in range(Xp.shape[1])]
        self.n_presence_ = Xp.shape[0]

        # 1) 배경점 분포를 기준으로 특징 변환기를 학습 (연구지역 전체 범위를 반영)
        self.transformer_ = FeatureTransformer(self.feature_types, self.n_knots, self.clamp)
        self.transformer_.fit(np.vstack([Xp, Xb]), self.var_names_)

        Fp = self.transformer_.transform(Xp)
        Fb = self.transformer_.transform(Xb)

        # 2) 특징별 정규화 가중치를 '특징 스케일링'으로 구현
        #    (feature/w 로 나눠 넣으면 L1 벌점이 w * |lambda| 와 같아집니다)
        w = self.transformer_.regularization_weights(self.n_presence_, self.beta_multiplier)
        w = np.maximum(w, 1e-8)
        self.reg_weights_ = w
        # 상대 가중치만 특징 스케일에 반영하고, 절대 크기는 C로 넘깁니다
        # (그래야 특징 값이 지나치게 커져 최적화가 느려지는 것을 막을 수 있습니다)
        w_mean = float(np.mean(w))
        w_rel = w / w_mean

        F = np.vstack([Fp, Fb]) / w_rel
        y = np.concatenate([np.ones(len(Fp)), np.zeros(len(Fb))])
        sw = np.concatenate([np.ones(len(Fp)), np.full(len(Fb), self.background_weight)])

        # 3) L1 로지스틱 회귀로 lambda 추정
        C = 1.0 / (max(self.n_presence_, 1) * w_mean)
        clf = LogisticRegression(
            l1_ratio=1.0,          # 1.0 = 라쏘(L1) 정규화
            solver="liblinear",
            C=C,
            fit_intercept=True,
            max_iter=5000,
            random_state=self.random_state,
        )
        clf.fit(F, y, sample_weight=sw)
        self.clf_ = clf
        self.lambdas_ = clf.coef_.ravel() / w_rel  # 원래 특징 스케일로 되돌린 계수

        # 4) 정규화 상수 Z와 엔트로피 H를 배경점에서 계산
        eta_b = self._link_from_features(Fb)
        self._eta_offset_ = float(np.max(eta_b))  # 오버플로 방지용
        expo = np.exp(eta_b - self._eta_offset_)
        self.Z_ = float(np.sum(expo))
        raw_b = expo / self.Z_
        nz = raw_b[raw_b > 0]
        self.entropy_ = float(-np.sum(nz * np.log(nz)))
        self.n_background_ = len(Fb)
        return self

    # -------------------------------------------------------------- predict
    def _link_from_features(self, F: np.ndarray) -> np.ndarray:
        return F @ self.lambdas_

    def predict(self, X: np.ndarray, output: str = "cloglog") -> np.ndarray:
        F = self.transformer_.transform(np.asarray(X, dtype=float))
        eta = self._link_from_features(F)
        if output == "link":
            return eta
        raw = np.exp(eta - self._eta_offset_) / self.Z_
        if output == "raw":
            return raw
        if output == "cloglog":
            return 1.0 - np.exp(-np.exp(self.entropy_) * raw)
        if output == "logistic":
            t = np.exp(self.entropy_) * raw
            return t / (1.0 + t)
        raise ValueError("output은 'cloglog', 'raw', 'logistic', 'link' 중 하나여야 합니다.")

    # ------------------------------------------------------------- 해석 도구
    def variable_contributions(self) -> dict:
        """변수별 기여도(%) - 각 변수에서 파생된 특징 계수 크기의 합 비율."""
        contrib = {v: 0.0 for v in self.var_names_}
        for name, lam in zip(self.transformer_.feature_names_, self.lambdas_):
            for v in self.var_names_:
                if v in name:
                    contrib[v] += abs(lam)
        total = sum(contrib.values()) or 1.0
        return {v: 100.0 * c / total for v, c in sorted(contrib.items(), key=lambda kv: -kv[1])}

    def response_curve(self, var: str, X_reference: np.ndarray, n_points: int = 100):
        """다른 변수는 평균에 고정하고 한 변수만 변화시켰을 때의 반응 곡선."""
        j = self.var_names_.index(var)
        Xr = np.asarray(X_reference, dtype=float)
        base = np.nanmean(Xr, axis=0)
        grid = np.linspace(np.nanmin(Xr[:, j]), np.nanmax(Xr[:, j]), n_points)
        Xq = np.tile(base, (n_points, 1))
        Xq[:, j] = grid
        return grid, self.predict(Xq, output="cloglog")

    def permutation_importance(self, X_presence, X_background, n_repeats: int = 5, random_state=None):
        """변수를 무작위로 섞었을 때 AUC가 얼마나 떨어지는지로 중요도를 계산."""
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        Xp, Xb = np.asarray(X_presence, float), np.asarray(X_background, float)
        base = self.auc(Xp, Xb)
        out = {}
        for j, v in enumerate(self.var_names_):
            drops = []
            for _ in range(n_repeats):
                Xp2, Xb2 = Xp.copy(), Xb.copy()
                allv = np.concatenate([Xp[:, j], Xb[:, j]])
                perm = rng.permutation(allv)
                Xp2[:, j] = perm[: len(Xp)]
                Xb2[:, j] = perm[len(Xp) :]
                drops.append(base - self.auc(Xp2, Xb2))
            out[v] = float(np.mean(drops))
        total = sum(max(v, 0) for v in out.values()) or 1.0
        return {v: 100.0 * max(s, 0) / total for v, s in sorted(out.items(), key=lambda kv: -kv[1])}

    def auc(self, X_presence, X_background) -> float:
        """presence가 background보다 높은 점수를 받을 확률(= ROC AUC)."""
        sp = self.predict(X_presence, "link")
        sb = self.predict(X_background, "link")
        scores = np.concatenate([sp, sb])
        order = np.argsort(scores)
        ranks = np.empty(len(scores), float)
        ranks[order] = np.arange(1, len(scores) + 1)
        # 동점 처리(평균 순위)
        _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
        m, n = len(sp), len(sb)
        return float((ranks[:m].sum() - m * (m + 1) / 2) / (m * n))
