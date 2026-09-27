"""
MaxEnt 특징(feature) 변환기.

MaxEnt는 환경변수를 그대로 쓰지 않고, 여러 가지 '특징(feature)'으로 바꿔서 사용합니다.
(Phillips et al. 2006; Elith et al. 2011 참고)

  - linear    : x                      (단순 선형 반응)
  - quadratic : x^2                    (최적 구간이 있는 종 모양 반응)
  - product   : x_i * x_j              (변수 간 상호작용)
  - hinge     : max(0, x - knot)       (특정 값부터 반응이 꺾이는 형태)
  - threshold : 1 if x > knot else 0   (계단형 반응)

모든 원본 변수는 먼저 [0, 1] 범위로 정규화(min-max)한 뒤 변환합니다.
학습 범위를 벗어난 값은 clamping(양끝값으로 고정)해서 외삽 폭주를 막습니다.
"""

from __future__ import annotations

import numpy as np

# MaxEnt가 관례적으로 쓰는 특징 종류별 정규화 계수(regularization) 기본값.
# 값이 클수록 그 특징 종류에 더 강한 벌점을 주어 과적합을 줄입니다.
DEFAULT_REG_WEIGHTS = {
    "linear": 0.05,
    "quadratic": 0.05,
    "product": 0.05,
    "hinge": 0.50,
    "threshold": 1.00,
}

ALL_FEATURE_TYPES = ("linear", "quadratic", "product", "hinge", "threshold")


class FeatureTransformer:
    """환경변수 행렬 X(n샘플 x p변수)를 MaxEnt 특징 행렬로 바꿉니다."""

    def __init__(
        self,
        feature_types=("linear", "quadratic", "hinge"),
        n_knots: int = 10,
        clamp: bool = True,
    ):
        bad = [f for f in feature_types if f not in ALL_FEATURE_TYPES]
        if bad:
            raise ValueError(f"알 수 없는 feature 종류: {bad}. 가능: {ALL_FEATURE_TYPES}")
        self.feature_types = tuple(feature_types)
        self.n_knots = int(n_knots)
        self.clamp = bool(clamp)

        # fit() 후에 채워지는 값들
        self.var_names_: list[str] = []
        self.x_min_: np.ndarray | None = None
        self.x_max_: np.ndarray | None = None
        self.feature_names_: list[str] = []
        self.feature_classes_: list[str] = []

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, var_names=None) -> "FeatureTransformer":
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        self.var_names_ = list(var_names) if var_names is not None else [f"x{i}" for i in range(p)]

        self.x_min_ = np.nanmin(X, axis=0)
        self.x_max_ = np.nanmax(X, axis=0)
        # 상수 변수(최대=최소)일 때 0으로 나누는 것을 방지
        span = self.x_max_ - self.x_min_
        span[span == 0] = 1.0
        self._span_ = span

        # hinge / threshold 용 분위수 knot을 변수별로 미리 계산
        Xs = self._scale(X)
        qs = np.linspace(0.0, 1.0, self.n_knots + 2)[1:-1]  # 양 끝은 제외
        self.knots_ = [np.unique(np.nanquantile(Xs[:, j], qs)) for j in range(p)]

        # 특징 이름과 종류를 한 번 만들어 둔다(변환 결과 열 순서와 정확히 일치)
        self.feature_names_, self.feature_classes_ = [], []
        for kind, name in self._iter_feature_specs():
            self.feature_classes_.append(kind)
            self.feature_names_.append(name)
        return self

    # ------------------------------------------------------------- transform
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.x_min_ is None:
            raise RuntimeError("먼저 fit()을 호출해야 합니다.")
        Xs = self._scale(np.asarray(X, dtype=float))
        cols = [f(Xs) for f in self._feature_funcs(Xs.shape[1])]
        return np.column_stack(cols) if cols else np.empty((Xs.shape[0], 0))

    def fit_transform(self, X, var_names=None) -> np.ndarray:
        return self.fit(X, var_names).transform(X)

    # --------------------------------------------------------------- 내부용
    def _scale(self, X: np.ndarray) -> np.ndarray:
        """min-max 정규화 + (옵션) clamping."""
        Xs = (X - self.x_min_) / self._span_
        if self.clamp:
            Xs = np.clip(Xs, 0.0, 1.0)
        return Xs

    def _iter_feature_specs(self):
        """(특징종류, 이름)을 _feature_funcs와 같은 순서로 생성."""
        p = len(self.var_names_)
        names = self.var_names_
        if "linear" in self.feature_types:
            for j in range(p):
                yield "linear", f"L({names[j]})"
        if "quadratic" in self.feature_types:
            for j in range(p):
                yield "quadratic", f"Q({names[j]})"
        if "product" in self.feature_types:
            for j in range(p):
                for k in range(j + 1, p):
                    yield "product", f"P({names[j]}*{names[k]})"
        if "hinge" in self.feature_types:
            for j in range(p):
                for kn in self.knots_[j]:
                    yield "hinge", f"H+({names[j]}>{kn:.3f})"
                for kn in self.knots_[j]:
                    yield "hinge", f"H-({names[j]}<{kn:.3f})"
        if "threshold" in self.feature_types:
            for j in range(p):
                for kn in self.knots_[j]:
                    yield "threshold", f"T({names[j]}>{kn:.3f})"

    def _feature_funcs(self, p: int):
        """열을 만드는 함수들의 리스트. _iter_feature_specs와 순서가 같아야 합니다."""
        funcs = []
        if "linear" in self.feature_types:
            for j in range(p):
                funcs.append(lambda Xs, j=j: Xs[:, j])
        if "quadratic" in self.feature_types:
            for j in range(p):
                funcs.append(lambda Xs, j=j: Xs[:, j] ** 2)
        if "product" in self.feature_types:
            for j in range(p):
                for k in range(j + 1, p):
                    funcs.append(lambda Xs, j=j, k=k: Xs[:, j] * Xs[:, k])
        if "hinge" in self.feature_types:
            for j in range(p):
                for kn in self.knots_[j]:  # forward hinge
                    denom = max(1.0 - kn, 1e-9)
                    funcs.append(lambda Xs, j=j, kn=kn, d=denom: np.maximum(0.0, Xs[:, j] - kn) / d)
                for kn in self.knots_[j]:  # reverse hinge
                    denom = max(kn, 1e-9)
                    funcs.append(lambda Xs, j=j, kn=kn, d=denom: np.maximum(0.0, kn - Xs[:, j]) / d)
        if "threshold" in self.feature_types:
            for j in range(p):
                for kn in self.knots_[j]:
                    funcs.append(lambda Xs, j=j, kn=kn: (Xs[:, j] > kn).astype(float))
        return funcs

    # ------------------------------------------------------- 정규화 가중치
    def regularization_weights(self, n_presence: int, beta_multiplier: float = 1.0) -> np.ndarray:
        """특징별 L1 벌점 가중치.

        MaxEnt와 동일하게 표본 수가 적을수록 벌점을 크게 줍니다
        (beta_j ∝ 1/sqrt(m)).
        """
        m = max(int(n_presence), 1)
        base = np.array([DEFAULT_REG_WEIGHTS[c] for c in self.feature_classes_], dtype=float)
        return beta_multiplier * base / np.sqrt(m)
