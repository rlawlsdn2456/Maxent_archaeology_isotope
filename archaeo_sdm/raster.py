"""
환경 래스터(GeoTIFF) 묶음을 다루는 도구.

여러 장의 환경 레이어(고도, 경사, 강수량, PaleoClim 기후, 동위원소 보간면 등)를
하나의 격자에 맞춰 정렬(align)하고, 점 위치에서 값을 뽑고, 예측 결과를
다시 GeoTIFF로 저장합니다.
"""

from __future__ import annotations

import os

import numpy as np
import rasterio
import rasterio.windows
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from pyproj import Transformer

RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}


class RasterStack:
    """같은 격자에 정렬된 환경 레이어 묶음."""

    def __init__(self, data: np.ndarray, names, profile: dict):
        self.data = data                    # (밴드, 행, 열), 결측은 np.nan
        self.names = list(names)
        self.profile = profile              # rasterio 프로파일(기준 격자)

    # ----------------------------------------------------------- 생성자들
    @classmethod
    def from_files(cls, paths, names=None, reference: int = 0, resampling: str = "bilinear",
                   categorical=None, bounds=None, scales=None, grid_from=None):
        """여러 GeoTIFF를 reference 번째 파일의 격자에 맞춰 읽어들입니다.

        bounds : (minx, miny, maxx, maxy) - 기준 래스터의 좌표계 기준으로 잘라낼 범위.
                 연구지역만 잘라 쓰면 계산이 훨씬 빨라집니다.
        scales : {레이어이름: 나눌 값} - 자료마다 단위가 다를 때 맞춰 줍니다.
                 (예: WorldClim 1.4 고기후는 기온이 10배 정수로 저장되어 있음)
        grid_from : 격자(해상도·원점)를 정의할 파일 경로. 시기마다 다른 기후자료를
                 쓸 때 이 값을 같게 두면 모든 시기의 결과가 같은 격자에 놓입니다.
        """
        paths = list(paths)
        if names is None:
            names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        categorical = set(categorical or [])

        with rasterio.open(grid_from or paths[reference]) as ref:
            profile = ref.profile.copy()
            ref_crs = ref.crs
            if bounds is None:
                ref_transform, H, W = ref.transform, ref.height, ref.width
            else:
                win = rasterio.windows.from_bounds(*bounds, transform=ref.transform)
                win = win.round_offsets().round_lengths()
                win = win.intersection(rasterio.windows.Window(0, 0, ref.width, ref.height))
                ref_transform = ref.window_transform(win)
                H, W = int(win.height), int(win.width)
            profile.update(count=1, dtype="float32", nodata=np.nan,
                           height=H, width=W, transform=ref_transform)

        bands = []
        for p, nm in zip(paths, names):
            with rasterio.open(p) as src:
                same_grid = (
                    src.crs == ref_crs and src.transform == ref_transform
                    and src.height == H and src.width == W
                )
                if same_grid:
                    out = src.read(1, masked=True).astype("float32").filled(np.nan)
                else:
                    arr = src.read(1, masked=True).astype("float32").filled(np.nan)
                    out = np.full((H, W), np.nan, dtype="float32")
                    method = RESAMPLING["nearest"] if nm in categorical else RESAMPLING[resampling]
                    reproject(
                        source=arr, destination=out,
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=ref_transform, dst_crs=ref_crs,
                        src_nodata=np.nan, dst_nodata=np.nan,
                        resampling=method,
                    )
                if scales and nm in scales:
                    out = out / float(scales[nm])
                bands.append(out)
        return cls(np.stack(bands), names, profile)

    # ------------------------------------------------------------- 속성들
    @property
    def crs(self):
        return self.profile["crs"]

    @property
    def transform(self):
        return self.profile["transform"]

    @property
    def shape(self):
        return self.data.shape[1], self.data.shape[2]

    @property
    def valid_mask(self) -> np.ndarray:
        """모든 레이어에 값이 있는 셀만 True."""
        return ~np.any(np.isnan(self.data), axis=0)

    # ------------------------------------------------------------ 값 추출
    def to_crs_xy(self, lon, lat, src_epsg: str = "EPSG:4326"):
        """경위도(또는 다른 CRS) 좌표를 래스터 CRS 좌표로 변환."""
        if str(self.crs).upper() == str(src_epsg).upper():
            return np.asarray(lon, float), np.asarray(lat, float)
        tf = Transformer.from_crs(src_epsg, self.crs, always_xy=True)
        x, y = tf.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def rowcol(self, x, y):
        rows, cols = rasterio.transform.rowcol(self.transform, np.asarray(x), np.asarray(y))
        return np.asarray(rows), np.asarray(cols)

    def extract(self, x, y, drop_invalid: bool = True):
        """점 좌표(래스터 CRS)에서 환경값 추출. 반환: (값 배열, 사용된 인덱스)."""
        rows, cols = self.rowcol(x, y)
        H, W = self.shape
        inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        vals = np.full((len(rows), len(self.names)), np.nan, dtype=float)
        vals[inside] = self.data[:, rows[inside], cols[inside]].T
        idx = np.arange(len(rows))
        if drop_invalid:
            keep = inside & ~np.any(np.isnan(vals), axis=1)
            return vals[keep], idx[keep]
        return vals, idx

    def thin_by_cell(self, x, y):
        """같은 격자 셀에 여러 점이 있으면 하나만 남깁니다(공간 편향 완화)."""
        rows, cols = self.rowcol(x, y)
        _, first = np.unique(np.stack([rows, cols], 1), axis=0, return_index=True)
        return np.sort(first)

    def sample_background(self, n: int = 10000, random_state: int = 0, bias_mask=None):
        """유효 셀에서 배경점을 무작위 추출. 반환: (환경값, x, y)."""
        rng = np.random.default_rng(random_state)
        mask = self.valid_mask
        if bias_mask is not None:
            mask = mask & bias_mask
        rr, cc = np.nonzero(mask)
        if len(rr) == 0:
            raise ValueError("유효한 셀이 없습니다. 레이어 범위/좌표계를 확인하세요.")
        n = min(n, len(rr))
        pick = rng.choice(len(rr), size=n, replace=False)
        rr, cc = rr[pick], cc[pick]
        X = self.data[:, rr, cc].T
        xs, ys = rasterio.transform.xy(self.transform, rr, cc)
        return X, np.asarray(xs), np.asarray(ys)

    # ---------------------------------------------------------- 예측/저장
    def predict_surface(self, model, output: str = "cloglog", chunk: int = 200_000) -> np.ndarray:
        """학습된 모델을 전체 격자에 적용해 적합도 지도(2D 배열)를 만듭니다."""
        H, W = self.shape
        surf = np.full((H, W), np.nan, dtype="float32")
        rr, cc = np.nonzero(self.valid_mask)
        for s in range(0, len(rr), chunk):
            r, c = rr[s:s + chunk], cc[s:s + chunk]
            X = self.data[:, r, c].T
            surf[r, c] = model.predict(X, output=output).astype("float32")
        return surf

    def write(self, path: str, array: np.ndarray, dtype: str = "float32"):
        prof = self.profile.copy()
        prof.update(count=1, dtype=dtype, nodata=np.nan, compress="lzw")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(array.astype(dtype), 1)
        return path

    def add_layer(self, name: str, array: np.ndarray):
        """이미 만들어진 2D 배열을 새 환경 레이어로 추가합니다(예: 동위원소 보간면)."""
        arr = np.asarray(array, dtype="float32")
        if arr.shape != self.shape:
            raise ValueError(f"격자 크기가 다릅니다: {arr.shape} != {self.shape}")
        self.data = np.concatenate([self.data, arr[None, ...]], axis=0)
        self.names.append(name)
        return self

    def subset(self, names):
        """일부 레이어만 골라 새 RasterStack을 만듭니다."""
        idx = [self.names.index(n) for n in names]
        return RasterStack(self.data[idx].copy(), list(names), self.profile.copy())
