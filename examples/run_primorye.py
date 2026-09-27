# -*- coding: utf-8 -*-
"""
연해주(Primorye) 출토 돼지 유체 - 고고학 MaxEnt 예제.

들어가는 자료
  - 출토지점 : 연해주 지역 돼지 위경도 포함.shp  (20점)
  - 동위원소 : 연해주 데이터.csv  (id, 13C, 15N)  -> id로 결합
  - 환경     : SRTM DEM 90m(기준 격자) + 경사/기복 + WorldClim bio1/bio12

실행:
    .venv\\Scripts\\python.exe examples\\run_primorye.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from archaeo_sdm.pipeline import run_sdm

# ---------------------------------------------------------------- 경로 설정
import os

from archaeo_sdm import paths

OCCURRENCE = paths.PRIMORYE_SHP
ISOTOPE_CSV = paths.PRIMORYE_CSV
OUT_DIR = paths.out("primorye_pig")

# 첫 번째 레이어가 '기준 격자'가 됩니다. 여기서는 90m DEM.
ENV = {
    "고도": os.path.join(paths.DEM_DIR, "srtm_63_04.tif"),        # 동경130~135, 북위40~45
    "연평균기온_bio1": os.path.join(paths.WORLDCLIM_DIR, "wc2.1_10m_bio_1.tif"),
    "연강수량_bio12": os.path.join(paths.WORLDCLIM_DIR, "wc2.1_10m_bio_12.tif"),
}

if __name__ == "__main__":
    result = run_sdm(
        occurrence_path=OCCURRENCE,
        env_layers=ENV,
        out_dir=OUT_DIR,
        # 동위원소는 별도 CSV에 있으므로 id로 결합한 뒤 보간면으로 사용
        isotope_table=ISOTOPE_CSV,
        join_on="id",
        isotope_cols={"d13C": "13C", "d15N": "15N"},
        # DEM에서 경사·북향도·지형기복 자동 생성
        terrain_from="고도",
        terrain_include=("slope", "northness", "tri"),
        feature_types=("linear", "quadratic", "hinge"),
        beta_multiplier=2.0,     # 표본(20점)이 적으므로 정규화를 강하게
        n_background=10000,
        buffer_ratio=0.5,
    )

    print("\n=== 요약 ===")
    for k, v in result["summary"].items():
        print(f"{k}: {v}")
