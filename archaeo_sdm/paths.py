"""
자료 경로 해결기.

예전에는 예제 스크립트마다 `D:\\data of research\\...` 처럼 외장하드 경로가
직접 적혀 있어서, 외장하드가 없으면 아무 것도 실행되지 않았습니다.
이제는 이 모듈이 경로를 찾아 줍니다.

찾는 순서
  1) 환경변수 ARCHAEO_SDM_DATA 가 가리키는 폴더
  2) 이 패키지 옆의 data/inputs  (기본값 - 노트북 내장 드라이브에 복사해 둔 사본)
  3) 예전 외장하드 경로 (있으면 그대로 사용)

즉 외장하드가 연결돼 있든 아니든 똑같이 동작하고,
사본이 없을 때만 외장하드를 찾습니다.
"""

from __future__ import annotations

import os
from pathlib import Path

# archaeo-sdm/ (패키지 폴더의 부모)
PKG_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ARCHAEO_SDM_DATA", PKG_ROOT / "data"))
INPUTS = DATA_DIR / "inputs"

# 예전 외장하드 위치 (사본이 없을 때만 대체로 사용)
_EXT_ROOT = Path(r"D:\data of research")
_EXT_GIS = _EXT_ROOT / r"QGIS\ru_shp\동물 생태권 및 동위원소 분포 시각화"
_EXT_MODEL = _EXT_GIS / r"관련 논문\동물생태권 분포\서식지 적합도 모델링"


def _first(*candidates) -> str | None:
    """존재하는 첫 번째 경로를 문자열로 반환. 없으면 None."""
    for c in candidates:
        if c is None:
            continue
        p = Path(c)
        if p.exists():
            return str(p)
    return None


CHINA_DB = _first(
    INPUTS / "china" / "Isotope dataset for archaeological biological remains in China "
                       "(updated through March 2026).xlsx",
    _EXT_ROOT / r"2026\한국고고학회 세션 발표준비\동위원소 식단 분석 연구자료\중국 요하-요동반도"
              / "Isotope dataset for archaeological biological remains in China "
                "(updated through March 2026).xlsx",
)

ISOMEMO_DB = _first(
    INPUTS / "isomemo" / "dataset-excel-version.xlsx",
    _EXT_ROOT / r"지역 별 동위원소 자료\몽골 및 카자흐스탄\dataset-excel-version.xlsx",
)

ECOLOGY_XLSX = _first(
    INPUTS / "local" / "생태권용 데이터.xlsx",
    _EXT_GIS / "생태권용 데이터.xlsx",
)

PRIMORYE_SHP = _first(
    INPUTS / "local" / "연해주 지역 돼지 위경도 포함.shp",
    _EXT_GIS / "연해주 지역 돼지 위경도 포함.shp",
)

PRIMORYE_CSV = _first(
    INPUTS / "local" / "연해주 데이터.csv",
    _EXT_GIS / "연해주 데이터.csv",
)

WORLDCLIM_DIR = _first(
    INPUTS / "worldclim",
    _EXT_MODEL / "wc2.1_10m_bio",
)

DEM_DIR = _first(
    INPUTS / "dem",
    _EXT_MODEL / r"한반도 기준 연습용 자료 만들기\한반도 SRTM\SRTM 5X5\srtm_63_04",
)

PALEOCLIM_DIR = str(DATA_DIR / "paleoclim")
GAZETTEER_MANUAL = str(DATA_DIR / "gazetteer_manual.csv")
OUTPUTS = str(PKG_ROOT / "outputs")


def out(*parts) -> str:
    """결과 폴더 아래 경로를 만들어 줍니다. out('northeast_asia') 처럼 씁니다."""
    p = Path(OUTPUTS).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def status() -> dict:
    """자료가 어디에서 잡혔는지 보여 줍니다."""
    items = {
        "중국 동위원소 DB": CHINA_DB,
        "IsoMemo/CIMA DB": ISOMEMO_DB,
        "직접 정리한 표": ECOLOGY_XLSX,
        "연해주 점 자료(SHP)": PRIMORYE_SHP,
        "연해주 동위원소 CSV": PRIMORYE_CSV,
        "현생 WorldClim": WORLDCLIM_DIR,
        "SRTM DEM": DEM_DIR,
        "고기후": PALEOCLIM_DIR if Path(PALEOCLIM_DIR).exists() else None,
    }
    return items


def print_status():
    print(f"패키지 위치 : {PKG_ROOT}")
    print(f"자료 폴더   : {DATA_DIR}")
    for name, path in status().items():
        if path is None:
            mark, where = "[없음]", "- 찾지 못함"
        elif str(path).lower().startswith("d:"):
            mark, where = "[외장]", path
        else:
            mark, where = "[내장]", path
        print(f"  {mark} {name:<20} {where}")


if __name__ == "__main__":
    print_status()
