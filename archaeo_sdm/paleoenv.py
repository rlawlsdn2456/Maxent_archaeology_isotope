"""
고환경(고기후·고지형) 레이어 관리.

PaleoClim(paleoclim.org)은 시기별 생물기후 변수(bio1~bio19)를 제공합니다.
이 모듈은 (1) 고고학적 시기 -> PaleoClim 시간대 매핑, (2) 이미 내려받은
PaleoClim 폴더를 읽어 레이어 사전으로 만들기, (3) 현생 WorldClim 대체 사용을
담당합니다.

중요한 한계
----------
PaleoClim의 홀로세 후기 구간은 'Late Holocene (Meghalayan, 4.2~0.3 ka)' 하나뿐입니다.
즉 청동기~요금(4.0~0.7 ka)은 전부 같은 기후면을 쓰게 되므로, 이 시기 구간에서
'시기별 기후 변화'는 PaleoClim만으로 표현되지 않습니다. 시기 간 차이는
동위원소 아이소스케이프 쪽에서 나오고, 기후면은 공통 배경으로 쓰는 것이 정직합니다.
더 세밀한 시기 변화를 넣으려면 화분(pollen)·호소퇴적 기반 지역 고기후 복원자료를
별도 래스터로 만들어 env_layers에 추가하십시오.
"""

from __future__ import annotations

import os

# PaleoClim 시간대 (이름, 시작 ka, 끝 ka, 배포 파일 접두어)
PALEOCLIM_SLICES = [
    ("LH",  "Late Holocene (Meghalayan)",   4.2,   0.3,  "LH_v1_2_5m"),
    ("MH",  "Mid Holocene (Northgrippian)", 8.326, 4.2,  "MH_v1_2_5m"),
    ("EH",  "Early Holocene (Greenlandian)", 11.7, 8.326, "EH_v1_2_5m"),
    ("YDS", "Younger Dryas",                12.9, 11.7,  "YDS_v1_2_5m"),
    ("BA",  "Bolling-Allerod",              14.7, 12.9,  "BA_v1_2_5m"),
    ("HS1", "Heinrich Stadial 1",           17.0, 14.7,  "HS1_v1_2_5m"),
    ("LGM", "Last Glacial Maximum",         21.0, 21.0,  "chelsa_LGM_v1_2B_r2_5m"),
]

# 내려받기 주소(공식 배포처). 자동 다운로드는 사용자가 명시적으로 허용할 때만 씁니다.
PALEOCLIM_BASE_URL = "http://sdmtoolbox.org/paleoclim.org/data/"


def slice_for_period(period) -> tuple[str, str]:
    """Period 객체(또는 BP 중앙값)에 맞는 PaleoClim 시간대를 반환합니다."""
    bp_mid = period.bp_mid if hasattr(period, "bp_mid") else float(period)
    ka = bp_mid / 1000.0
    for key, name, start, end, _ in PALEOCLIM_SLICES:
        if end <= ka <= start:
            return key, name
    return "LH", "Late Holocene (Meghalayan)"   # 0.3ka 이후는 가장 가까운 구간으로


def paleoclim_layers(dir_path: str, bio_numbers=(1, 4, 12, 15), prefix: str = "bio_") -> dict:
    """이미 내려받아 압축을 푼 PaleoClim 폴더에서 레이어 사전을 만듭니다.

    PaleoClim 압축 파일 안의 이름은 보통 bio_1.tif, bio_12.tif 형태입니다.
    """
    layers = {}
    for n in bio_numbers:
        for cand in (f"{prefix}{n}.tif", f"bio{n}.tif", f"bio_{n}.tif"):
            p = os.path.join(dir_path, cand)
            if os.path.exists(p):
                layers[f"bio{n}"] = p
                break
    if not layers:
        raise FileNotFoundError(
            f"{dir_path} 안에서 PaleoClim bio 파일을 찾지 못했습니다. "
            "압축을 풀었는지, 파일 이름이 bio_1.tif 형태인지 확인하세요.")
    return layers


def worldclim_layers(dir_path: str, bio_numbers=(1, 4, 12, 15),
                     pattern: str = "wc2.1_10m_bio_{n}.tif") -> dict:
    """현생 WorldClim 레이어 사전 (고기후 자료가 없을 때의 기본값)."""
    layers = {}
    for n in bio_numbers:
        p = os.path.join(dir_path, pattern.format(n=n))
        if os.path.exists(p):
            layers[f"bio{n}"] = p
    if not layers:
        raise FileNotFoundError(f"{dir_path} 안에서 WorldClim 파일을 찾지 못했습니다.")
    return layers


BIO_LABELS = {
    "bio1": "연평균기온", "bio2": "일교차", "bio4": "기온계절성",
    "bio5": "최난월최고기온", "bio6": "최한월최저기온", "bio7": "연교차",
    "bio12": "연강수량", "bio13": "최다우월강수", "bio14": "최소우월강수",
    "bio15": "강수계절성", "bio18": "최난분기강수", "bio19": "최한분기강수",
}


def download_paleoclim(slice_key: str, out_dir: str, confirm: bool = False) -> str:
    """PaleoClim 자료를 내려받습니다. confirm=True를 직접 넘겨야 실행됩니다.

    (네트워크 접속이 일어나므로 기본값에서는 아무 것도 하지 않습니다.)
    """
    match = [s for s in PALEOCLIM_SLICES if s[0] == slice_key]
    if not match:
        raise ValueError(f"알 수 없는 시간대: {slice_key}")
    url = PALEOCLIM_BASE_URL + match[0][4] + ".zip"
    if not confirm:
        return (f"[미실행] 아래 파일을 내려받아 {out_dir} 에 압축을 푸세요.\n  {url}\n"
                f"자동으로 받으려면 download_paleoclim('{slice_key}', out_dir, confirm=True)")

    import urllib.request
    import zipfile

    os.makedirs(out_dir, exist_ok=True)
    zpath = os.path.join(out_dir, os.path.basename(url))
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out_dir)
    return out_dir


# ===================== WorldClim 1.4 고기후 (CMIP5 다운스케일) =====================
# paleoclim.org 접속이 막힌 환경을 위한 실제 동작하는 대안 경로입니다.
# 중기 홀로세(약 6 ka)와 최종빙기최성기(약 21 ka)를 10분각으로 제공합니다.
WORLDCLIM_PALEO_BASE = "https://geodata.ucdavis.edu/climate/cmip5/"
WORLDCLIM_PALEO_SLICES = {
    "mid": ("중기 홀로세 (약 6 ka)", 6000),
    "lgm": ("최종빙기최성기 (약 21 ka)", 21000),
}
# 기후모델 코드: mr=MIROC-ESM, cc=CCSM4, mp=MPI-ESM-P ...
WORLDCLIM_PALEO_MODEL = "mr"

# WorldClim 1.4는 기온 계열을 10배 정수로 저장합니다(bio1=연평균기온 x10).
# 현생 WorldClim 2.1(섭씨 그대로)과 같이 쓰려면 10으로 나눠야 합니다.
WC14_TEMP_BIOS = {1, 2, 5, 6, 7, 8, 9, 10, 11}
WC14_SCALE100_BIOS = {4}     # bio4(기온계절성)는 표준편차 x100


def download_worldclim_paleo(slice_key: str, out_dir: str, model: str = WORLDCLIM_PALEO_MODEL,
                             resolution: str = "10m", confirm: bool = False) -> str:
    """중기 홀로세('mid') 또는 LGM('lgm') 생물기후 자료를 내려받아 압축을 풉니다.

    파일 크기는 각 15~18MB 정도이며, 공개 학술 미러(UC Davis)에서 받습니다.
    실제로 내려받으려면 confirm=True를 넘기십시오.
    """
    if slice_key not in WORLDCLIM_PALEO_SLICES:
        raise ValueError(f"'mid' 또는 'lgm' 중에서 고르세요 (받은 값: {slice_key})")
    fname = f"{model}{slice_key}bi_{resolution}.zip"
    url = f"{WORLDCLIM_PALEO_BASE}{slice_key}/{fname}"
    if not confirm:
        return f"[미실행] {url}\n  -> download_worldclim_paleo('{slice_key}', out_dir, confirm=True)"

    import urllib.request
    import zipfile

    dest = os.path.join(out_dir, slice_key)
    os.makedirs(dest, exist_ok=True)
    zpath = os.path.join(out_dir, fname)
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    return dest


def worldclim_paleo_layers(dir_path: str, bio_numbers=(1, 4, 12, 15),
                           model: str = WORLDCLIM_PALEO_MODEL,
                           slice_key: str = "mid") -> dict:
    """내려받은 WorldClim 고기후 폴더에서 레이어 사전을 만듭니다."""
    layers = {}
    for n in bio_numbers:
        for cand in (f"{model}{slice_key}bi{n}.tif", f"bio{n}.tif", f"bio_{n}.tif"):
            p = os.path.join(dir_path, cand)
            if os.path.exists(p):
                layers[f"bio{n}"] = p
                break
    if not layers:
        raise FileNotFoundError(f"{dir_path} 안에서 고기후 bio 파일을 찾지 못했습니다.")
    return layers


def worldclim14_scales(bio_numbers) -> dict:
    """WorldClim 1.4 -> 2.1 단위 맞추기용 나눗셈 계수."""
    out = {}
    for n in bio_numbers:
        if n in WC14_TEMP_BIOS:
            out[f"bio{n}"] = 10.0
        elif n in WC14_SCALE100_BIOS:
            out[f"bio{n}"] = 100.0
    return out


def climate_for_period(period, paleo_dir: str | None = None, worldclim_dir: str | None = None,
                       bio_numbers=(1, 4, 12, 15), slices=("mid",), verbose: bool = False):
    """시기에 맞는 기후 레이어 사전과 단위 계수를 돌려줍니다.

    쓸 수 있는 시간대(중기 홀로세 6 ka, 현생 약 0 ka) 중에서 그 시기의 BP 중앙값과
    가장 가까운 것을 고릅니다.

    주의: 청동기~요금(4.0~0.7 ka)에 딱 맞는 전지구 고기후 격자는 존재하지 않습니다.
    중기 홀로세(6 ka)와 현생 사이에서 '더 가까운 쪽'을 쓰는 근사이며, 시기 간
    기후 차이는 실제보다 과소·과대 평가될 수 있습니다. 정밀한 시기별 기후가
    필요하면 화분·호소퇴적 기반 지역 복원자료를 별도 레이어로 넣으십시오.
    """
    bp = period.bp_mid if hasattr(period, "bp_mid") else float(period)

    cands = []
    if paleo_dir:
        for key in slices:
            d = os.path.join(paleo_dir, key)
            if os.path.isdir(d):
                cands.append((key, WORLDCLIM_PALEO_SLICES[key][1], d))
    if worldclim_dir:
        cands.append(("current", -30.0, worldclim_dir))   # 1970~2000년 평년값
    if not cands:
        raise FileNotFoundError("쓸 수 있는 기후 자료가 없습니다.")

    key, slice_bp, path = min(cands, key=lambda c: abs(bp - c[1]))
    if verbose:
        gaps = ", ".join(f"{k}:{abs(bp - b):.0f}년" for k, b, _ in cands)
        print(f"      {getattr(period, 'key', bp)} (BP {bp:.0f}) -> {key} (차이 {gaps})")

    if key == "current":
        return worldclim_layers(path, bio_numbers), {}, "현생 WorldClim 2.1 (1970~2000)"
    label = f"{WORLDCLIM_PALEO_SLICES[key][0]} — 실제 시기와 {abs(bp - slice_bp):.0f}년 차이"
    return (worldclim_paleo_layers(path, bio_numbers, slice_key=key),
            worldclim14_scales(bio_numbers), label)
