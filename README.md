# archaeo-sdm — 고고학용 MaxEnt 종분포모델

유적에서 출토된 인골·동물유체의 **위치(경위도)** + **탄소·질소 동위원소 값** + **고기후/고지형 GIS 레이어**를
넣으면, 서식지 적합도(habitat suitability) 지도와 통계 결과를 만들어 주는 파이썬 툴킷입니다.

기존 MaxEnt(자바 프로그램)와 달리 전 과정이 코드로 되어 있어서
동위원소 면(isoscape) 추가, 시기별 비교, 시뮬레이션 같은 고고학 특화 분석을 자유롭게 붙일 수 있습니다.

---

> **자료 안내:** 이 저장소에는 코드만 들어 있습니다. 입력 자료와 결과물은 올리지 않으며,
> 필요한 자료와 구하는 곳은 [DATA_NOTICE.md](DATA_NOTICE.md) 에 정리해 두었습니다.

## 0. 설치 위치와 자료 경로

이 프로젝트는 노트북 **내장 드라이브**에 있습니다 (외장하드 없이 동작합니다).

```
C:\Users\user\archaeo-sdm\
  archaeo_sdm\      코드
  data\inputs\      입력 자료 사본 (중국DB·IsoMemo·연해주·WorldClim·SRTM DEM)
  data\paleoclim\   고기후 (mid=중기 홀로세, lgm=최종빙기최성기)
  outputs\          결과
  .venv\            가상환경
```

경로는 코드에 직접 적지 않고 `archaeo_sdm/paths.py` 가 찾아 줍니다.
찾는 순서는 ① 환경변수 `ARCHAEO_SDM_DATA` → ② `<프로젝트>/data/inputs` → ③ 예전 외장하드(D:) 입니다.

지금 어디에서 자료를 읽고 있는지 확인하려면:

```bash
.venv/Scripts/python.exe -m archaeo_sdm.paths
```

`[내장]` 이면 내장 드라이브 사본을, `[외장]` 이면 D: 를 쓰고 있다는 뜻입니다.
외장하드의 원본은 지우지 않고 그대로 두었으므로, 원본을 고쳤을 때는
`data/inputs` 사본도 다시 복사해야 반영됩니다.

## 1. 설치

이 폴더 안에 이미 가상환경(`.venv`)이 만들어져 있습니다.

```bash
.venv/Scripts/python.exe -m pip install numpy pandas scikit-learn matplotlib rasterio geopandas shapely pyproj openpyxl
```

## 2. 가장 빠른 사용법

```python
from archaeo_sdm.pipeline import run_sdm

run_sdm(
    occurrence_path=r"연해주 지역 돼지 위경도 포함.shp",   # .shp / .csv / .xlsx 모두 가능
    env_layers={                                          # 환경 레이어 (GeoTIFF)
        "연평균기온": r"...\wc2.1_10m_bio_1.tif",
        "연강수량":   r"...\wc2.1_10m_bio_12.tif",
    },
    isotope_cols={"d13C": "13C", "d15N": "15N"},          # 있으면 동위원소 면도 자동 생성
    out_dir="outputs/내분석",
)
```

`out_dir`에 다음이 저장됩니다.

| 파일 | 내용 |
|---|---|
| `suitability_cloglog.tif` | 서식지 적합도 지도 (QGIS에서 바로 열림) |
| `isoscape_d13C.tif`, `isoscape_d15N.tif` | 동위원소 보간면 |
| `map_suitability.png` | 적합도 지도 그림 |
| `response_curves.png` | 변수별 반응 곡선 |
| `variable_importance.png/.csv` | 변수 중요도 |
| `summary.json` | AUC 등 요약 통계 |

## 3. 모듈 구성

| 파일 | 역할 |
|---|---|
| `features.py` | MaxEnt 특징 변환 (linear / quadratic / product / hinge / threshold) |
| `maxent.py` | MaxEnt 모델 학습·예측·AUC·반응곡선·변수중요도 |
| `raster.py` | GeoTIFF 정렬, 점 값 추출, 배경점 추출, 예측면 저장 |
| `occurrence.py` | SHP/CSV/XLSX 점 자료 읽기 (한글 인코딩 자동 처리) |
| `isotope.py` | 동위원소 IDW 보간면 생성, δ13C-δ15N 집단 분류 |
| `plotting.py` | 지도·곡선·중요도·동위원소 산점도 |
| `pipeline.py` | 위 과정을 한 번에 실행하는 `run_sdm()` |

## 4. 모델에 대한 짧은 설명

MaxEnt는 "출토지점의 환경 조건"과 "연구지역 전체의 환경 조건(배경점)"을 비교해서,
환경변수의 함수로 표현되는 확률분포

```
p(x) = exp(λ·f(x)) / Z
```

를 학습합니다. 이 구현은 배경점에 큰 가중치를 준 **L1 정규화 로지스틱 회귀(IWLR)** 로 λ를 추정하는데,
이는 원래 MaxEnt와 수학적으로 동일합니다(Fithian & Hastie 2013; Renner et al. 2015).
출력은 Phillips et al. (2017)의 **cloglog** 변환을 사용합니다.

### 고고학에 쓸 때 주의할 점
- 출토지점은 "동물이 살던 곳"이 아니라 "사람이 버린 곳"입니다. 발굴 편향(조사가 많이 된 지역)이
  결과를 왜곡할 수 있으므로, 필요하면 `sample_background(bias_mask=...)`로 조사 이력이 있는
  지역에서만 배경점을 뽑는 *target-group background* 방식을 쓰세요.
- 표본이 적으면(<20) `beta_multiplier`를 1.5~3으로 올려 과적합을 막습니다.
- 동위원소 보간면은 시료가 적을수록 신뢰도가 떨어집니다. `idw_surface(max_distance=...)`로
  시료에서 먼 지역은 결측 처리하는 것을 권합니다.

## 5. 시기별(편년별) 분석 — v0.2에서 추가된 부분

대규모 동위원소 DB를 넣으면 **고고학적 시기 × 분류군 × 동위원소 프록시**로 갈라서
아이소스케이프·변화면·MaxEnt를 한 번에 만듭니다.

| 파일 | 역할 |
|---|---|
| `chronology.py` | 편년 정의(`MANCHURIA_PERIODS`)와 시기 자동 배정 — 절대연대 우선, 없으면 문화명 키워드 |
| `isotope_db.py` | 중국 DB(4시트)·IsoMemo 계열 DB를 하나의 표준 표로 병합, 지역·C:N 품질 필터 |
| `isoscape.py` | 유적 단위 평균화 → IDW 아이소스케이프 + 지지거리(km) 마스크, Δ면, Schoener's D |
| `terrain.py` | DEM에서 경사·사면향(북향/동향 분해)·지형기복 생성 |
| `paleoenv.py` | PaleoClim 시간대 매핑·로딩, WorldClim 대체, 다운로드 도우미 |
| `temporal.py` | `run_temporal_study()` — 위 전부를 묶은 실행기 |

```python
from archaeo_sdm.temporal import run_temporal_study

run_temporal_study(
    df, env_layers, out_dir, bbox=(110, 35, 136, 52),
    proxies=("d13C_coll", "d15N_coll", "d13C_ap", "d18O_ap"),
    iso_taxa=("사람", "돼지", "소", "양·염소"),
    sdm_taxa=("돼지", "사람", "소"),
)
```

SDM은 시기마다 **두 모델**을 만듭니다.
- `sdm_*_env.tif` : 고환경(기후) 변수만 → 연구지역 전체를 예측
- `sdm_*_env_iso.tif` : 여기에 그 시기의 아이소스케이프를 추가 → 동위원소 지지범위 안만 예측

두 모델의 AUC는 **직접 비교하면 안 됩니다**. env+iso는 예측 범위가 좁아 배경점이
균질해지므로 AUC가 구조적으로 낮게 나옵니다.

### 해석할 때 반드시 확인할 것
- **시기별 유적 분포가 다르면 '시간 변화'가 아니라 '공간 변화'일 수 있습니다.**
  예를 들어 청동기 유적이 화북 조·기장 지대에 몰려 있고 발해·요금 유적이 만주 삼림대에
  몰려 있으면, δ13C 하락은 사육방식 변화가 아니라 표본 지역 이동을 볼 가능성이 큽니다.
  `isoscape_summary.csv`의 유적수·좌표 분포를 항상 함께 보십시오.
- 아이소스케이프는 `max_distance_km` 밖을 결측 처리합니다. 섬처럼 떨어진 덩어리로
  보인다면 그 시기 시료가 그만큼 희소하다는 뜻입니다.
- PaleoClim은 4.2~0.3 ka를 **한 장(Late Holocene)** 으로만 제공합니다. 청동기~요금은
  전부 같은 기후면을 쓰게 되므로, 시기별 기후 변화를 보려면 화분·호소퇴적 기반
  지역 복원자료를 별도 레이어로 넣어야 합니다.

## 6. v0.3에서 추가된 것

| 파일 | 역할 |
|---|---|
| `gazetteer.py` | 유적 이름 → 좌표 사전. 좌표 출처와 정확도(정확/이름대조/수기/지역근사)를 함께 기록 |
| `local_data.py` | 직접 정리한 엑셀 읽기 — 열이 밀린 행 자동 교정, `1200-400 BCE`·`398–494 AD`·세기 표기 연대 해석 |
| `regional.py` | 소지역(zone) 구분과 **같은 지역 안에서의 시기 비교** (Mann-Whitney·Kruskal-Wallis) |
| `validation.py` | **공간 블록 교차검증**(k-means/바둑판), TSS, 무작위 기준선 |
| `paleoenv.py` | 고기후 자동 다운로드 + 시기별 자동 배정(가장 가까운 시간대) |
| `cli.py` | JSON 설정 하나로 전체 실행 — 브라우저 UI와 QGIS 플러그인이 이것을 호출 |
| `webapp/` | 브라우저 UI (Flask + Leaflet) |
| `r-package/` | R 인터페이스 (reticulate) |
| `qgis_plugin/` | QGIS 플러그인 |

### 명령줄
```bash
python -m archaeo_sdm.cli template > config.json   # 설정 뼈대
python -m archaeo_sdm.cli run config.json          # 실행
python -m archaeo_sdm.cli download-paleo mid ./data/paleoclim
python -m archaeo_sdm.cli web                      # 브라우저 UI (http://127.0.0.1:8765)
```

### 고기후 자료
`paleoclim.org` 는 접속이 막히는 환경이 있어, 실제로 내려받아지는
**WorldClim 1.4 CMIP5 다운스케일 고기후**(UC Davis 미러)를 기본 경로로 씁니다.
중기 홀로세(약 6 ka)와 LGM(약 21 ka), 10분각. 기온 계열은 10배 정수로 저장되어 있어
`scales` 로 자동 보정합니다.

시기별 배정은 '가장 가까운 시간대' 규칙입니다.
EBA·LBA → 중기 홀로세, EIA 이후 → 현생 WorldClim 2.1.
**청동기~요금(4.0~0.7 ka)에 딱 맞는 전지구 고기후 격자는 존재하지 않습니다.**
6 ka와 현생 사이의 근사이므로, 시기 간 기후 차이는 실제와 다를 수 있습니다.

### 같은 지역 안에서의 시기 비교 (가장 중요한 사용법)
```python
from archaeo_sdm.regional import assign_zone, zone_period_table, within_zone_change
df["zone"] = assign_zone(df)
within_zone_change(df[df.taxon_group == "돼지"], "d13C_coll", PERIOD_ORDER)
```
시기마다 조사된 지역이 다르면 '시간 변화'가 아니라 '공간 이동'을 보게 됩니다.
`zone_traj_*.png` 의 **같은 색 선 위에서만** 시기 비교가 유효합니다.

### 공간 블록 교차검증
```python
from archaeo_sdm.validation import spatial_block_cv, null_model_auc
spatial_block_cv(Xp, (px, py), Xb, (bx, by), n_folds=4, method="kmeans")
```
학습에 쓰지 않은 '다른 지역'에서 성능을 재기 때문에, 보통의 AUC보다 훨씬 낮게 나옵니다.
낮게 나오는 것이 정상이며, `무작위기준선` 보다 확실히 높지 않으면 그 모델은
환경을 배운 것이 아니라 조사 분포를 외운 것입니다.

## 7. SDM Studio — MaxEnt 너머의 모델 실험실 (v0.4)

```bash
.venv/Scripts/python.exe -m archaeo_sdm.cli studio      # http://127.0.0.1:8766
```

기존 브라우저 UI(8765)가 MaxEnt 전체 파이프라인이라면, Studio는 여러 방법을 나란히 시험하는 곳입니다.
표준화된 자료(`outputs/*/dataset_*.csv`)가 있어야 하므로 `examples/run_northeast_asia.py` 를 먼저 한 번 실행하십시오.

| 탭 | 하는 일 | 모듈 |
|---|---|---|
| ① 모델 비교 | BIOCLIM·Mahalanobis·GLM·GAM·RF·BRT·MaxEnt를 공간 블록 교차검증으로 비교, 기준 통과 모델만 가중 앙상블 | `models.py` |
| ② 기후 시나리오 | LGM → 중기 홀로세 → 현생 투영, 면적·중심 이동, 외삽(MESS) 지도 | `scenarios.py` |
| ③ 분산 시뮬레이션 | 이동 거리 제한 하의 범위 확산(셀룰러 오토마타), GIF | `scenarios.py` |
| ④ 동위원소 니치 | SEAc(부트스트랩 CI)·Layman 지표·표준타원 겹침 | `isoniche.py` |
| 도구 안내 | 모델·도구 조사 문서 | `docs/SDM_tools_survey.md` |

유적 수가 각 알고리즘의 최소 기준(BIOCLIM 3 · Mahalanobis 4 · GLM/MaxEnt 5 · GAM 8 · RF 10 · BRT 15)보다
적으면 그 알고리즘은 자동으로 빠집니다. 계산부(`studio/engine.py`)는 화면 없이 스크립트에서도 쓸 수 있습니다.

## 8. 앞으로 붙일 것

1. 시기별 지역 고기후 복원 레이어(화분·호소퇴적) 결합
2. 발굴 편향 보정(target-group background)
3. 아파타이트(C·O) 자료 보강 — 만주권은 현재 거의 비어 있음
4. 미해결 유적 좌표 채우기 (`gazetteer_TODO.csv`)
