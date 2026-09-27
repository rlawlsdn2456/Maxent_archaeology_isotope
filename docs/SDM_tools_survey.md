# MaxEnt 너머: 고고학 동물유체·동위원소 자료를 위한 종분포·니치 모델 조사

작성: 2026-09 · archaeo-sdm 프로젝트

## 1. 왜 MaxEnt 하나로는 부족한가

MaxEnt는 출토지(존재)와 배경점만으로 학습할 수 있어 고고학에 널리 쓰이지만,
우리 자료에서 네 가지 한계가 드러났습니다.

| 문제 | 우리 자료에서 확인된 것 |
|---|---|
| **소표본** | 동물유체 동위원소 유적은 만주·요서·연해주를 합쳐 18곳. 시기×분류군으로 나누면 1–5곳 |
| **공간 자기상관** | 일반 AUC 0.83–0.93 → 공간 블록 교차검증 0.25–0.70으로 급락 |
| **발굴 편향** | 출토지는 "동물이 살던 곳"이 아니라 "조사가 된 곳"을 반영 |
| **시간 투영** | 한 시점 모델을 다른 시점 기후에 적용하면 학습 범위 밖(외삽)이 생김 |

이 한계를 보완하는 모델은 크게 **① 다른 SDM 알고리즘 ② 앙상블 ③ 동적(분산) 모델
④ 동위원소 니치 모델**로 나뉩니다.

## 2. 종분포 알고리즘 비교

| 계열 | 모델 | 필요 자료 | 소표본 | 장점 | 한계 |
|---|---|---|---|---|---|
| 환경 포락선 | **BIOCLIM** (Busby 1991) | 존재만 | 매우 강함 | 단순·보수적, 외삽을 막음 | 변수 간 상관 무시, 사각형 니치 |
| | **DOMAIN** (Carpenter et al. 1993) | 존재만 | 강함 | 가장 가까운 출토지와의 유사도 | 이상치에 민감 |
| | **Mahalanobis 거리** (Farber & Kadmon 2003) | 존재만 | 강함 | 변수 상관 반영, 타원형 니치 | 니치가 한 봉우리라고 가정 |
| 회귀 | **GLM** (로지스틱+2차항) | 존재+배경 | 보통 | 반응 모양 해석이 쉬움 | 복잡한 반응은 못 잡음 |
| | **GAM** | 존재+배경 | 보통 | 곡선 반응을 자유롭게 | 표본이 적으면 과적합 |
| 점과정 | **MaxEnt / PPM** (Renner & Warton 2013) | 존재+배경 | 보통 | 발굴편향 공변량을 명시적으로 넣을 수 있음(PPM) | 정규화 설정에 민감 |
| 기계학습 | **Random Forest** (down-sampled, Valavi et al. 2021) | 존재+배경 | 약함 | 비선형·상호작용 | 학습 AUC가 과하게 높음 |
| | **BRT / GBM** (Elith et al. 2008) | 존재+배경 | 약함 | 예측력 우수 | 표본을 가장 많이 요구 |
| | SVM, 신경망 | 존재+배경 | 약함 | | 고고학 규모에선 권하지 않음 |
| 다종 | **HMSC** 공동종분포모델 (Ovaskainen et al. 2017) | 여러 종 동시 | 약함 | 종 간 연관(예: 돼지–사람) 추정 | 계산량 큼, 베이즈 |
| 점유 | Occupancy (unmarked) | 반복 조사 | — | 탐지율 보정 | **고고학에는 반복 조사가 없어 부적합** |

### 표본 규모에 따른 선택 기준 (이 프로젝트 권장)

| 유적 수 | 쓸 수 있는 것 |
|---|---|
| 3–4곳 | BIOCLIM, Mahalanobis (모델이라기보다 환경 범위 요약) |
| 5–9곳 | + GLM, MaxEnt (강한 정규화) |
| 10–19곳 | + GAM, Random Forest |
| 20곳 이상 | + BRT, 앙상블, 공간 블록 교차검증이 안정적 |

## 3. 앙상블과 모델 조율

여러 알고리즘의 예측을 합치면 한 모델의 버릇에 덜 휘둘립니다(Araújo & New 2007).
다만 **약한 모델을 평균한다고 좋아지지는 않으므로**, 공간 교차검증 성능이 기준을 넘는
모델만 넣어야 합니다.

| 도구 | 언어 | 특징 |
|---|---|---|
| **biomod2** (Thuiller et al.) | R | 가장 널리 쓰이는 앙상블 플랫폼. 10여 개 알고리즘 |
| **sdm** (Naimi & Araújo 2016) | R | 앙상블·병렬 처리 |
| **SSDM** (Schmitt et al. 2017) | R | 다종 누적(stacked) SDM, GUI 포함 |
| **ENMeval** (Muscarella et al. 2014; Kass et al. 2021) | R | MaxEnt 정규화·특징 조율, 공간 블록 CV |
| **blockCV** (Valavi et al. 2019) | R | 공간 블록 교차검증 전용 |
| **Wallace** (Kass et al. 2018, 2023) | R/Shiny | **브라우저 기반 SDM GUI** — 우리 Studio와 가장 비슷한 형태 |
| **elapid** (Anderson) | Python | MaxEnt·NicheEnvelope의 파이썬 구현 |
| dismo / predicts | R | BIOCLIM·DOMAIN·Mahalanobis 고전 알고리즘 |

## 4. 환경 변화에 따른 '이동'을 다루는 모델

일반 SDM은 "그 기후에서 살 수 있는가"만 답합니다. **실제로 그곳에 도달했는가**는
이동 능력과 시간이 결정합니다.

| 모델 | 방식 | 필요 정보 | 고고학 활용 |
|---|---|---|---|
| **MigClim** (Engler & Guisan 2009) | 셀룰러 오토마타 + 분산 거리 | 적합도 시계열, 분산 거리 | 빙기 이후 재확산 경로 |
| **KISSMig** (Nobis & Normand 2014) | 단순 확산 | 적합도 시계열 | 피난처(refugia) 추정 |
| **RangeShifter** (Bocedi et al. 2014) | 개체 기반 인구·분산 | 번식·생존 모수 | 야생 종의 개체군 동태 |
| **NicheMapR** (Kearney & Porter 2017) | 생물물리(열수지) | 체온·대사 모수 | 기후 한계를 기작으로 설명 |
| **CLIMEX** | 성장·스트레스 지수 | 생리 모수 | 해충·작물 중심 |
| **NetLogo** 등 행위자 기반 모델 | 개체·집단 의사결정 | 행동 규칙 | **목축민·가축 이동, 교역** |
| **Circuitscape / Omniscape** (McRae) | 전기회로 연결성 | 저항 지도 | 유적 간 이동 회랑 |
| QGIS/GRASS `r.cost`, `r.walk` | 최소비용 경로 | 비용 지도 | 유적 간 경로 |

가축(돼지·소·양)은 사람이 옮기므로 **분산 모델의 '이동 거리'는 사람의 이동·교류로 해석**해야
합니다. 야생종(사슴·멧돼지)에는 생물학적 분산 거리를 쓸 수 있습니다.

## 5. 동위원소 니치와 아이소스케이프 도구

유적이 적어도 **개체가 많으면** 쓸 수 있어, 우리 자료에 가장 잘 맞는 분야입니다.

| 도구 | 언어 | 하는 일 |
|---|---|---|
| **SIBER** (Jackson et al. 2011) | R | 표준타원(SEAc)·Layman 지표, 베이즈 비교 |
| **nicheROVER** (Swanson et al. 2015) | R | 다차원 니치 크기·겹침 확률(방향성 있음) |
| **MixSIAR** (Stock et al. 2018) / **simmr** | R | 먹이원 혼합 비율(C3 식물·C4 곡물·해양 자원) |
| **IsoriX** (Courtiol et al. 2019) | R | 아이소스케이프 + 기원지 추정(주로 산소·수소·스트론튬) |
| **assignR** (Ma et al. 2020) | R | 개체의 출신 지역 확률 지도 |
| **IsoMemo / Pandora** | 웹 | 고고 동위원소 DB·공간 모델링 플랫폼 |
| **isocat** | R | 아이소스케이프 기반 기원 추정 |

δ15N 기저값은 지역(건조도·토양·해양)에 따라 다르므로, 지역 간 비교에서는
같은 지역 초식동물 값을 기준선으로 함께 제시해야 합니다.

## 6. SDM Studio에 구현한 것

| 기능 | 구현 | 대응하는 외부 도구 |
|---|---|---|
| 7개 알고리즘 비교 | BIOCLIM, Mahalanobis, GLM, GAM, RF, BRT, MaxEnt | dismo, biomod2 |
| 공간 블록 교차검증 + 무작위 기준선 | k-means 블록 | blockCV, ENMeval |
| 가중 앙상블 | 공간CV AUC ≥ 기준만 (AUC−0.5 가중) | biomod2 |
| 기후 시나리오 투영 | LGM → 중기 홀로세 → 현생 | biomod2 projection |
| 외삽 경고 | MESS 지도 | dismo::mess |
| 범위 변화 지표 | 면적 증감·유지·중심 이동 거리/방향 | biomod2 range size |
| 분산 시뮬레이션 | 셀룰러 오토마타, GIF 애니메이션 | MigClim, KISSMig |
| 동위원소 니치 | SEA·SEAc(부트스트랩 CI)·Layman·타원 겹침 | SIBER |

### 구현하지 않은 것 (다음 단계 후보)

- **PPM + 발굴편향 공변량** — 조사 밀도 지도를 만들 수 있으면 가장 효과가 큼
- **MixSIAR형 혼합모델** — C3/C4/해양 기여율을 개체별로 추정 (먹이원 기준값 필요)
- **HMSC** — 사람·돼지·개를 동시에 모델링해 종 간 연관 추정
- **최소비용 경로·연결성** — 유적 간 이동 회랑
- **행위자 기반 목축 모델** — 가축 이동을 사람의 결정으로 모델링

## 7. 우리 자료로 해 본 첫 비교 (초기철기 사람, 58유적)

| 모델 | 학습 AUC | 공간CV AUC | 해석 |
|---|---|---|---|
| Mahalanobis | 0.909 | **0.750** | 가장 좋음 — 니치가 한 덩어리라는 가정이 맞는 편 |
| Random Forest | 0.999 | 0.702 | 학습 0.999는 과적합 신호 |
| BRT | 0.963 | 0.673 | |
| MaxEnt | 0.808 | 0.623 | |
| GLM / GAM / BIOCLIM | 0.87–0.90 | 0.62 | |
| 무작위 기준선 | | 0.556 | 이보다 확실히 높아야 의미 있음 |

같은 자료라도 알고리즘에 따라 공간 일반화 성능이 0.62–0.75로 갈립니다.
**단일 알고리즘 결과만 보고하면 그 알고리즘의 버릇을 결과로 착각할 수 있습니다.**

## 참고문헌 (핵심)

- Araújo, M.B. & New, M. (2007) Ensemble forecasting of species distributions. *TREE* 22.
- Busby, J.R. (1991) BIOCLIM — a bioclimate analysis and prediction system.
- Elith, J. et al. (2008) A working guide to boosted regression trees. *J. Anim. Ecol.* 77.
- Elith, J. et al. (2010) The art of modelling range-shifting species. *MEE* 1.
- Engler, R. & Guisan, A. (2009) MigClim. *Diversity Distrib.* 15.
- Farber, O. & Kadmon, R. (2003) Mahalanobis distance in habitat suitability modelling. *Ecol. Model.* 160.
- Jackson, A.L. et al. (2011) Comparing isotopic niche widths (SIBER). *J. Anim. Ecol.* 80.
- Kass, J.M. et al. (2018) Wallace. *MEE* 9.
- Muscarella, R. et al. (2014) ENMeval. *MEE* 5.
- Renner, I.W. & Warton, D.I. (2013) Equivalence of MAXENT and Poisson point process models. *Biometrics* 69.
- Roberts, D.R. et al. (2017) Cross-validation strategies for data with structure. *Ecography* 40.
- Valavi, R. et al. (2019) blockCV. *MEE* 10.
- Valavi, R. et al. (2021) Modelling species presence-only data with random forests. *Ecography* 44.
