# 자료 안내 (DATA NOTICE)

이 저장소에는 **코드만** 들어 있습니다. 입력 자료(`data/`)와 분석 결과(`outputs/`)는
저장소에 올리지 않고 연구자 컴퓨터에만 보관합니다 (`.gitignore` 로 제외).

코드를 실행하려면 아래 자료를 각자 `data/inputs/` 에 넣으십시오.
`python -m archaeo_sdm.paths` 로 어떤 자료가 잡혀 있는지 확인할 수 있습니다.

| 넣을 위치 | 자료 | 구하는 곳 |
|---|---|---|
| `data/inputs/china/` | Isotope dataset for archaeological biological remains in China | 원 편찬자 배포본 (인용 필수) |
| `data/inputs/isomemo/` | IsoMemo / CIMA 유라시아 동위원소 데이터셋 | IsoMemo 이니셔티브 (인용 필수) |
| `data/inputs/local/` | 연구자 본인 정리표·SHP | 비공개 — 저장소에 없음 |
| `data/inputs/worldclim/` | WorldClim 2.1, 10분각, bio1–19 | worldclim.org (Fick & Hijmans 2017) |
| `data/inputs/dem/` | SRTM 90 m DEM | CGIAR-CSI (Jarvis et al. 2008) |
| `data/paleoclim/` | 중기 홀로세·LGM 고기후 | `python -m archaeo_sdm.cli download-paleo mid ./data/paleoclim` |

제3자 데이터베이스는 재배포하지 않습니다. 필요한 사람에게는 원 배포처를 안내하십시오.
