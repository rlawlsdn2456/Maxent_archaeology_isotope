# Archaeo-SDM QGIS 플러그인 설치법

1. 이 폴더(`archaeo_sdm`)를 통째로 QGIS 플러그인 폴더에 복사합니다.
   - Windows: `C:\Users\user\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\`
   - QGIS에서 `설정 > 사용자 프로필 > 활성 프로필 폴더 열기` 로 바로 갈 수 있습니다.
2. QGIS를 다시 시작하고 `플러그인 > 플러그인 관리 및 설치 > 설치됨` 에서
   **Archaeo-SDM** 을 체크합니다.
3. 툴바 아이콘이나 `플러그인 > Archaeo-SDM > Archaeo-SDM 분석…` 으로 창을 엽니다.
4. 첫 실행에서 두 가지만 지정하면 됩니다.
   - **파이썬 실행 파일**: `C:\Users\user\archaeo-sdm\.venv\Scripts\python.exe`
   - **패키지 폴더**: `C:\Users\user\archaeo-sdm`

   나머지 자료 경로와 설정은 다음 실행 때 자동으로 기억됩니다.
   자료 경로는 비워 두어도 됩니다 — 비어 있으면 `archaeo_sdm/paths.py` 가
   내장 드라이브의 `data/inputs` 사본을 찾아 씁니다.

QGIS 내장 파이썬에는 아무 것도 설치하지 않습니다. 분석은 위에서 지정한
가상환경 파이썬을 별도 프로세스로 실행해서 수행하고, QGIS는 결과 GeoTIFF만
레이어로 불러옵니다(색상표 자동 적용: 아이소스케이프=발산형, 변화면=0 중심 발산형,
적합도=viridis).

`현재 지도 범위 사용` 버튼을 누르면 QGIS 화면에 보이는 범위가 분석 범위로 들어갑니다.
