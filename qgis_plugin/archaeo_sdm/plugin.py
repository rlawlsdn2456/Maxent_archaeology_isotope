# -*- coding: utf-8 -*-
"""
QGIS 플러그인 본체.

QGIS 내장 파이썬에는 우리 패키지가 설치돼 있지 않을 수 있으므로,
분석은 archaeo-sdm 가상환경의 python.exe 를 별도 프로세스로 실행합니다.
끝나면 결과 GeoTIFF를 QGIS 레이어로 자동으로 불러옵니다.
"""

import json
import os
import subprocess
import tempfile

from qgis.core import (Qgis, QgsColorRampShader, QgsProject, QgsRasterLayer,
                       QgsRasterShader, QgsSingleBandPseudoColorRenderer)
from qgis.PyQt.QtCore import QProcess, QSettings, Qt
from qgis.PyQt.QtGui import QColor, QIcon
from qgis.PyQt.QtWidgets import (QAction, QCheckBox, QComboBox, QDialog, QFileDialog,
                                 QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                                 QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout)

SETTINGS_PREFIX = "archaeo_sdm/"


def _row(parent, placeholder, pick="file", filt="모든 파일 (*.*)"):
    """한 줄짜리 '경로 입력 + 찾아보기' 위젯."""
    box = QHBoxLayout()
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    btn = QPushButton("…")
    btn.setFixedWidth(30)

    def browse():
        if pick == "dir":
            p = QFileDialog.getExistingDirectory(parent, placeholder)
        else:
            p, _ = QFileDialog.getOpenFileName(parent, placeholder, "", filt)
        if p:
            edit.setText(p)

    btn.clicked.connect(browse)
    box.addWidget(edit)
    box.addWidget(btn)
    return box, edit


class ArchaeoSdmDialog(QDialog):
    def __init__(self, iface):
        super().__init__(iface.mainWindow())
        self.iface = iface
        self.proc = None
        self.out_dir = None
        self.setWindowTitle("Archaeo-SDM — 고고학 동위원소·종분포 분석")
        self.setMinimumWidth(640)
        s = QSettings()
        lay = QVBoxLayout(self)

        # ---------------------------------------------------------- 실행환경
        g0 = QGroupBox("실행 환경")
        f0 = QFormLayout(g0)
        b, self.py = _row(self, "archaeo-sdm/.venv/Scripts/python.exe", "file",
                          "python.exe (python.exe)")
        f0.addRow("파이썬 실행 파일", b)
        b, self.pkg = _row(self, "archaeo_sdm 이 들어 있는 폴더", "dir")
        f0.addRow("패키지 폴더", b)
        lay.addWidget(g0)

        # -------------------------------------------------------------- 자료
        g1 = QGroupBox("자료")
        f1 = QFormLayout(g1)
        self.inputs = {}
        for key, label, kind, filt in [
            ("china_db", "중국 동위원소 DB", "file", "Excel (*.xlsx)"),
            ("isomemo_db", "IsoMemo/CIMA DB", "file", "Excel (*.xlsx)"),
            ("ecology_xlsx", "직접 정리한 표", "file", "Excel (*.xlsx)"),
            ("point_shapefile", "좌표 있는 점 자료", "file", "Shapefile (*.shp)"),
            ("point_csv", "점 자료 동위원소 CSV", "file", "CSV (*.csv)"),
            ("gazetteer_manual", "유적 좌표 사전", "file", "CSV (*.csv)"),
        ]:
            b, e = _row(self, label, kind, filt)
            f1.addRow(label, b)
            self.inputs[key] = e
        lay.addWidget(g1)

        # ------------------------------------------------------------ 고환경
        g2 = QGroupBox("고환경")
        f2 = QFormLayout(g2)
        b, self.wc = _row(self, "현생 WorldClim 폴더", "dir")
        f2.addRow("WorldClim 폴더", b)
        b, self.pc = _row(self, "고기후 폴더 (mid/lgm 포함)", "dir")
        f2.addRow("고기후 폴더", b)
        self.bio = QLineEdit("1,4,12,15")
        f2.addRow("생물기후 변수", self.bio)
        self.bbox = QLineEdit("105,33,141,56")
        f2.addRow("범위 (서,남,동,북)", self.bbox)
        self.btn_extent = QPushButton("현재 지도 범위 사용")
        self.btn_extent.clicked.connect(self.use_canvas_extent)
        f2.addRow("", self.btn_extent)
        lay.addWidget(g2)

        # ------------------------------------------------------------ 분석
        g3 = QGroupBox("분석 설정")
        f3 = QFormLayout(g3)
        self.proxy = QComboBox()
        self.proxy.addItem("콜라겐 δ13C + δ15N", ["d13C_coll", "d15N_coll"])
        self.proxy.addItem("콜라겐 + 아파타이트 전체",
                           ["d13C_coll", "d15N_coll", "d13C_ap", "d18O_ap"])
        self.proxy.addItem("콜라겐 δ13C만", ["d13C_coll"])
        self.proxy.addItem("콜라겐 δ15N만", ["d15N_coll"])
        f3.addRow("동위원소 프록시", self.proxy)
        self.iso_taxa = QLineEdit("사람,돼지,소,양·염소")
        f3.addRow("아이소스케이프 분류군", self.iso_taxa)
        self.sdm_taxa = QLineEdit("돼지,사람")
        f3.addRow("MaxEnt 분류군", self.sdm_taxa)
        self.dist = QSpinBox()
        self.dist.setRange(10, 2000)
        self.dist.setValue(350)
        self.dist.setSuffix(" km")
        f3.addRow("아이소스케이프 지지거리", self.dist)
        self.cv = QCheckBox("공간 블록 교차검증 수행")
        self.cv.setChecked(True)
        f3.addRow("", self.cv)
        b, self.out = _row(self, "결과를 저장할 폴더", "dir")
        f3.addRow("출력 폴더", b)
        lay.addWidget(g3)

        # ------------------------------------------------------------- 실행
        run_box = QHBoxLayout()
        self.btn_run = QPushButton("분석 실행")
        self.btn_run.clicked.connect(self.run)
        self.btn_close = QPushButton("닫기")
        self.btn_close.clicked.connect(self.reject)
        run_box.addWidget(self.btn_run)
        run_box.addWidget(self.btn_close)
        lay.addLayout(run_box)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(160)
        lay.addWidget(QLabel("진행 상황"))
        lay.addWidget(self.log)

        self._restore(s)

    # ------------------------------------------------------------- 설정 저장
    def _fields(self):
        d = {"python": self.py, "pkg": self.pkg, "worldclim": self.wc, "paleoclim": self.pc,
             "bio": self.bio, "bbox": self.bbox, "iso_taxa": self.iso_taxa,
             "sdm_taxa": self.sdm_taxa, "out": self.out}
        d.update(self.inputs)
        return d

    def _restore(self, s):
        for k, w in self._fields().items():
            v = s.value(SETTINGS_PREFIX + k, "")
            if v:
                w.setText(v)

    def _save(self):
        s = QSettings()
        for k, w in self._fields().items():
            s.setValue(SETTINGS_PREFIX + k, w.text())

    def use_canvas_extent(self):
        e = self.iface.mapCanvas().extent()
        crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        if crs.authid() != "EPSG:4326":
            from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform
            tr = QgsCoordinateTransform(crs, QgsCoordinateReferenceSystem("EPSG:4326"),
                                        QgsProject.instance())
            e = tr.transformBoundingBox(e)
        self.bbox.setText(f"{e.xMinimum():.3f},{e.yMinimum():.3f},"
                          f"{e.xMaximum():.3f},{e.yMaximum():.3f}")

    # ---------------------------------------------------------------- 실행
    def build_config(self):
        nums = lambda t: [float(v) for v in t.split(",") if v.strip()]
        strs = lambda t: [v.strip() for v in t.split(",") if v.strip()]
        return {
            "출력폴더": self.out.text(),
            "자료": {k: w.text() for k, w in self.inputs.items()},
            "고환경": {"worldclim_dir": self.wc.text(), "paleoclim_dir": self.pc.text(),
                       "bio": [int(v) for v in nums(self.bio.text())]},
            "범위": {"bbox": nums(self.bbox.text())},
            "분석": {"프록시": self.proxy.currentData(),
                     "아이소스케이프_분류군": strs(self.iso_taxa.text()),
                     "SDM_분류군": strs(self.sdm_taxa.text()),
                     "최소유적수_아이소스케이프": 3,
                     "최소유적수_SDM": 4,
                     "지지거리_km": self.dist.value(),
                     "beta_multiplier": 3.0,
                     "공간블록교차검증": self.cv.isChecked()},
        }

    def run(self):
        if not self.py.text() or not os.path.exists(self.py.text()):
            self.log.appendPlainText("[오류] 파이썬 실행 파일을 지정하세요.")
            return
        if not self.out.text():
            self.log.appendPlainText("[오류] 출력 폴더를 지정하세요.")
            return
        self._save()
        cfg = self.build_config()
        self.out_dir = cfg["출력폴더"]
        os.makedirs(self.out_dir, exist_ok=True)

        fd, path = tempfile.mkstemp(suffix=".json", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

        self.log.clear()
        self.log.appendPlainText("분석을 시작합니다… (자료 크기에 따라 몇 분 걸립니다)")
        self.btn_run.setEnabled(False)

        self.proc = QProcess(self)
        env = self.proc.processEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        if self.pkg.text():
            env.insert("PYTHONPATH", self.pkg.text())
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.proc.start(self.py.text(), ["-m", "archaeo_sdm.cli", "run", path])

    def _read(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in data.splitlines():
            if line.strip():
                self.log.appendPlainText(line.rstrip())
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _done(self, code, _status):
        self.btn_run.setEnabled(True)
        if code != 0:
            self.log.appendPlainText(f"[오류] 종료 코드 {code}. 위 메시지를 확인하세요.")
            self.iface.messageBar().pushMessage("Archaeo-SDM", "분석 실패",
                                                level=Qgis.Critical, duration=6)
            return
        n = self.load_outputs()
        self.log.appendPlainText(f"완료 — 래스터 {n}장을 레이어로 불러왔습니다.")
        self.iface.messageBar().pushMessage(
            "Archaeo-SDM", f"완료: 래스터 {n}장을 불러왔습니다 ({self.out_dir})",
            level=Qgis.Success, duration=8)

    # ------------------------------------------------------- 결과 레이어 적재
    def load_outputs(self):
        if not self.out_dir or not os.path.isdir(self.out_dir):
            return 0
        root = QgsProject.instance().layerTreeRoot()
        group = root.insertGroup(0, "Archaeo-SDM 결과")
        count = 0
        for name in sorted(os.listdir(self.out_dir)):
            if not name.lower().endswith(".tif"):
                continue
            layer = QgsRasterLayer(os.path.join(self.out_dir, name),
                                   os.path.splitext(name)[0])
            if not layer.isValid():
                continue
            self.style(layer, name)
            QgsProject.instance().addMapLayer(layer, False)
            group.addLayer(layer)
            count += 1
        return count

    @staticmethod
    def style(layer, name):
        """이름에 따라 색상표를 다르게 적용합니다."""
        stats = layer.dataProvider().bandStatistics(1)
        lo, hi = stats.minimumValue, stats.maximumValue
        if hi <= lo:
            return
        if name.startswith("delta_"):                 # 변화면: 0을 흰색으로 하는 발산형
            m = max(abs(lo), abs(hi))
            stops = [(-m, QColor(33, 102, 172)), (0.0, QColor(247, 247, 247)),
                     (m, QColor(178, 24, 43))]
        elif name.startswith("sdm_"):                 # 적합도: 0~1
            stops = [(0.0, QColor(68, 1, 84)), (0.5, QColor(33, 145, 140)),
                     (1.0, QColor(253, 231, 37))]
        else:                                         # 아이소스케이프
            mid = (lo + hi) / 2
            stops = [(lo, QColor(49, 54, 149)), (mid, QColor(255, 255, 191)),
                     (hi, QColor(165, 0, 38))]
        ramp = QgsColorRampShader(min(s[0] for s in stops), max(s[0] for s in stops))
        ramp.setColorRampType(QgsColorRampShader.Interpolated)
        ramp.setColorRampItemList([QgsColorRampShader.ColorRampItem(v, c, f"{v:.2f}")
                                   for v, c in stops])
        shader = QgsRasterShader()
        shader.setRasterShaderFunction(ramp)
        layer.setRenderer(QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader))
        layer.triggerRepaint()


class ArchaeoSdmPlugin:
    """QGIS가 불러가는 진입점."""

    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dialog = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        self.action = QAction(icon, "Archaeo-SDM 분석…", self.iface.mainWindow())
        self.action.triggered.connect(self.show)
        self.iface.addPluginToMenu("&Archaeo-SDM", self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        self.iface.removePluginMenu("&Archaeo-SDM", self.action)
        self.iface.removeToolBarIcon(self.action)

    def show(self):
        if self.dialog is None:
            self.dialog = ArchaeoSdmDialog(self.iface)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
