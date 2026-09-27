"""
브라우저 UI (로컬 실행).

    python -m archaeo_sdm.cli web
    -> http://127.0.0.1:8765 접속

자료 경로를 넣고 [분석 실행]을 누르면 백그라운드에서 분석이 돌고,
진행 상황이 실시간으로 표시되며, 끝나면 지도·그림·표를 브라우저에서 바로 봅니다.
GeoTIFF는 지도 위에 겹쳐 볼 수 있고, 원본 파일은 그대로 QGIS에서 열 수 있습니다.
"""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import traceback
import uuid

from flask import Flask, abort, jsonify, render_template, request, send_file

from ..cli import TEMPLATE, run_config

app = Flask(__name__)
JOBS: dict[str, dict] = {}


# ------------------------------------------------------------------ 실행 관리
def _worker(job_id: str, cfg: dict):
    job = JOBS[job_id]

    def progress(msg):
        job["log"].append(str(msg))

    try:
        job["status"] = "running"
        job["summary"] = run_config(cfg, progress)
        job["status"] = "done"
    except Exception as e:                     # 사용자가 원인을 볼 수 있어야 합니다
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        job["log"].append(job["error"])
        job["traceback"] = traceback.format_exc()


@app.route("/")
def index():
    return render_template("index.html", template=json.dumps(TEMPLATE, ensure_ascii=False,
                                                             indent=2))


@app.route("/api/defaults")
def api_defaults():
    """이 컴퓨터에서 찾은 자료 경로를 돌려줍니다(예시 경로 채우기 버튼)."""
    from .. import paths
    return jsonify({
        "china_db": paths.CHINA_DB or "",
        "isomemo_db": paths.ISOMEMO_DB or "",
        "ecology_xlsx": paths.ECOLOGY_XLSX or "",
        "point_shapefile": paths.PRIMORYE_SHP or "",
        "point_csv": paths.PRIMORYE_CSV or "",
        "gazetteer_manual": paths.GAZETTEER_MANUAL,
        "worldclim_dir": paths.WORLDCLIM_DIR or "",
        "paleoclim_dir": paths.PALEOCLIM_DIR,
        "out": os.path.join(paths.OUTPUTS, "web_run"),
    })


@app.route("/api/run", methods=["POST"])
def api_run():
    cfg = request.get_json(force=True)
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "queued", "log": [], "cfg": cfg}
    threading.Thread(target=_worker, args=(job_id, cfg), daemon=True).start()
    return jsonify({"job": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    job = JOBS.get(job_id) or abort(404)
    return jsonify({"status": job["status"], "log": job["log"],
                    "error": job.get("error"), "summary": job.get("summary")})


@app.route("/api/files/<job_id>")
def api_files(job_id):
    job = JOBS.get(job_id) or abort(404)
    out = job.get("summary", {}).get("출력폴더") or job["cfg"]["출력폴더"]
    if not os.path.isdir(out):
        return jsonify({"images": [], "tables": [], "rasters": []})
    names = sorted(os.listdir(out))
    return jsonify({
        "dir": os.path.abspath(out),
        "images": [n for n in names if n.lower().endswith(".png")],
        "tables": [n for n in names if n.lower().endswith(".csv")],
        "rasters": [n for n in names if n.lower().endswith(".tif")],
    })


def _job_dir(job_id):
    job = JOBS.get(job_id) or abort(404)
    return job.get("summary", {}).get("출력폴더") or job["cfg"]["출력폴더"]


@app.route("/file/<job_id>/<path:name>")
def file_(job_id, name):
    path = os.path.join(_job_dir(job_id), name)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path)


@app.route("/api/table/<job_id>/<path:name>")
def api_table(job_id, name):
    import pandas as pd

    path = os.path.join(_job_dir(job_id), name)
    if not os.path.isfile(path):
        abort(404)
    d = pd.read_csv(path).head(300)
    return jsonify({"columns": list(d.columns),
                    "rows": d.astype(object).where(d.notna(), None).values.tolist()})


@app.route("/api/overlay/<job_id>/<path:name>")
def api_overlay(job_id, name):
    """GeoTIFF를 지도에 겹칠 PNG(투명 배경)와 경위도 범위로 변환."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds

    path = os.path.join(_job_dir(job_id), name)
    if not os.path.isfile(path):
        abort(404)
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float32").filled(np.nan)
        b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

    finite = np.isfinite(arr)
    if not finite.any():
        abort(404)
    lo, hi = np.nanpercentile(arr[finite], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((arr - lo) / (hi - lo), 0, 1)
    cmap = cm.get_cmap("viridis") if hasattr(cm, "get_cmap") else matplotlib.colormaps["viridis"]
    rgba = (cmap(norm) * 255).astype("uint8")
    rgba[..., 3] = np.where(finite, 200, 0)

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return jsonify({
        "png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        "bounds": [[b[1], b[0]], [b[3], b[2]]],     # [[남,서],[북,동]]
        "min": float(lo), "max": float(hi),
    })


def serve(host: str = "127.0.0.1", port: int = 8765, debug: bool = False):
    print(f"\n  브라우저에서 http://{host}:{port} 를 여세요.  (Ctrl+C 로 종료)\n")
    app.run(host=host, port=port, debug=debug, threaded=True)
