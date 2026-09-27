"""
SDM Studio — 브라우저 화면.

    python -m archaeo_sdm.cli studio          →  http://127.0.0.1:8766

기존 브라우저 UI(8765번)가 'MaxEnt 전체 파이프라인'이라면, Studio는
여러 알고리즘을 나란히 비교하고, 기후 시나리오·분산·동위원소 니치를 따로 실험하는 곳입니다.
"""

from __future__ import annotations

import os
import threading
import traceback
import uuid

from flask import Flask, abort, jsonify, render_template, request, send_file

from .. import paths
from ..models import MODEL_INFO
from . import engine

app = Flask(__name__)
JOBS: dict[str, dict] = {}
_DF_CACHE: dict[str, object] = {}

RUNNERS = {"compare": engine.run_compare, "scenario": engine.run_scenario,
           "dispersal": engine.run_dispersal, "niche": engine.run_niche}


def _df(path=None):
    path = path or (engine.list_datasets() or [None])[0]
    if path not in _DF_CACHE:
        _DF_CACHE[path] = engine.load_dataset(path)
    return _DF_CACHE[path]


@app.route("/")
def index():
    return render_template("studio.html",
                           models={k: v[0] for k, v in MODEL_INFO.items()},
                           min_sites={k: v[1] for k, v in MODEL_INFO.items()},
                           slices=engine.SLICES)


@app.route("/api/summary")
def api_summary():
    try:
        s = engine.dataset_summary(_df(request.args.get("path") or None))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    def tbl(t):
        t = t.reset_index()
        return {"columns": [str(c) for c in t.columns], "rows": t.astype(object).values.tolist()}

    return jsonify({"path": s["path"], "datasets": engine.list_datasets(), "n": s["n"],
                    "sites": s["sites"], "taxa": s["taxa"], "zones": s["zones"],
                    "periods": s["periods"], "taxon_period": tbl(s["taxon_period_sites"]),
                    "zone_period": tbl(s["zone_period_sites"])})


def _worker(job_id, kind, cfg):
    job = JOBS[job_id]
    try:
        job["status"] = "running"
        res = RUNNERS[kind](_df(cfg.get("dataset")), cfg, job["dir"],
                            log=lambda m: job["log"].append(str(m)))
        job["result"] = engine.to_json(res)
        job["status"] = "done"
    except Exception as e:                                     # 화면에 원인을 보여 줍니다
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
        job["trace"] = traceback.format_exc()


@app.route("/api/run/<kind>", methods=["POST"])
def api_run(kind):
    if kind not in RUNNERS:
        abort(404)
    cfg = request.get_json(force=True)
    job_id = uuid.uuid4().hex[:8]
    out = os.path.join(paths.OUTPUTS, "studio", f"{kind}_{job_id}")
    JOBS[job_id] = {"status": "queued", "log": [], "dir": out, "kind": kind}
    threading.Thread(target=_worker, args=(job_id, kind, cfg), daemon=True).start()
    return jsonify({"job": job_id})


@app.route("/api/status/<job_id>")
def api_status(job_id):
    j = JOBS.get(job_id) or abort(404)
    return jsonify({k: j.get(k) for k in ("status", "log", "error", "result")})


@app.route("/file/<job_id>/<path:name>")
def file_(job_id, name):
    j = JOBS.get(job_id) or abort(404)
    p = os.path.join(j["dir"], name)
    return send_file(p) if os.path.isfile(p) else abort(404)


@app.route("/guide")
def guide():
    p = os.path.join(paths.PKG_ROOT, "docs", "SDM_tools_survey.md")
    if not os.path.exists(p):
        abort(404)
    import markdown

    with open(p, encoding="utf-8") as f:
        html = markdown.markdown(f.read(), extensions=["tables", "fenced_code"])
    return jsonify({"html": html})


def serve(host="127.0.0.1", port=8766):
    print(f"\n  SDM Studio: http://{host}:{port}  (Ctrl+C 로 종료)\n")
    app.run(host=host, port=port, threaded=True)
