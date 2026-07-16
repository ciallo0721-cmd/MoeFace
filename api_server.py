"""
MoeFace REST API 服务
提供 HTTP 接口供外部程序调用识别功能
"""
import os, sys, json, threading, uuid, base64, io, tempfile, time, traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
from flask import Flask, request, jsonify

# 确保能导入父目录的 recognize 模块
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

app = Flask("MoeFace-API")

# 共享状态
_api_database = {}
_api_db_name = ""
_api_negative_db = {}
_api_ready = False
_api_tasks: dict = {}
_api_lock = threading.Lock()

# ── 延迟导入 recognize 模块 ──
_recognize = None
def _get_recognize():
    global _recognize
    if _recognize is None:
        import recognize as _recognize
    return _recognize

def init_api(db_name="全部特征库", log_fn=print):
    """初始化 API 服务（加载模型和特征库）"""
    global _api_database, _api_db_name, _api_negative_db, _api_ready
    rec = _get_recognize()
    if not rec._ensure_models(log_fn):
        log_fn("❌ API: 模型加载失败")
        return False
    db, _ = rec.get_or_build_database(db_name, log_fn=log_fn)
    _api_database = db
    _api_db_name = db_name
    ndb = rec.get_or_build_negative_database(log_fn=log_fn)
    _api_negative_db = ndb
    _api_ready = True
    log_fn(f"✅ API 就绪：{len(db)} 个角色，{len(ndb)} 个负面类别")
    return True

# ── 路由 ──

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if _api_ready else "loading",
        "db_name": _api_db_name,
        "roles_count": len(_api_database),
        "timestamp": datetime.now().isoformat(),
    })

@app.route("/api/roles", methods=["GET"])
def list_roles():
    return jsonify({
        "roles": sorted(_api_database.keys()),
        "count": len(_api_database),
    })

@app.route("/api/recognize", methods=["POST"])
def recognize():
    if not _api_ready:
        return jsonify({"error": "API 未就绪"}), 503

    rec = _get_recognize()
    import cv2

    # 解析输入：支持 multipart/form-data 或 JSON base64
    img_data = None
    if "image" in request.files:
        img_data = request.files["image"].read()
    elif request.is_json:
        body = request.get_json(silent=True) or {}
        b64 = body.get("image") or body.get("base64", "")
        if b64:
            img_data = base64.b64decode(b64)
    else:
        return jsonify({"error": "请提供 image 文件或 base64 数据"}), 400

    if not img_data:
        return jsonify({"error": "图片数据为空"}), 400

    try:
        data = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "无法解码图片"}), 400
    except Exception as e:
        return jsonify({"error": f"图片解码失败: {e}"}), 400

    # 参数
    threshold = request.form.get("threshold", type=float) or \
                (request.get_json(silent=True) or {}).get("threshold", 0.45)
    body_mode = request.form.get("body_mode", type=bool) or \
                (request.get_json(silent=True) or {}).get("body_mode", False)

    # 姿态检测
    persons = []
    if body_mode:
        persons = rec.detect_body_pose(img, print)

    # 人脸检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(str(rec.CASCADE_PATH))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=5,
                                      minSize=(20, 20), maxSize=(800, 800))

    results = []
    for (x, y, w, h) in faces:
        if not rec._is_valid_face_box(w, h):
            continue
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img.shape[1], x+w), min(img.shape[0], y+h)
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        name, score = rec.recognize_character(face_rgb, _api_database, threshold,
                                               full_img=img, body_persons=persons,
                                               negative_db=_api_negative_db)
        results.append({
            "bbox": {"x": int(x1), "y": int(y1), "w": int(x2-x1), "h": int(y2-y1)},
            "name": name,
            "score": round(float(score), 4),
        })

    return jsonify({
        "success": True,
        "faces_detected": len(faces),
        "recognized": len([r for r in results if r["name"]]),
        "results": results,
        "threshold": threshold,
    })

@app.route("/api/recognize/video", methods=["POST"])
def recognize_video_async():
    """提交视频异步处理"""
    if not _api_ready:
        return jsonify({"error": "API 未就绪"}), 503

    data = request.get_json(silent=True) or {}
    video_path = data.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "请提供有效的 video_path"}), 400

    task_id = str(uuid.uuid4())[:8]
    threshold = data.get("threshold", 0.45)
    skip_frames = data.get("skip_frames", 2)

    def _run():
        rec = _get_recognize()
        output = f"{Path(video_path).stem}_api_result_{task_id}.mp4"
        _api_tasks[task_id] = {"status": "processing", "progress": 0}

        def log_fn(msg):
            print(f"[API {task_id}] {msg}")

        rec.process_video_file(
            source=video_path, database=_api_database,
            output_path=output, threshold=threshold,
            skip_frames=skip_frames, log_fn=log_fn,
            negative_db=_api_negative_db,
        )
        _api_tasks[task_id] = {"status": "done", "output": output}

    threading.Thread(target=_run, daemon=True).start()
    _api_tasks[task_id] = {"status": "queued"}
    return jsonify({"task_id": task_id, "status": "queued"})

@app.route("/api/tasks/<task_id>", methods=["GET"])
def get_task(task_id):
    task = _api_tasks.get(task_id)
    if not task:
        return jsonify({"error": "task_id 不存在"}), 404
    return jsonify({"task_id": task_id, **task})

def start_api_server(host="0.0.0.0", port=5000, db_name="全部特征库", log_fn=print):
    """启动 API 服务器（线程中运行）"""
    if not init_api(db_name, log_fn):
        return False
    log_fn(f"🌐 API 服务启动: http://{host}:{port}")
    log_fn(f"   GET  /api/health  — 健康检查")
    log_fn(f"   GET  /api/roles   — 角色列表")
    log_fn(f"   POST /api/recognize  — 识别图片")
    log_fn(f"   POST /api/recognize/video  — 视频识别（异步）")
    log_fn(f"   GET  /api/tasks/{{id}} — 查询异步任务状态")
    threading.Thread(target=app.run, args=(host, port),
                     kwargs={"debug": False, "use_reloader": False},
                     daemon=True).start()
    return True
