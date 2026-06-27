"""
MoeFace — 动漫人脸识别系统 / VTuber 二次元角色识别工具
基于 FaceNet + OpenCV 实现动漫人脸检测与特征匹配
支持 VTuber / 虚拟主播识别，本地运行保护隐私
GUI 版本 (Tkinter) + CLI 版本 (终端图形化)
"""

"""
带"～(｡•́︿•̀｡)"是严重错误
"""
import os
import sys
import json
import threading
import warnings
import shutil
import tempfile
import traceback
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any


# ── 确保以脚本所在目录为基准路径 ────────────────────────────────────────────
import sys as _sys

def _resource_base() -> Path:
    """兼容 PyInstaller 打包：打包后资源在 _MEIPASS，开发时在脚本目录"""
    if getattr(_sys, "frozen", False) and hasattr(_sys, "_MEIPASS"):
        return Path(_sys._MEIPASS)
    return Path(__file__).resolve().parent

BASE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = _resource_base()

os.chdir(BASE_DIR)

CASCADE_PATH = RESOURCE_DIR / "lbpcascade_animeface.xml"
FONT_PATH    = RESOURCE_DIR / "simhei.ttf"
FEATURES_DIR = BASE_DIR / "features"
DATA_DIR     = BASE_DIR / "data"
CNAME_PATH   = RESOURCE_DIR / "cname" / "name.json"
DEFAULT_DB_NAME = "全部特征库"

# ── 新 .moe 文本格式的部位键名 ──────────────────────────────────────────
FEATURE_KEYS = ["eye", "eye2", "nose", "mouth", "head",
                "arm", "arm2", "hand", "hand2", "leg", "leg2"]
FACE_KEYS = FEATURE_KEYS[:5]    # eye, eye2, nose, mouth, head
LIMB_KEYS = FEATURE_KEYS[5:]    # arm, arm2, hand, hand2, leg, leg2

FEATURES_DIR.mkdir(exist_ok=True)

# ── 延迟导入重型库（避免 import 时卡死 GUI）───────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

try:
    import tkinterdnd2 as tkdnd
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

# ── 角色文件夹扫描 ────────────────────────────────────────────────────────
def scan_role_folders(data_root: Path = DATA_DIR):
    """扫描 data 目录，返回所有角色名称列表。"""
    roles = []
    if not data_root.exists():
        return roles

    for game_dir in data_root.iterdir():
        if not game_dir.is_dir():
            continue

        subdirs = [d for d in game_dir.iterdir() if d.is_dir()]
        if subdirs and any(d.is_dir() for d in game_dir.iterdir()):
            for role_dir in subdirs:
                roles.append(f"{game_dir.name}/{role_dir.name}")
        else:
            roles.append(game_dir.name)

    return sorted(roles)


# ── 别名模块：从 cname/name.json 加载 ─────────────────────────────────────
def load_alias_map(path: Path = CNAME_PATH):
    if not path.exists():
        warnings.warn(f"(｡•́︿•̀｡),主人,别名配置文件丢了喵: {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        warnings.warn(f"(｡•́︿•̀｡),主人,加载别名配置失败: {e}")
        return []

ALIAS_MAP = load_alias_map()

def get_db_name_from_filename(filename: str) -> str:
    """根据文件名匹配角色别名（短别名要求词边界，避免误杀）"""
    import re as _re
    name_lower = filename.lower()
    for entry in ALIAS_MAP:
        for alias in entry.get("aliases", []):
            alias_lower = alias.lower()
            # 拉丁短别名（< 4 字）：严格词边界匹配
            if len(alias_lower) <= 3 and not any('\u4e00' <= c <= '\u9fff' for c in alias_lower):
                if _re.search(r'(?<!\w)' + _re.escape(alias_lower) + r'(?!\w)', name_lower):
                    return entry["db_name"]
            # 单字中文别名：前后须为文件名边界（避免"希"匹配"希望"）
            elif len(alias) == 1 and '\u4e00' <= alias <= '\u9fff':
                if _re.search(r'(^|[^\u4e00-\u9fff])' + _re.escape(alias) + r'([^\u4e00-\u9fff]|$)', filename):
                    return entry["db_name"]
            else:
                if alias_lower in name_lower:
                    return entry["db_name"]
    return DEFAULT_DB_NAME


# ── 特征库管理（自研 .moe 格式）───────────────────────────────────────
def _safe_filename(name: str) -> str:
    """生成安全的文件名（保留中文、字母、数字、部分符号）"""
    keep = set("._- ·•")
    return "".join(c for c in name if c.isalnum() or c in keep or "\u4e00" <= c <= "\u9fff")


def save_database_to_moe(database: dict, db_name: str):
    """
    保存特征库为自研 .moe 文本格式
    格式：("角色名"{key1:val1:key2:val2:...keyN:valN:}"角色名2"{...})
    其中 val 为逗号分隔的浮点数（特征向量）
    """
    if not database:
        return None
    import numpy as np
    safe = _safe_filename(db_name)
    moe_path = FEATURES_DIR / f"{safe}.moe"
    moe_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    for name, parts in database.items():
        content_parts = []
        for key in FEATURE_KEYS:
            if key in parts and isinstance(parts[key], np.ndarray):
                vec_str = ",".join(f"{v:.10f}" for v in parts[key])
                content_parts.append(f"{key}:{vec_str}")
        content = ":".join(content_parts)
        # 加 trailing colon 以匹配格式规范
        chunks.append(f'"{name}"{{{content}:}}')

    all_text = "(" + "".join(chunks) + ")"
    moe_path.write_text(all_text, encoding="utf-8")
    return moe_path


def load_database_from_moe(db_name: str):
    """
    从自研 .moe 文本格式加载特征库
    格式：("角色名"{key1:val1:key2:val2:...}"角色名2"{...})
    """
    import numpy as np

    safe = _safe_filename(db_name)
    moe_path = FEATURES_DIR / f"{safe}.moe"
    if not moe_path.exists():
        return None

    try:
        text = moe_path.read_text(encoding="utf-8").strip()
        if not text.startswith("(") or not text.endswith(")"):
            warnings.warn(f"(｡•́︿•̀｡),主人,.moe 文件格式错误（缺少外层括号）: {moe_path}")
            return None

        inner = text[1:-1]  # 去掉外层 ()
        database = {}

        # 按 " 分割提取每个角色条目
        # 格式： "Name1"{...}"Name2"{...}
        # 分割后: ["", "Name1", "{content}", "Name2", "{content}", ""]
        parts = inner.split('"')
        i = 1
        while i + 1 < len(parts):
            name = parts[i]
            content_block = parts[i + 1]
            i += 2

            if not content_block.startswith("{") or not content_block.endswith("}"):
                continue

            content = content_block[1:-1]  # 去掉 {}
            if not content:
                continue

            # 去掉结尾可能的 :
            if content.endswith(":"):
                content = content[:-1]

            # 解析 key:val:key:val:...
            kv_parts = content.split(":")
            parts_dict = {}
            for j in range(0, len(kv_parts), 2):
                if j + 1 >= len(kv_parts):
                    break
                key = kv_parts[j]
                val_str = kv_parts[j + 1]
                if not val_str or val_str == "placeholder":
                    continue
                try:
                    vals = [float(x) for x in val_str.split(",")]
                    parts_dict[key] = np.array(vals, dtype=np.float32)
                except ValueError:
                    continue

            if parts_dict:
                database[name] = parts_dict

        return database

    except Exception as e:
        warnings.warn(f"(｡•́︿•̀｡),主人,加载 .moe 特征库失败了: {e}")
        return None


# 保留 JSON 版本作为兼容备份（可选）
def save_database_to_json(database: dict, json_name: str):
    if not database:
        return None
    safe = _safe_filename(json_name)
    json_path = FEATURES_DIR / f"{safe}.json.bak"  # 改为 .bak 后缀，不主动使用
    json_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {n: emb.tolist() for n, emb in database.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return json_path


def load_database_from_json(json_name: str):
    """仅用于兼容旧版，转换为新 .moe 多部位格式"""
    import numpy as np
    safe = _safe_filename(json_name)
    json_path = FEATURES_DIR / f"{safe}.json"
    if not json_path.exists():
        # 尝试 .bak
        json_path = FEATURES_DIR / f"{safe}.json.bak"
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # 旧格式 {n: [512 float]} → 新格式 {n: {key: emb}}
        database = {}
        for n, e in raw.items():
            vec = np.array(e, dtype=np.float32)
            parts = {}
            for key in FEATURE_KEYS:
                parts[key] = vec.copy()
            database[n] = parts
        return database
    except Exception as e:
        warnings.warn(f"(｡•́︿•̀｡),主人,加载 JSON 特征库失败了: {e}")
        return None


# ── 核心识别逻辑（懒加载，首次使用时初始化）──────────────────────────────────
_model_lock = threading.Lock()
_models_ready = False
_anime_cascade = None
_resnet = None
_device = None
# 每个线程独立的 CascadeClassifier（解决线程安全问题）
_thread_local = threading.local()

def _get_thread_cascade():
    """获取当前线程的 CascadeClassifier（每个线程独立）"""
    if not hasattr(_thread_local, 'cascade'):
        cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
        _thread_local.cascade = cascade
    return _thread_local.cascade

def _ensure_models(log_fn=print):
    """确保模型已加载（线程安全，只初始化一次）"""
    global _models_ready, _anime_cascade, _resnet, _device
    if _models_ready:
        return True
    with _model_lock:
        if _models_ready:
            return True
        try:
            import cv2
            import torch
            from facenet_pytorch import InceptionResnetV1

            cascade_p = str(CASCADE_PATH)
            if not CASCADE_PATH.exists():
                log_fn("❌ (｡•́︿•̀｡),找不到 lbpcascade_animeface.xml")
                return False
            clf = cv2.CascadeClassifier(cascade_p)
            if clf.empty():
                log_fn("❌ (｡•́︿•̀｡),CascadeClassifier 加载失败")
                return False
            _anime_cascade = clf

            log_fn("正在加载 FaceNet 模型喵~（首次可能需要几秒）...")
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _resnet = InceptionResnetV1(pretrained="vggface2").eval().to(_device)
            log_fn(f"✅ 模型加载完成了,主人~（{'GPU' if _device.type == 'cuda' else 'CPU'}）")

            _models_ready = True
            return True
        except Exception as e:
            log_fn(f"❌ (｡•́︿•̀｡),模型初始化失败: {e}")
            traceback.print_exc()
            return False


def extract_features_from_image(image_path: str, log_fn=print):
    """提取单张图片的特征，若模型未就绪或图片无效则返回 None"""
    global _resnet, _device
    if not _ensure_models(log_fn):
        log_fn("❌ 模型未就绪，无法提取特征")
        return None

    import cv2, torch
    import numpy as np

    cascade = _get_thread_cascade()

    try:
        with open(image_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            log_fn(f"⚠️ 无法解码图片: {image_path}")
            return None
    except Exception as e:
        log_fn(f"⚠️ 读取图片失败 {image_path}: {e}")
        return None

    MAX_DIM = 4096
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.02, minNeighbors=3,
        minSize=(20, 20), maxSize=(800, 800)
    )
    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_rsz = cv2.resize(face_rgb, (160, 160))
    tensor = (torch.tensor(face_rsz).permute(2, 0, 1)
              .float().unsqueeze(0).to(_device) / 255.0)
    with torch.no_grad():
        emb = _resnet(tensor).cpu().numpy().flatten()
    return emb


# ── 多部位特征提取 ─────────────────────────────────────────────────────
def _extract_face_embedding(face_rgb_img):
    """从 RGB 人脸图像提取 FaceNet 512维嵌入"""
    import cv2, torch
    face = cv2.resize(face_rgb_img, (160, 160))
    tensor = (torch.tensor(face).permute(2, 0, 1)
              .float().unsqueeze(0).to(_device) / 255.0)
    with torch.no_grad():
        return _resnet(tensor).cpu().numpy().flatten()


def _extract_limb_embeddings(full_img_bgr, body_persons):
    """
    从全身图像 + YOLO-Pose 检测结果提取肢体特征
    返回 dict: {key: 512-dim embedding}
    若无可用的肢体检测结果，返回空 dict
    """
    import cv2
    import numpy as np

    LIMB_KP_MAP = {
        "arm":  (5, 7),    # left_shoulder → left_elbow
        "arm2": (6, 8),    # right_shoulder → right_elbow
        "hand": (9,),      # left_wrist
        "hand2": (10,),    # right_wrist
        "leg":  (11, 13),  # left_hip → left_knee
        "leg2": (12, 14),  # right_hip → right_knee
    }

    if not body_persons:
        return {}

    h, w = full_img_bgr.shape[:2]
    results = {}
    # 使用检测到的第一个人体
    _, _, _, _, kps = body_persons[0]
    MAX_KPS = 17

    for key, kp_indices in LIMB_KP_MAP.items():
        points = []
        for idx in kp_indices:
            if idx < len(kps) and kps[idx][2] > 0.5:
                px = int(kps[idx][0] * w)
                py = int(kps[idx][1] * h)
                points.append((px, py))

        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cx, cy = sum(xs) // len(xs), sum(ys) // len(ys)

        # 单点（如 hand）用固定矩形, 两点用连线矩形+边距
        MARGIN = 50
        if len(points) == 1:
            x1 = max(0, cx - MARGIN)
            y1 = max(0, cy - MARGIN)
            x2 = min(w, cx + MARGIN)
            y2 = min(h, cy + MARGIN)
        else:
            x1 = max(0, min(xs) - 15)
            y1 = max(0, min(ys) - 15)
            x2 = min(w, max(xs) + 15)
            y2 = min(h, max(ys) + 15)

        if (x2 - x1) < 30 or (y2 - y1) < 30:
            continue

        region = full_img_bgr[y1:y2, x1:x2]
        if region.size == 0:
            continue

        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        try:
            emb = _extract_face_embedding(region_rgb)
            results[key] = emb
        except Exception:
            continue

    return results


def extract_multi_features_from_image(image_path: str, log_fn=print):
    """
    从图片提取全部 11 个部位特征（面部 5 + 肢体 6）
    返回 dict: {key: 512-dim embedding} 或 None(完全失败)
    """
    import cv2, torch
    import numpy as np

    if not _ensure_models(log_fn):
        log_fn("❌ 模型未就绪，无法提取特征")
        return None

    cascade = _get_thread_cascade()
    try:
        with open(image_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            log_fn(f"⚠️ 无法解码图片: {image_path}")
            return None
    except Exception as e:
        log_fn(f"⚠️ 读取图片失败 {image_path}: {e}")
        return None

    # 缩放超大图
    MAX_DIM = 4096
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    results = {}

    # ── 1. 面部特征 ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.02, minNeighbors=3,
        minSize=(20, 20), maxSize=(800, 800)
    )
    face_emb = None
    if len(faces) > 0:
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        x, y, fw, fh = faces[0]
        face = img[y:y+fh, x:x+fw]
        if face.size > 0:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            try:
                face_emb = _extract_face_embedding(face_rgb)
                # 面部 5 个键共享同一个面部分数
                for key in FACE_KEYS:
                    results[key] = face_emb.copy()
            except Exception as e:
                log_fn(f"⚠️ 面部特征提取失败: {e}")

    # ── 2. 肢体特征 ──
    try:
        persons = detect_body_pose(img, lambda m: None)
        if persons:
            limb_embs = _extract_limb_embeddings(img, persons)
            results.update(limb_embs)
    except Exception:
        pass

    return results if results else None


def build_database(data_root: Path, log_fn=print, progress_fn=None, stop_event=None):
    """构建多部位特征库——每个角色存储 11 个部位的平均特征向量"""
    if not _ensure_models(log_fn):
        log_fn("❌ 模型加载失败，无法构建特征库")
        return {}

    import numpy as np
    import time

    database = {}
    if not data_root.exists():
        log_fn(f"❌呜呜呜,主人,路径不存在喵～(｡•́︿•̀｡): {data_root}")
        return database

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def _gather(folder):
        files = []
        for ext in exts:
            files += list(folder.glob(f"**/*{ext}"))
            files += list(folder.glob(f"**/*{ext.upper()}"))
        return files

    MAX_PER_PERSON = 50
    total_images = 0
    person_images = {}

    subdirs = [d for d in data_root.iterdir() if d.is_dir()]

    if subdirs:
        # 分成两类：游戏目录（含子目录）和直接角色目录（直接放图片）
        game_dirs  = [d for d in subdirs if any(e.is_dir() for e in d.iterdir())]
        role_dirs  = [d for d in subdirs if d not in game_dirs]

        # 直接角色目录（如 ATRI、warma）→ 直接收集图片
        for person_dir in role_dirs:
            imgs = _gather(person_dir)
            if imgs:
                person_images[person_dir.name] = imgs[:MAX_PER_PERSON]
                total_images += min(len(imgs), MAX_PER_PERSON)

        # 游戏目录（如 ddlc、vtuber）→ 递归处理子目录
        for game_dir in game_dirs:
            for role_dir in game_dir.iterdir():
                if role_dir.is_dir():
                    imgs = _gather(role_dir)
                    if imgs:
                        key = f"{game_dir.name}/{role_dir.name}"
                        person_images[key] = imgs[:MAX_PER_PERSON]
                        total_images += min(len(imgs), MAX_PER_PERSON)
    else:
        imgs = _gather(data_root)
        if imgs:
            person_images[data_root.name] = imgs[:MAX_PER_PERSON]
            total_images = min(len(imgs), MAX_PER_PERSON)

    log_fn(f"多角色模式，共 {len(person_images)} 个角色，{total_images} 张图片，"
           f"每角色 {len(FEATURE_KEYS)} 个部位特征")

    start_time = time.time()
    processed = [0]

    def _process_person(person_name: str):
        import gc
        imgs = person_images[person_name][:50]
        if not imgs:
            return None

        if progress_fn:
            progress_fn(processed[0], total_images, time.time() - start_time, person_name)

        # 按部位键名收集特征向量
        collected = {key: [] for key in FEATURE_KEYS}

        for i, p in enumerate(imgs):
            if stop_event and stop_event.is_set():
                log_fn(f"⏹ 建库停止（{person_name} 处理到第 {i+1}/{len(imgs)} 张）")
                return None
            all_feats = extract_multi_features_from_image(str(p), log_fn)
            if all_feats:
                for key in FEATURE_KEYS:
                    if key in all_feats and all_feats[key] is not None:
                        collected[key].append(all_feats[key])
            processed[0] += 1

            if progress_fn and (i + 1) % 5 == 0:
                progress_fn(processed[0], total_images, time.time() - start_time, person_name)

        gc.collect()

        # 每个部位分别取平均
        result = {}
        has_any = False
        for key in FEATURE_KEYS:
            if collected[key]:
                result[key] = np.mean(collected[key], axis=0).astype(np.float32)
                has_any = True

        if has_any:
            return (person_name, result)
        return None

    for person_name in person_images.keys():
        if stop_event and stop_event.is_set():
            log_fn("⏹ 建库已停止")
            break
        result = _process_person(person_name)
        if result is not None:
            name, parts = result
            database[name] = parts
            face_count = sum(1 for k in FACE_KEYS if k in parts)
            limb_count = sum(1 for k in LIMB_KEYS if k in parts)
            log_fn(f"  ✅ {name} 完成（面部 {face_count} 部位 + 肢体 {limb_count} 部位）")
        else:
            log_fn(f"  ⚠️  {person_name} 无有效特征喵～(｡•́︿•̀｡)")

        if progress_fn:
            progress_fn(processed[0], total_images, time.time() - start_time, "")

    if progress_fn:
        progress_fn(total_images, total_images, time.time() - start_time, "")

    return database


def _find_role_path(db_name: str) -> Path:
    if not DATA_DIR.exists():
        return None

    old_path = DATA_DIR / db_name
    if old_path.exists() and old_path.is_dir():
        return old_path

    for game_dir in DATA_DIR.iterdir():
        if game_dir.is_dir():
            role_path = game_dir / db_name
            if role_path.exists() and role_path.is_dir():
                return role_path
            if "/" in db_name:
                parts = db_name.split("/", 1)
                if game_dir.name == parts[0] and (game_dir / parts[1]).exists():
                    return game_dir / parts[1]
    return None


def get_or_build_database(db_name: str, force_rebuild=False, log_fn=print, progress_fn=None, stop_event=None) -> Tuple[dict, bool]:
    """获取或构建特征库，返回 (database, built)"""
    if not force_rebuild:
        # 优先加载 .moe 格式
        db = load_database_from_moe(db_name)
        if db is not None:
            log_fn(f"✅ 从 .moe 缓存加载特征库 [{db_name}]，共 {len(db)} 个角色")
            return db, False
        # 兼容旧版：尝试加载 .json
        db = load_database_from_json(db_name)
        if db is not None:
            log_fn(f"✅ 从 .json 缓存加载特征库 [{db_name}]，共 {len(db)} 个角色（建议转为 .moe 格式）")
            return db, False

    if stop_event and stop_event.is_set():
        log_fn("⏹ 建库已停止")
        return {}, True

    if db_name == DEFAULT_DB_NAME:
        log_fn("构建全部特征库喵...")
        db = build_database(DATA_DIR, log_fn, progress_fn, stop_event)
    else:
        db_path = _find_role_path(db_name)
        if db_path is None:
            db_path = DATA_DIR / db_name
            if not db_path.exists():
                log_fn(f"❌ 文件夹不存在: {db_path}")
                return {}, True
        log_fn(f"构建特征库喵~: {db_name}")
        db = build_database(db_path, log_fn, progress_fn, stop_event)

    if db:
        save_database_to_moe(db, db_name)
        log_fn(f"✅ 特征库已缓存为 .moe 格式喵~，共 {len(db)} 个角色")
    return db, True


# ── 识别辅助 ──────────────────────────────────────────────────────────────
def cosine_similarity(a, b):
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

def recognize_character(face_img, database: dict, threshold=0.45,
                        full_img=None, body_persons=None):
    """
    多部位综合识别——同时对比面部 + 肢体特征
    face_img:  裁剪的人脸 RGB 图像
    database:  特征库（{name: {key: embedding}}）
    full_img:  完整 BGR 图像（用于提取肢体特征）
    body_persons: YOLO-Pose 检测结果（用于提取肢体特征）
    返回 (name, score)
    """
    import numpy as np

    # 1. 提取输入图像的面部特征
    face_emb = _extract_face_embedding(face_img)
    if face_emb is None:
        return (None, 0.0)

    # 2. 如果有全身图像，提取肢体特征
    limb_embs = {}
    if full_img is not None and body_persons:
        limb_embs = _extract_limb_embeddings(full_img, body_persons)

    # 3. 对每个角色计算综合相似度
    best_name, best_score = None, 0.0
    for name, parts in database.items():
        if not isinstance(parts, dict):
            continue

        scores = []
        weights = []

        # 面部特征（5个键全部比较）
        for key in FACE_KEYS:
            if key in parts:
                s = cosine_similarity(face_emb, parts[key])
                scores.append(s)
                weights.append(1.5)  # 面部权重更高

        # 肢体特征（如果有）
        for key in LIMB_KEYS:
            if key in limb_embs and key in parts:
                s = cosine_similarity(limb_embs[key], parts[key])
                scores.append(s)
                weights.append(1.0)

        if scores:
            avg_score = np.average(scores, weights=weights)
            if avg_score > best_score:
                best_score, best_name = avg_score, name

    return (best_name, best_score) if best_score >= threshold else (None, best_score)


def draw_chinese_text(img, text, position, font_size=18, color=(0, 255, 0)):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(img_pil)
        font    = (ImageFont.truetype(str(FONT_PATH), font_size)
                   if FONT_PATH.is_file()
                   else ImageFont.load_default())
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img


# ── 人体姿态检测（ONNX Runtime + YOLO Pose，兼容 Python 3.13）──────────

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

BODY_KEYPOINT_LABELS = {
    0: "鼻子", 1: "左眼", 2: "右眼", 3: "左耳", 4: "右耳",
    5: "左肩", 6: "右肩", 7: "左肘", 8: "右肘",
    9: "左腕", 10: "右腕", 11: "左髋", 12: "右髋",
    13: "左膝", 14: "右膝", 15: "左踝", 16: "右踝",
}

BODY_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

BODY_KEYPOINTS = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

_onnx_pose_session = None
_onnx_pose_ready = False

def _ensure_pose_engine(log_fn=print):
    """确保 ONNX Runtime Pose 引擎已加载"""
    global _onnx_pose_ready
    if _onnx_pose_ready:
        return True
    try:
        import onnxruntime
        _onnx_pose_ready = True
        log_fn("✅ ONNX Runtime Pose 已就绪喵~")
        return True
    except ImportError:
        log_fn("❌ 未安装 onnxruntime，请运行: pip install onnxruntime")
        return False

def _get_pose_session(log_fn=print):
    """创建并返回 ONNX Runtime YOLO Pose 推理会话"""
    import onnxruntime

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'yolo11n-pose.onnx'
    )
    if not os.path.exists(model_path):
        log_fn("⬇️  首次运行，正在下载姿态检测模型（yolo11n-pose ONNX）...")
        import urllib.request
        url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.onnx'
        try:
            urllib.request.urlretrieve(url, model_path)
            log_fn(f"✅ 模型已保存: {model_path}")
        except Exception:
            log_fn("⏳ GitHub 下载失败，尝试镜像...")
            url2 = 'https://gitee.com/mirrors_ultralytics/ultralytics-assets/releases/download/v8.3.0/yolo11n-pose.onnx'
            try:
                urllib.request.urlretrieve(url2, model_path)
                log_fn(f"✅ 模型已保存（镜像）: {model_path}")
            except Exception as e2:
                log_fn(f"❌ 模型下载失败，请手动下载 yolo11n-pose.onnx 放到项目目录: {e2}")
                return None

    if os.path.getsize(model_path) < 1024 * 1024:
        log_fn("❌ 模型文件损坏或不完整，请删除后重新下载")
        os.unlink(model_path)
        return None

    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return session

def detect_body_pose(image_bgr, log_fn=print):
    """
    检测单张图片中的人体姿态，返回列表，每个元素为:
        (x1, y1, x2, y2, keypoints)
    其中 keypoints 是长度为 17 的列表，每个元素为 (x, y, conf) 归一化坐标
    如果没有检测到人体，返回空列表
    """
    if not _ensure_pose_engine(log_fn):
        return []
    import cv2
    import numpy as np

    h, w = image_bgr.shape[:2]
    max_dim = 1280
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        h, w = image_bgr.shape[:2]

    session = _get_pose_session(log_fn)
    if session is None:
        return []

    img_size = 640
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    shape = img_rgb.shape[:2]
    r = min(img_size / shape[0], img_size / shape[1])
    new_shape = (int(shape[1] * r), int(shape[0] * r))
    pad_w = img_size - new_shape[0]
    pad_h = img_size - new_shape[1]
    top, left = pad_h // 2, pad_w // 2

    img_resized = cv2.resize(img_rgb, new_shape, interpolation=cv2.INTER_LINEAR)
    img_padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    img_padded[top:top + new_shape[1], left:left + new_shape[0]] = img_resized

    blob = img_padded.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, 0)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})

    pred = outputs[0][0]  # (56, 8400)
    pred = pred.T  # (8400, 56)

    num_keypoints = 17
    expected_cols = 4 + 1 + num_keypoints * 3

    if pred.shape[1] >= expected_cols:
        bboxes = pred[:, :4]   # cx, cy, w, h
        confidences = pred[:, 4]
        kps = pred[:, 5:5 + num_keypoints * 3].reshape(-1, num_keypoints, 3)
    elif pred.shape[1] >= 4 + num_keypoints * 3:
        bboxes = pred[:, :4]
        confidences = pred[:, 4]
        kps = pred[:, 5:5 + num_keypoints * 3].reshape(-1, num_keypoints, 3)
    else:
        log_fn("❌ 模型输出格式不匹配")
        return []

    mask = confidences > 0.25
    if not mask.any():
        return []

    persons = []
    for i in np.where(mask)[0]:
        cx, cy, bw, bh = bboxes[i]
        # 将中心点+宽高转换为左上右下坐标 (原图坐标)
        x1 = (cx - bw/2 - left) / r
        y1 = (cy - bh/2 - top) / r
        x2 = (cx + bw/2 - left) / r
        y2 = (cy + bh/2 - top) / r
        # 限制在图像范围内
        x1 = max(0, min(w, x1))
        y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        # 关键点反 letterbox
        kps_norm = []
        for kp in kps[i]:
            kx = (kp[0] - left) / r / w
            ky = (kp[1] - top) / r / h
            kx = max(0.0, min(1.0, kx))
            ky = max(0.0, min(1.0, ky))
            kps_norm.append([kx, ky, float(kp[2])])
        persons.append((int(x1), int(y1), int(x2), int(y2), kps_norm))

    return persons

def draw_body_skeleton(image_bgr, persons,
                       bbox_color=(0, 255, 0),
                       line_color=(0, 255, 0),
                       point_color=(0, 255, 255),
                       bbox_thickness=2,
                       line_thickness=2,
                       point_radius=5,
                       show_labels=True):
    """
    在图像上绘制人体矩形框 + 骨骼连线 + 关键点圆圈 + 标签
    persons: list of (x1, y1, x2, y2, keypoints)
    """
    import cv2
    h, w = image_bgr.shape[:2]

    if not persons:
        return image_bgr

    for (x1, y1, x2, y2, landmarks) in persons:
        # 1. 绘制人体边界框
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), bbox_color, bbox_thickness)

        # 2. 画骨骼连线
        for (idx1, idx2) in BODY_CONNECTIONS:
            if idx1 >= len(landmarks) or idx2 >= len(landmarks):
                continue
            kp1 = landmarks[idx1]
            kp2 = landmarks[idx2]
            if kp1[2] > 0.5 and kp2[2] > 0.5:
                pt1 = (int(kp1[0] * w), int(kp1[1] * h))
                pt2 = (int(kp2[0] * w), int(kp2[1] * h))
                cv2.line(image_bgr, pt1, pt2, line_color, line_thickness)

        # 3. 画关键点圆点
        for idx, label in BODY_KEYPOINT_LABELS.items():
            if idx >= len(landmarks):
                continue
            kp = landmarks[idx]
            if kp[2] > 0.5:
                x, y = int(kp[0] * w), int(kp[1] * h)
                cv2.circle(image_bgr, (x, y), point_radius, point_color, -1)
                cv2.circle(image_bgr, (x, y), point_radius + 1, line_color, 1)

        # 4. 画中文标签（可选）
        if show_labels:
            for idx, label in BODY_KEYPOINT_LABELS.items():
                if idx >= len(landmarks):
                    continue
                kp = landmarks[idx]
                if kp[2] > 0.5:
                    x, y = int(kp[0] * w), int(kp[1] * h)
                    image_bgr = draw_chinese_text(
                        image_bgr, label, (x + 8, y - 8),
                        font_size=12, color=(255, 255, 255)
                    )

    return image_bgr


# ── 视频/图片处理（运行于子线程）──────────────────────────────────────────────
MOVIEPY_AVAILABLE = False

def _add_audio(src_video, out_video, tmp_video):
    global MOVIEPY_AVAILABLE
    if not MOVIEPY_AVAILABLE:
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip
            MOVIEPY_AVAILABLE = True
        except (ImportError, RuntimeError):
            MOVIEPY_AVAILABLE = False

    if not MOVIEPY_AVAILABLE:
        shutil.move(tmp_video, out_video)
        return
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        vc  = VideoFileClip(src_video)
        sc  = VideoFileClip(tmp_video)
        fin = sc.with_audio(vc.audio)
        fin.write_videofile(out_video, codec="libx264", audio_codec="aac", logger=None)
        vc.close(); sc.close(); fin.close()
        if os.path.exists(tmp_video):
            os.unlink(tmp_video)
    except Exception as e:
        warnings.warn(f"音频合并失败: {e}")
        shutil.move(tmp_video, out_video)

def process_image_file(source: str, database: dict, threshold=0.45,
                       min_neighbors=3, body_mode=False,
                       log_fn=print, preview_fn=None, done_fn=None,
                       stop_event=None):
    import cv2
    import numpy as np
    try:
        if stop_event and stop_event.is_set():
            log_fn("⏹ 已停止")
            if done_fn: done_fn()
            return

        with open(source, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            log_fn("❌ 无法读取图片喵~")
            if done_fn: done_fn()
            return

        # 人体姿态检测（如果需要）并绘制框+骨骼
        persons = []
        if body_mode:
            persons = detect_body_pose(img, log_fn)
            if persons:
                log_fn(f"✅ 检测到 {len(persons)} 个人体，绘制框和骨骼...")
                img = draw_body_skeleton(img, persons, show_labels=False)
            else:
                log_fn("⚠️ 未检测到人体喵~")

        # 人脸识别（始终执行，无论 body_mode 是否开启，实现“同时检测”）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _anime_cascade.detectMultiScale(
            gray, scaleFactor=1.02, minNeighbors=min_neighbors,
            minSize=(20, 20), maxSize=(800, 800)
        )
        if body_mode:
            log_fn(f"人脸识别额外检测到 {len(faces)} 张人脸")
        else:
            log_fn(f"检测到 {len(faces)} 张人脸")

        for (x, y, w, h) in faces:
            if stop_event and stop_event.is_set():
                log_fn("⏹ 已停止")
                break
            x1 = max(0, x); y1 = max(0, y)
            x2 = min(img.shape[1], x + w)
            y2 = min(img.shape[0], y + h)
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            name, score = recognize_character(face_rgb, database, threshold,
                                              full_img=img, body_persons=persons)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            txt = f"{name} ({score:.2f})" if name else f"Unknown ({score:.2f})"
            img = draw_chinese_text(img, txt, (x1, max(0, y1 - 25)), 18, (0, 255, 0))
            log_fn(f"  {'✅' if name else '❓'} {txt}")

        if preview_fn:
            preview_fn(img)
    except Exception:
        log_fn(f"❌ 呜呜呜,(｡•́︿•̀｡)处理图片出错了:\n{traceback.format_exc()}")
    finally:
        if done_fn: done_fn()


def process_video_file(source: str, database: dict, output_path=None,
                       threshold=0.45, skip_frames=2, min_neighbors=3,
                       body_mode=False,
                       log_fn=print, preview_fn=None,
                       stop_event: threading.Event = None, done_fn=None):
    import cv2
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log_fn(f"❌ 无法打开视频喵~: {source}")
        if done_fn: done_fn()
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log_fn(f"视频信息: {w}×{h}  {fps:.1f}fps  共 {total} 帧")
    if body_mode:
        log_fn("🔍 同时开启: 人体姿态检测(框+骨骼) + 人脸角色识别")

    out = tmp_out = None
    if output_path:
        tmp_dir = Path(output_path).parent / "temp"
        tmp_dir.mkdir(exist_ok=True)
        tmp_fd, tmp_out = tempfile.mkstemp(suffix=".mp4", dir=str(tmp_dir))
        os.close(tmp_fd)
        out = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx  = 0
    found_names: set = set()
    persons = []  # 用于跨循环传递 body_persons

    try:
        while True:
            if stop_event and stop_event.is_set():
                log_fn("⏹ 已停止")
                break
            ret, frame = cap.read()
            if not ret:
                break

            if total > 0 and frame_idx % 30 == 0:
                pct = frame_idx / total * 100
                log_fn(f"进度: {pct:.1f}%  ({frame_idx}/{total})")

            # 人体姿态检测（如果启用）
            if body_mode and frame_idx % skip_frames == 0:
                persons = detect_body_pose(frame, log_fn)
                if persons:
                    frame = draw_body_skeleton(frame, persons, show_labels=False)

            # 人脸识别（始终执行，与 body_mode 无关）
            if frame_idx % skip_frames == 0:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = _anime_cascade.detectMultiScale(
                    gray, scaleFactor=1.02, minNeighbors=min_neighbors,
                    minSize=(20, 20), maxSize=(800, 800)
                )
                for (x, y, fw, fh) in faces:
                    x1 = max(0, x); y1 = max(0, y)
                    x2 = min(frame.shape[1], x + fw)
                    y2 = min(frame.shape[0], y + fh)
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    name, score = recognize_character(face_rgb, database, threshold,
                                                      full_img=frame, body_persons=persons)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    txt = f"{name} ({score:.2f})" if name else f"Unknown ({score:.2f})"
                    frame = draw_chinese_text(frame, txt, (x1, max(0, y1 - 25)), 16, (0, 255, 0))
                    if name:
                        found_names.add(name)

            if preview_fn and frame_idx % skip_frames == 0:
                preview_fn(frame.copy())

            if out:
                out.write(frame)
            frame_idx += 1

    finally:
        cap.release()
        if out:
            out.release()

    log_fn(f"✅ 处理完成: {frame_idx} 帧")
    if found_names:
        log_fn(f"识别角色: {', '.join(sorted(found_names))}")

    if output_path and tmp_out and os.path.exists(tmp_out):
        log_fn("正在合并音频...")
        _add_audio(source, output_path, tmp_out)
        log_fn(f" 已保存: {output_path}了喵~")

    if done_fn:
        done_fn()


# ══════════════════════════════════════════════════════════════════════════════
#  Tkinter GUI
# ══════════════════════════════════════════════════════════════════════════════
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}


class MoeFaceApp:
    WIN_W, WIN_H = 1024, 768

    def __init__(self):
        if DND_AVAILABLE:
            self.root = tkdnd.Tk()
        else:
            self.root = tk.Tk()

        self.root.title("MoeFace — 动漫人脸识别系统")
        self.root.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.root.minsize(800, 600)
        self.root.configure(bg="#1e1e2e")

        # 状态
        self._database: dict   = {}
        self._db_name:  str    = ""
        self._stop_evt: threading.Event = threading.Event()
        self._busy      = False
        self._preview_img = None
        self._models_ready = False

        self._auto_shutdown_var = tk.BooleanVar(value=False)
        self._body_mode_var = tk.BooleanVar(value=False)  # 人体姿态检测模式

        self._build_ui()
        self._load_models_async()

    # ── UI 构建 ────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root
        DARK   = "#1e1e2e"
        PANEL  = "#2a2a3e"
        ACCENT = "#7c3aed"
        TEXT   = "#cdd6f4"
        MUTED  = "#6c7086"

        hdr = tk.Frame(root, bg=ACCENT, height=48)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  MoeFace  动漫人脸识别",
                 font=("微软雅黑", 14, "bold"),
                 bg=ACCENT, fg="white").pack(side="left", padx=16, pady=8)

        self._status_lbl = tk.Label(hdr, text=" 正在初始化…",
                                    font=("微软雅黑", 10),
                                    bg=ACCENT, fg="#e9d5ff")
        self._status_lbl.pack(side="right", padx=16)

        body = tk.Frame(root, bg=DARK)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=PANEL, width=260)
        left.pack(side="left", fill="y", padx=(8, 0), pady=8)
        left.pack_propagate(False)

        right = tk.Frame(body, bg=DARK)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self._build_left(left, PANEL, ACCENT, TEXT, MUTED)
        self._build_right(right, DARK, PANEL, TEXT, MUTED)

    def _lbl(self, parent, text, bg, fg, font=("微软雅黑", 9), **kw):
        return tk.Label(parent, text=text, bg=bg, fg=fg, font=font, **kw)

    def _btn(self, parent, text, cmd, bg, fg="#ffffff", **kw):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg=fg, relief="flat",
                         activebackground="#6d28d9", activeforeground="white",
                         cursor="hand2", font=("微软雅黑", 9), **kw)

    def _build_left(self, parent, PANEL, ACCENT, TEXT, MUTED):
        pad = {"padx": 10, "pady": 4, "anchor": "w"}

        self._lbl(parent, "特征库", PANEL, MUTED,
                  font=("微软雅黑", 8)).pack(**pad)

        db_frame = tk.Frame(parent, bg=PANEL)
        db_frame.pack(fill="x", padx=10, pady=2)

        self._db_var = tk.StringVar(value=DEFAULT_DB_NAME)
        db_names = [DEFAULT_DB_NAME] + sorted(scan_role_folders())
        self._db_combo = ttk.Combobox(db_frame, textvariable=self._db_var,
                                      values=db_names, state="readonly", width=20)
        self._db_combo.pack(side="left", fill="x", expand=True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
                         fieldbackground="#2a2a3e", background="#2a2a3e",
                         foreground="#cdd6f4", bordercolor="#7c3aed",
                         arrowcolor="#cdd6f4")

        self._progress_frame = tk.Frame(parent, bg=PANEL)
        self._progress_frame.pack(fill="x", padx=10, pady=(4, 2))
        self._progress_label = self._lbl(self._progress_frame, 
                                          "等待加载特征库...", PANEL, MUTED,
                                          font=("微软雅黑", 8))
        self._progress_label.pack(anchor="w")
        self._progress_bar = ttk.Progressbar(self._progress_frame, 
                                              length=200, mode="determinate",
                                              style="Moe.Horizontal.TProgressbar")
        self._progress_bar.pack(fill="x", pady=2)
        self._progress_bar["maximum"] = 100
        self._progress_bar["value"] = 0

        self._rebuild_btn = self._btn(parent, " 重建特征库", self._rebuild_db,
                                      ACCENT, state="disabled")
        self._rebuild_btn.pack(fill="x", padx=10, pady=(4, 2))
        self._load_btn = self._btn(parent, " 加载特征库", self._load_db_now,
                                   "#374151", state="disabled")
        self._load_btn.pack(fill="x", padx=10, pady=2)

        self._auto_shutdown_cb = tk.Checkbutton(
            parent, text="训练完成后自动关机", variable=self._auto_shutdown_var,
            bg=PANEL, fg=TEXT, selectcolor=PANEL, font=("微软雅黑", 9),
            activebackground=PANEL, activeforeground=TEXT
        )
        self._auto_shutdown_cb.pack(anchor="w", padx=10, pady=2)

        self._body_mode_cb = tk.Checkbutton(
            parent, text="🧍 人体姿态检测模式",
            variable=self._body_mode_var,
            bg=PANEL, fg=TEXT, selectcolor=PANEL, font=("微软雅黑", 9),
            activebackground=PANEL, activeforeground="#7c3aed"
        )
        self._body_mode_cb.pack(anchor="w", padx=10, pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=8)

        self._lbl(parent, "识别阈值", PANEL, MUTED,
                  font=("微软雅黑", 8)).pack(**pad)
        th_frame = tk.Frame(parent, bg=PANEL)
        th_frame.pack(fill="x", padx=10, pady=2)
        self._threshold_var = tk.DoubleVar(value=0.45)
        th_scale = tk.Scale(th_frame, from_=0.1, to=0.95, resolution=0.01,
                            orient="horizontal", variable=self._threshold_var,
                            bg=PANEL, fg=TEXT, highlightthickness=0,
                            troughcolor="#374151", activebackground=ACCENT)
        th_scale.pack(fill="x")
        self._th_label = self._lbl(th_frame,
                                   f"当前: {self._threshold_var.get():.2f}",
                                   PANEL, MUTED, font=("微软雅黑", 8))
        self._th_label.pack(anchor="e")
        self._threshold_var.trace_add("write", lambda *_: self._th_label.config(
            text=f"当前: {self._threshold_var.get():.2f}"))

        self._lbl(parent, "视频跳帧（每N帧识别一次）", PANEL, MUTED,
                  font=("微软雅黑", 8)).pack(**pad)
        self._skip_var = tk.IntVar(value=2)

        def _validate_int(val):
            return val.isdigit() or val == ""
        vcmd = parent.register(_validate_int)
        tk.Spinbox(parent, from_=1, to=30, textvariable=self._skip_var,
                   validate="key", validatecommand=(vcmd, "%P"),
                   bg="#374151", fg=TEXT, buttonbackground="#374151",
                   relief="flat", font=("微软雅黑", 9), width=6
                   ).pack(padx=10, pady=2, anchor="w")

        self._lbl(parent, "检测灵敏度（越大误检越少）", PANEL, MUTED,
                  font=("微软雅黑", 8)).pack(**pad)
        mn_frame = tk.Frame(parent, bg=PANEL)
        mn_frame.pack(fill="x", padx=10, pady=2)
        self._min_neighbors_var = tk.IntVar(value=3)
        tk.Scale(mn_frame, from_=1, to=10, resolution=1,
                 orient="horizontal", variable=self._min_neighbors_var,
                 bg=PANEL, fg=TEXT, highlightthickness=0,
                 troughcolor="#374151", activebackground=ACCENT
                 ).pack(fill="x")
        self._mn_label = self._lbl(mn_frame,
                                   f"当前: {self._min_neighbors_var.get()}",
                                   PANEL, MUTED, font=("微软雅黑", 8))
        self._mn_label.pack(anchor="e")
        self._min_neighbors_var.trace_add("write", lambda *_: self._mn_label.config(
            text=f"当前: {self._min_neighbors_var.get()}"))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=8)

        self._lbl(parent, "保存识别视频（可选）", PANEL, MUTED,
                  font=("微软雅黑", 8)).pack(**pad)
        out_frame = tk.Frame(parent, bg=PANEL)
        out_frame.pack(fill="x", padx=10, pady=2)
        self._out_var = tk.StringVar()
        tk.Entry(out_frame, textvariable=self._out_var,
                 bg="#374151", fg=TEXT, relief="flat",
                 insertbackground=TEXT, font=("微软雅黑", 9)
                 ).pack(side="left", fill="x", expand=True)
        self._btn(out_frame, "…", self._pick_output, "#374151",
                  width=3).pack(side="left", padx=2)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=8)

        self._open_btn = self._btn(parent, "  打开图片/视频", self._open_file,
                                   ACCENT, state="disabled")
        self._open_btn.pack(fill="x", padx=10, pady=2)
        self._stop_btn = self._btn(parent, "⏹  停止处理", self._stop_processing,
                                   "#dc2626", state="disabled")
        self._stop_btn.pack(fill="x", padx=10, pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=8)

        self._btn(parent, " 管理角色别名", self._open_alias_editor,
                  "#374151").pack(fill="x", padx=10, pady=2)
        self._btn(parent, " 清空日志", self._clear_log,
                  "#374151").pack(fill="x", padx=10, pady=2)

    def _build_right(self, parent, DARK, PANEL, TEXT, MUTED):
        preview_frame = tk.Frame(parent, bg=PANEL, bd=2, relief="groove")
        preview_frame.pack(fill="both", expand=True, pady=(0, 6))

        self._drop_label = tk.Label(
            preview_frame,
            text="将图片或视频拖拽到此处\n或点击左侧按钮打开文件\n\n支持 JPG / PNG / MP4 / AVI / MKV 等",
            font=("微软雅黑", 12),
            bg=PANEL, fg=MUTED,
            justify="center"
        )
        self._drop_label.place(relx=0.5, rely=0.5, anchor="center")

        self._preview_canvas = tk.Canvas(preview_frame, bg=PANEL,
                                         highlightthickness=0)
        self._preview_canvas.pack(fill="both", expand=True)

        if DND_AVAILABLE:
            self._preview_canvas.drop_target_register(tkdnd.DND_FILES)
            self._preview_canvas.dnd_bind("<<Drop>>", self._on_drop)
        else:
            self._drop_label.config(
                text="主人,将图片或视频拖拽到此处喵~（需安装 tkinterdnd2）\n或点击左侧「打开图片/视频」按钮\n\n支持 JPG / PNG / MP4 / AVI / MKV 等"
            )

        log_frame = tk.Frame(parent, bg=PANEL)
        log_frame.pack(fill="x", ipady=4)
        tk.Label(log_frame, text="运行日志", bg=PANEL, fg=MUTED,
                 font=("微软雅黑", 8)).pack(anchor="w", padx=6, pady=(4, 0))
        self._log_box = scrolledtext.ScrolledText(
            log_frame, height=8, bg="#12121e", fg="#a6e3a1",
            font=("Consolas", 9), relief="flat",
            state="disabled", wrap="word"
        )
        self._log_box.pack(fill="x", padx=6, pady=4)

    # ── 模型初始化（异步）────────────────────────────────────────────────
    def _load_models_async(self):
        def _run():
            ok = _ensure_models(self._log)
            self._models_ready = ok
            if ok:
                self._set_status("✅ 就绪，请拖入文件")
                self.root.after(0, self._enable_controls)
            else:
                self._set_status("❌ 模型加载失败(｡•́︿•̀｡)，请检查依赖")
        threading.Thread(target=_run, daemon=True).start()

    def _enable_controls(self):
        """模型加载完成后启用相关按钮"""
        self._rebuild_btn.config(state="normal")
        self._load_btn.config(state="normal")
        self._open_btn.config(state="normal")

    # ── 日志 ──────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        def _upd():
            self._log_box.config(state="normal")
            self._log_box.insert("end", msg + "\n")
            self._log_box.see("end")
            self._log_box.config(state="disabled")
        self.root.after(0, _upd)

    def _clear_log(self):
        self._log_box.config(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.config(state="disabled")

    def _set_status(self, text: str):
        self.root.after(0, lambda: self._status_lbl.config(text=text))

    # ── 预览帧更新 ───────────────────────────────────────────────────────
    def _show_frame_cv(self, cv_img):
        import cv2
        from PIL import Image, ImageTk
        def _upd():
            cw = self._preview_canvas.winfo_width()
            ch = self._preview_canvas.winfo_height()
            if cw < 10 or ch < 10:
                return
            h, w = cv_img.shape[:2]
            scale = min(cw / w, ch / h, 1.0)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(cv_img, (nw, nh))
            rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tk_img  = ImageTk.PhotoImage(pil_img)
            self._preview_canvas.delete("all")
            self._preview_canvas.create_image(cw // 2, ch // 2,
                                              anchor="center", image=tk_img)
            self._preview_canvas._tk_img = tk_img
            self._drop_label.place_forget()
        self.root.after(0, _upd)

    # ── 特征库操作 ────────────────────────────────────────────────────────
    def _progress_update(self, current: int, total: int, elapsed: float, db_name: str):
        if total <= 0:
            return
        pct = min(current / total * 100, 100)
        if current > 0 and current < total:
            eta_sec = (elapsed / current) * (total - current)
            if eta_sec >= 60:
                eta_str = f"约 {int(eta_sec // 60)} 分 {int(eta_sec % 60)} 秒"
            else:
                eta_str = f"约 {int(eta_sec)} 秒"
        else:
            eta_str = "即将完成"
        name_str = f" [{db_name}]" if db_name else ""
        text = f"构建中{name_str} {current}/{total} ({pct:.0f}%) 剩余: {eta_str}"
        def _upd():
            self._progress_bar["value"] = pct
            self._progress_label.config(text=text)
        self.root.after(0, _upd)

    def _load_db_now(self):
        if not self._models_ready:
            self._log("请等待模型加载完成后再操作")
            return
        db_name = self._db_var.get()
        self._log(f"加载特征库: {db_name}")
        def _reset():
            self._progress_bar["value"] = 0
            self._progress_label.config(text="准备加载...")
        self.root.after(0, _reset)

        def _run():
            db, built = get_or_build_database(db_name, force_rebuild=False,
                                              log_fn=self._log, progress_fn=self._progress_update)
            self._database = db
            self._db_name  = db_name
            cnt = len(db)
            self._log(f"特征库就绪: {cnt} 个角色")
            self._set_status(f"✅ 特征库已加载（{cnt} 角色）")
            def _done():
                self._progress_bar["value"] = 100
                self._progress_label.config(text=f"完成！共 {cnt} 个角色")
            self.root.after(0, _done)

            if built and self._auto_shutdown_var.get():
                self._log("训练完成，即将关机...")
                self._shutdown_system()
        threading.Thread(target=_run, daemon=True).start()

    def _rebuild_db(self):
        if not self._models_ready:
            self._log("请等待模型加载完成后再操作")
            return
        db_name = self._db_var.get()
        self._log(f"强制重建: {db_name}")
        def _reset():
            self._progress_bar["value"] = 0
            self._progress_label.config(text="准备重建...")
        self.root.after(0, _reset)

        def _run():
            db, built = get_or_build_database(db_name, force_rebuild=True,
                                              log_fn=self._log, progress_fn=self._progress_update)
            self._database = db
            self._db_name  = db_name
            self._log(f"✅ 重建完成: {len(db)} 个角色")
            self._set_status(f"✅ 特征库已重建（{len(db)} 角色）")
            def _done():
                self._progress_bar["value"] = 100
                self._progress_label.config(text=f"完成！共 {len(db)} 个角色")
            self.root.after(0, _done)

            if built and self._auto_shutdown_var.get():
                self._log("训练完成，即将关机...")
                self._shutdown_system()
        threading.Thread(target=_run, daemon=True).start()

    # ── 关机功能 ─────────────────────────────────────────────────────────
    def _shutdown_system(self):
        import platform
        import subprocess
        system = platform.system()
        try:
            if system == "Windows":
                subprocess.run(["shutdown", "/s", "/t", "0"], check=True)
            elif system == "Linux":
                subprocess.run(["shutdown", "-h", "now"], check=True)
            elif system == "Darwin":
                subprocess.run(["shutdown", "-h", "now"], check=True)
            else:
                self._log(f"不支持自动关机的系统: {system}", "error")
                return
            self._log("关机命令已执行，系统即将关闭...")
        except Exception as e:
            self._log(f"关机失败: {e}，请手动关机或检查权限", "error")

    # ── 文件处理 ─────────────────────────────────────────────────────────
    def _pick_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 视频", "*.mp4"), ("AVI 视频", "*.avi")]
        )
        if path:
            self._out_var.set(path)

    def _open_file(self):
        if not self._models_ready:
            self._log("模型未就绪，请稍候...")
            return
        path = filedialog.askopenfilename(
            filetypes=[
                ("图片/视频", " ".join(f"*{e}" for e in IMAGE_EXTS | VIDEO_EXTS)),
                ("图片",  " ".join(f"*{e}" for e in IMAGE_EXTS)),
                ("视频",  " ".join(f"*{e}" for e in VIDEO_EXTS)),
                ("全部",  "*.*"),
            ]
        )
        if path:
            self._dispatch_file(path)

    def _on_drop(self, event):
        import re
        raw = event.data.strip()
        matches = re.findall(r'\{([^}]+)\}|(\S+)', raw)
        if matches:
            path = (matches[0][0] or matches[0][1]).strip()
        else:
            path = raw
        self._dispatch_file(path)

    def _dispatch_file(self, path: str):
        if not self._models_ready:
            self._log("模型尚未加载完成，请稍候…")
            return
        if self._busy:
            self._log("⚠️  正在处理中，请等待完成或点击停止")
            return

        suffix = Path(path).suffix.lower()
        suggested = get_db_name_from_filename(Path(path).name)

        self._set_busy(True)
        self._stop_evt.clear()

        def _load_and_run():
            db = self._database
            db_name = self._db_name
            if suggested != db_name or not db:
                self._log(f" 自动选择特征库: {suggested}")
                self.root.after(0, lambda: self._db_var.set(suggested))
                db, _ = get_or_build_database(suggested, log_fn=self._log, stop_event=self._stop_evt)
                self._database = db
                self._db_name  = suggested

            if not db:
                self._log("❌ 特征库为空，无法识别。请先建库或检查 ./data 文件夹")
                self._set_busy(False)
                return

            mn = self._min_neighbors_var.get()
            body_mode = self._body_mode_var.get()

            if suffix in IMAGE_EXTS:
                self._log(f"\n🖼  图片: {path}")
                process_image_file(
                    source=path,
                    database=db,
                    threshold=self._threshold_var.get(),
                    min_neighbors=mn,
                    body_mode=body_mode,
                    log_fn=self._log,
                    preview_fn=self._show_frame_cv,
                    stop_event=self._stop_evt,
                    done_fn=lambda: self._set_busy(False),
                )
            elif suffix in VIDEO_EXTS:
                self._log(f"\n🎬  视频: {path}")
                out = self._out_var.get().strip() or None
                process_video_file(
                    source=path,
                    database=db,
                    output_path=out,
                    threshold=self._threshold_var.get(),
                    skip_frames=self._skip_var.get(),
                    min_neighbors=mn,
                    body_mode=body_mode,
                    log_fn=self._log,
                    preview_fn=self._show_frame_cv,
                    stop_event=self._stop_evt,
                    done_fn=lambda: self._set_busy(False),
                )
            else:
                self._log(f"⚠️  不支持的文件格式: {suffix}")
                self._set_busy(False)

        threading.Thread(target=_load_and_run, daemon=True).start()

    def _stop_processing(self):
        self._stop_evt.set()
        self._log("⏹ 已发送停止信号…")

    def _set_busy(self, busy: bool):
        self._busy = busy
        def _upd():
            state = "normal" if busy else "disabled"
            self._stop_btn.config(state=state)
        self.root.after(0, _upd)

    # ── 别名编辑器 ────────────────────────────────────────────────────────
    def _open_alias_editor(self):
        win = tk.Toplevel(self.root)
        win.title("角色别名管理")
        win.geometry("600x500")
        win.configure(bg="#1e1e2e")

        tk.Label(win, text="编辑 cname/name.json（每行一个别名，逗号分隔）",
                 bg="#1e1e2e", fg="#cdd6f4",
                 font=("微软雅黑", 10)).pack(pady=8)

        txt = scrolledtext.ScrolledText(win, bg="#12121e", fg="#cdd6f4",
                                        font=("Consolas", 9), relief="flat")
        txt.pack(fill="both", expand=True, padx=10, pady=4)

        try:
            with open(CNAME_PATH, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = "[]"
        txt.insert("1.0", content)

        def _save():
            try:
                data = json.loads(txt.get("1.0", "end"))
                with open(CNAME_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                global ALIAS_MAP
                ALIAS_MAP = load_alias_map()
                messagebox.showinfo("保存成功", "别名配置已更新！", parent=win)
                win.destroy()
            except json.JSONDecodeError as e:
                messagebox.showerror("JSON 格式错误", str(e), parent=win)

        btn_frame = tk.Frame(win, bg="#1e1e2e")
        btn_frame.pack(pady=6)
        tk.Button(btn_frame, text="💾 保存", command=_save,
                  bg="#7c3aed", fg="white", relief="flat",
                  font=("微软雅黑", 10), cursor="hand2",
                  padx=16, pady=4).pack(side="left", padx=8)
        tk.Button(btn_frame, text="取消", command=win.destroy,
                  bg="#374151", fg="white", relief="flat",
                  font=("微软雅黑", 10), cursor="hand2",
                  padx=16, pady=4).pack(side="left", padx=8)

    # ── 主循环 ────────────────────────────────────────────────────────────
    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI 模式（终端图形化界面）- 增强进度条显示
# ══════════════════════════════════════════════════════════════════════════════

class CLIColors:
    """ANSI 终端颜色"""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    LGRAY   = "\033[90m"
    LRED    = "\033[91m"
    LGREEN  = "\033[92m"
    LYELLOW = "\033[93m"
    LBLUE   = "\033[94m"
    LMAGENTA= "\033[95m"
    LCYAN   = "\033[96m"
    LWHITE  = "\033[97m"

    BGBLACK  = "\033[40m"
    BGRED    = "\033[41m"
    BGGREEN  = "\033[42m"
    BGYELLOW = "\033[43m"
    BGBLUE   = "\033[44m"
    BGMAGENTA= "\033[45m"
    BGCYAN   = "\033[46m"
    BGWHITE  = "\033[47m"

    HEADER  = BOLD + LGRAY
    TITLE   = BOLD + MAGENTA
    SUCCESS = BOLD + LGREEN
    WARNING = BOLD + LYELLOW
    ERROR   = BOLD + LRED
    INFO    = BOLD + LCYAN
    MUTED   = DIM + LGRAY
    PROGRESS_BG = BGCYAN
    PROGRESS_FG = BGGREEN

    @staticmethod
    def p(text, color="", end="\n"):
        if color:
            print(f"{color}{text}{CLIColors.RESET}", end=end)
        else:
            print(text, end=end)

    @staticmethod
    def clear_line():
        sys.stdout.write("\033[2K\r")
        sys.stdout.flush()

    @staticmethod
    def progress_bar(percent: float, width: int = 50,
                     fg_color=None, bg_color=None):
        """生成进度条字符串（更明显）"""
        filled = int(width * percent)
        empty  = width - filled
        bar = "█" * filled + "░" * empty
        percent_str = f"{percent * 100:5.1f}%"
        fg = fg_color or CLIColors.LGREEN
        bg = bg_color or CLIColors.PROGRESS_BG
        return f"{fg}{CLIColors.BOLD}{bar}{CLIColors.RESET} {CLIColors.LYELLOW}{percent_str}{CLIColors.RESET}"


class MoeFaceCLI:
    """CLI 模式主类"""

    BANNER = rf"""{CLIColors.TITLE}
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ███╗   ███╗  ██████╗ ███████╗ ███████╗ █████╗  ██████╗███████╗ ║
║   ████╗ ████║ ██╔═══██╗██╔════╝ ██╔════╝██╔══██╗██╔════╝██╔════╝ ║
║   ██╔████╔██║ ██║   ██║█████╗   █████╗  ███████║██║     █████╗   ║
║   ██║╚██╔╝██║ ██║   ██║██╔══╝   ██╔══╝  ██╔══██║██║     ██╔══╝   ║
║   ██║ ╚═╝ ██║ ╚██████╔╝███████╗ ██║     ██║  ██║╚██████╗███████╗ ║
║   ╚═╝     ╚═╝  ╚═════╝ ╚══════╝ ╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝ ║
║                                                                  ║
║               Anime Face Recognition System                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    {CLIColors.RESET}"""

    def __init__(self, args):
        self.args      = args
        self.db        = {}
        self.stats     = {"total_frames": 0, "processed": 0,
                          "found_names": {}, "errors": 0}
        self.stop_evt  = threading.Event()
        self._console_height = 0
        self._last_pct = -1.0
        self._last_eta_str = ""

    def _log(self, msg: str, level: str = "info"):
        """带颜色的日志输出"""
        now  = datetime.now().strftime("%H:%M:%S")
        icons = {"info": "ℹ", "ok": "✓", "warn": "⚠", "error": "✗",
                 "skip": "➜", "done": "✓"}
        icon  = icons.get(level, "·")
        color = {"info": CLIColors.INFO, "ok": CLIColors.SUCCESS,
                 "warn": CLIColors.WARNING, "error": CLIColors.ERROR,
                 "skip": CLIColors.MUTED, "done": CLIColors.SUCCESS}.get(level, "")
        CLIColors.p(f"  {icon}  {msg}", color)

    def _header(self, title: str):
        """分组标题"""
        CLIColors.p(f"\n{CLIColors.HEADER}{'═' * 60}{CLIColors.RESET}")
        CLIColors.p(f"{CLIColors.TITLE}  {title}{CLIColors.RESET}")
        CLIColors.p(f"{CLIColors.HEADER}{'─' * 60}{CLIColors.RESET}")

    def _progress(self, current: int, total: int, elapsed: float = 0, label: str = ""):
        """更新进度条（同一行覆盖）- 更明显的显示"""
        if total <= 0:
            return
        pct = min(current / total, 1.0)
        
        eta_text = ""
        if elapsed > 0 and current > 0 and current < total:
            eta_sec = elapsed * (total - current) / current
            if eta_sec >= 3600:
                eta_text = f"⏱️ 剩余: {int(eta_sec // 3600)}时{int((eta_sec % 3600) // 60)}分"
            elif eta_sec >= 60:
                eta_text = f"⏱️ 剩余: {int(eta_sec // 60)}分{int(eta_sec % 60)}秒"
            else:
                eta_text = f"⏱️ 剩余: {int(eta_sec)}秒"
        
        count_text = f"📊 {current}/{total}"
        bar = CLIColors.progress_bar(pct, width=40)
        label_text = f" 🎭 {label[:20]}" if label and label.strip() else ""
        line = f"\r{CLIColors.LCYAN}{bar}{CLIColors.RESET} {CLIColors.LYELLOW}{count_text}{CLIColors.RESET} {CLIColors.LGREEN}{eta_text}{CLIColors.RESET}{label_text}"
        
        CLIColors.clear_line()
        print(line, end="", flush=True)
        self._last_pct = pct

    def _stat_row(self, name: str, count: int, color: str = ""):
        """统计行"""
        fg = color or CLIColors.SUCCESS
        CLIColors.p(f"  {CLIColors.MUTED}├─{CLIColors.RESET} {fg}{name}{CLIColors.RESET}"
                    f" {CLIColors.MUTED}({count} 次){CLIColors.RESET}")

    def run(self):
        print(self.BANNER)

        self._header("① 加载模型")
        import torch
        device_info = "NVIDIA GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        CLIColors.p(f"  🔧 设备: {CLIColors.LGREEN}{device_info}{CLIColors.RESET}")
        
        if not _ensure_models(lambda m: self._log(m, "info" if "✓" in m or "✅" in m else "warn")):
            self._log("模型加载失败，主人～(｡•́︿•̀｡)", "error")
            sys.exit(1)
        
        CLIColors.p(f"  ✅ 模型加载完成！{CLIColors.LGREEN}(°▽°)/{CLIColors.RESET}")

        self._header("② 加载特征库")
        db_name = self.args.get("db_name") or DEFAULT_DB_NAME
        CLIColors.p(f"  📚 特征库: {CLIColors.LCYAN}{db_name}{CLIColors.RESET}")
        force_rebuild = self.args.get("rebuild", False)
        
        if force_rebuild:
            CLIColors.p(f"  🔨 强制重建模式开启{CLIColors.RESET}")
        
        CLIColors.p("")
        
        self.db, _ = get_or_build_database(
            db_name, force_rebuild=force_rebuild,
            log_fn=self._log,
            progress_fn=lambda cur, tot, elapsed=0, lbl="": self._progress(cur, tot, elapsed, lbl)
        )
        
        CLIColors.p("")
        if not self.db:
            self._log("特征库为空，请检查 ./data 文件夹或添加角色图片", "error")
            sys.exit(1)
        
        CLIColors.p(f"\n  {CLIColors.SUCCESS}✨ 已加载 {len(self.db)} 个角色特征，就绪~ ✨{CLIColors.RESET}")

        source = self.args.get("source")
        output = self.args.get("output")
        camera = self.args.get("camera", False)

        if camera:
            self._header("③ 摄像头模式")
            self._log("按 Ctrl+C 或 Ctrl+Break 停止", "warn")
            self._run_camera()
        elif source:
            suffix = Path(source).suffix.lower()
            if suffix in IMAGE_EXTS:
                self._header("③ 识别图片")
                self._run_image(source)
            elif suffix in VIDEO_EXTS:
                self._header("③ 识别视频")
                self._log(f"📹 输入: {source}")
                self._log(f"💾 输出: {output or '仅预览（无保存）'}")
                self._run_video(source, output)
            else:
                self._log(f"不支持的文件格式: {suffix}", "error")
                sys.exit(1)
        else:
            self._log("请指定 --source（文件路径）或 --camera（摄像头模式）", "error")
            self._print_help()
            sys.exit(1)

        self._print_report()

    def _run_image(self, path: str):
        """识别单张图片（同时人体框+骨骼 + 人脸识别）"""
        self._log(f"📸 文件: {CLIColors.LCYAN}{path}{CLIColors.RESET}")
        threshold    = self.args.get("threshold", 0.45)
        min_neighbors = self.args.get("min_neighbors", 3)
        body_mode    = self.args.get("body", False)

        import cv2, numpy as np

        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            self._log("无法读取图片喵~", "error")
            return

        # 人体姿态检测（如果需要）并绘制框+骨骼
        persons = []
        if body_mode:
            persons = detect_body_pose(img, self._log)
            if persons:
                self._log(f"✅ 检测到 {len(persons)} 个人体，绘制框和骨骼...")
                img = draw_body_skeleton(img, persons, show_labels=False)
            else:
                self._log("⚠️ 未检测到人体喵~", "warn")

        # 人脸识别（始终执行，无论 body_mode）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _anime_cascade.detectMultiScale(
            gray, scaleFactor=1.02, minNeighbors=min_neighbors,
            minSize=(20, 20), maxSize=(800, 800)
        )
        if body_mode:
            self._log(f"人脸识别额外检测到 {len(faces)} 张人脸")
        else:
            self._log(f"检测到 {len(faces)} 张人脸")

        for (x, y, w, h) in faces:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img.shape[1], x+w), min(img.shape[0], y+h)
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            name, score = recognize_character(face_rgb, self.db, threshold,
                                              full_img=img, body_persons=persons)
            tag = CLIColors.SUCCESS if name else CLIColors.MUTED
            self._log(f"  {tag}{name or '未知'}{CLIColors.RESET} "
                      f"{CLIColors.MUTED}({score:.2f}){CLIColors.RESET} "
                      f"@ ({x1},{y1},{w}×{h})", "ok" if name else "skip")
            if name:
                self.stats["found_names"][name] = \
                    self.stats["found_names"].get(name, 0) + 1

        output = self.args.get("output")
        if output:
            cv2.imwrite(output, img)
            self._log(f"已保存标注图: {output}", "ok")

    def _run_video(self, source: str, output_path: str):
        """识别视频文件（同时人体框+骨骼 + 人脸识别）"""
        threshold    = self.args.get("threshold", 0.45)
        skip_frames  = self.args.get("skip_frames", 2)
        min_neighbors = self.args.get("min_neighbors", 3)
        body_mode    = self.args.get("body", False)

        import cv2, numpy as np

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self._log(f"无法打开视频喵~: {source}", "error")
            return

        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._log(f"🎬 视频: {w}×{h}  {fps:.1f}fps  共 {total} 帧")
        if body_mode:
            self._log("🔍 同时开启: 人体姿态检测(框+骨骼) + 人脸角色识别")
        CLIColors.p("")

        out = tmp_out = None
        if output_path:
            tmp_dir = Path(output_path).parent / "temp"
            tmp_dir.mkdir(exist_ok=True)
            tmp_fd, tmp_out = tempfile.mkstemp(suffix=".mp4", dir=str(tmp_dir))
            os.close(tmp_fd)
            out = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, (w, h))

        frame_idx = 0
        persons = []  # 用于跨循环传递 body_persons
        try:
            while True:
                if self.stop_evt.is_set():
                    CLIColors.p(f"\n  {CLIColors.WARNING}⏹ 已停止{CLIColors.RESET}")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                self.stats["total_frames"] += 1

                if total > 0:
                    self._progress(frame_idx, total, 0, f"帧 {frame_idx}/{total}")

                # 人体姿态检测（如果启用）
                if body_mode and frame_idx % skip_frames == 0:
                    persons = detect_body_pose(frame, self._log)
                    if persons:
                        frame = draw_body_skeleton(frame, persons, show_labels=False)

                # 人脸识别（始终执行）
                if frame_idx % skip_frames == 0:
                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = _anime_cascade.detectMultiScale(
                        gray, scaleFactor=1.02, minNeighbors=min_neighbors,
                        minSize=(20,20), maxSize=(800,800)
                    )
                    for (x, y, fw, fh) in faces:
                        x1, y1 = max(0,x), max(0,y)
                        x2, y2 = min(frame.shape[1],x+fw), min(frame.shape[0],y+fh)
                        face = frame[y1:y2, x1:x2]
                        if face.size == 0:
                            continue
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        name, score = recognize_character(face_rgb, self.db, threshold,
                                                          full_img=frame, body_persons=persons)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        frame = draw_chinese_text(frame,
                            f"{name or '?'} ({score:.2f})",
                            (x1, max(0,y1-25)), 16, (0,255,0))
                        if name:
                            self.stats["found_names"][name] = \
                                self.stats["found_names"].get(name, 0) + 1

                    self.stats["processed"] += 1

                if out:
                    out.write(frame)
                frame_idx += 1

        finally:
            cap.release()
            if out:
                out.release()

        CLIColors.clear_line()
        CLIColors.p(f"\n  {CLIColors.SUCCESS}✓ 处理完成{CLIColors.RESET}："
                    f" {frame_idx} 帧，识别了 {self.stats['processed']} 个关键帧")

        if output_path and tmp_out and os.path.exists(tmp_out):
            self._log("正在合并音频...")
            _add_audio(source, output_path, tmp_out)
            self._log(f"已保存: {output_path}", "ok")

    def _run_camera(self):
        """摄像头实时识别（同时人体框+骨骼 + 人脸识别）"""
        threshold    = self.args.get("threshold", 0.45)
        min_neighbors = self.args.get("min_neighbors", 3)
        body_mode    = self.args.get("body", False)

        import cv2
        camera_id = self.args.get("camera_id", 0)
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self._log(f"无法打开摄像头 {camera_id}喵~", "error")
            return

        self._log(f"📷 摄像头 {camera_id} 已启动，按 Q 退出")
        if body_mode:
            self._log("🔍 同时开启: 人体姿态检测(框+骨骼) + 人脸角色识别")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay_ms = max(int(1000 / fps), 1)

        try:
            while not self.stop_evt.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue

                persons = []
                if body_mode:
                    persons = detect_body_pose(frame, self._log)
                    if persons:
                        frame = draw_body_skeleton(frame, persons, show_labels=False)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = _anime_cascade.detectMultiScale(
                    gray, scaleFactor=1.02, minNeighbors=min_neighbors,
                    minSize=(20,20), maxSize=(800,800)
                )
                for (x, y, fw, fh) in faces:
                    x1, y1 = max(0,x), max(0,y)
                    x2, y2 = min(frame.shape[1],x+fw), min(frame.shape[0],y+fh)
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    name, score = recognize_character(face_rgb, self.db, threshold,
                                                      full_img=frame, body_persons=persons)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    frame = draw_chinese_text(frame,
                        f"{name or '?'} ({score:.2f})",
                        (x1, max(0,y1-25)), 14, (0,255,0))

                cv2.imshow("MoeFace CLI (按 Q 退出)", frame)
                if cv2.waitKey(delay_ms) & 0xFF in (ord("q"), ord("Q")):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._log("摄像头已关闭", "done")

    def _print_report(self):
        """打印识别报告"""
        if not self.stats["found_names"]:
            CLIColors.p("")
            self._log("未识别到任何已知角色喵~", "warn")
            return

        CLIColors.p("")
        self._header("④ 识别报告")
        CLIColors.p(f"  📊 总帧数: {CLIColors.LCYAN}{self.stats['total_frames']}{CLIColors.RESET}，"
                    f"关键帧: {CLIColors.LCYAN}{self.stats['processed']}{CLIColors.RESET}，"
                    f"识别次数: {CLIColors.SUCCESS}{sum(self.stats['found_names'].values())}{CLIColors.RESET}")
        CLIColors.p(f"  🎭 角色分布:")
        sorted_names = sorted(self.stats["found_names"].items(),
                             key=lambda x: x[1], reverse=True)
        max_name_len = max(len(name) for name, _ in sorted_names) if sorted_names else 0
        for name, count in sorted_names:
            pct = count / sum(self.stats["found_names"].values()) * 100
            bar_width = 30
            bar_len = int(bar_width * pct / 100)
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            CLIColors.p(f"  {CLIColors.SUCCESS}{name:<{max_name_len}}{CLIColors.RESET}"
                        f" {CLIColors.LGREEN}{bar}{CLIColors.RESET} "
                        f"{CLIColors.LYELLOW}{count} 次 ({pct:5.1f}%){CLIColors.RESET}")
        CLIColors.p("")
        self._log("识别完成~ 主人再见喵！(｡•́︿•̀｡)", "done")

    def _print_help(self):
        """打印帮助"""
        print(f"""
{CLIColors.TITLE}用法:{CLIColors.RESET}
  {CLIColors.LCYAN}python recognize.py --mode cli --source <文件>{CLIColors.RESET}
  {CLIColors.LCYAN}python recognize.py --mode gui{CLIColors.RESET}

{CLIColors.TITLE}CLI 参数:{CLIColors.RESET}
  {CLIColors.LGREEN}--mode{CLIColors.RESET}         运行模式: gui / cli (默认: gui)
  {CLIColors.LGREEN}--source{CLIColors.RESET}       视频或图片路径
  {CLIColors.LGREEN}--camera{CLIColors.RESET}       启用摄像头模式
  {CLIColors.LGREEN}--camera-id{CLIColors.RESET}    摄像头 ID (默认: 0)
  {CLIColors.LGREEN}--output{CLIColors.RESET}       输出视频/图片路径
  {CLIColors.LGREEN}--db-name{CLIColors.RESET}      特征库名称 (默认: 全部特征库)
  {CLIColors.LGREEN}--threshold{CLIColors.RESET}    识别阈值 0.1~0.95 (默认: 0.45)
  {CLIColors.LGREEN}--skip-frames{CLIColors.RESET}  视频跳帧数 (默认: 2)
  {CLIColors.LGREEN}--min-neighbors{CLIColors.RESET}检测灵敏度 1~10 (默认: 3)
  {CLIColors.LGREEN}--rebuild{CLIColors.RESET}      强制重建特征库
  {CLIColors.LGREEN}--body{CLIColors.RESET}         启用人体姿态检测（框+骨骼，同时识别人脸角色）
  {CLIColors.LGREEN}--list{CLIColors.RESET}         列出所有可用特征库

{CLIColors.TITLE}示例:{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode cli --source 视频.mp4 --output out.mp4 --body{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode cli --camera --threshold 0.6 --body{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode cli --source 图片.jpg --output annotated.jpg --body{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode cli --source 视频.mp4 --db-name 永雏塔菲 --body{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode gui{CLIColors.RESET}
""")


def _parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        prog="MoeFace",
        description="MoeFace 动漫人脸识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python recognize.py --mode gui  （启动 Tkinter 界面）\n"
               "       python recognize.py --mode cli --source video.mp4 --output out.mp4"
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["gui", "cli"],
        default="gui",
        help="运行模式: gui=Tkinter桌面界面(默认), cli=终端界面"
    )

    parser.add_argument(
        "--source", "-s",
        help="视频或图片文件路径（支持 JPG/PNG/MP4/AVI/MKV 等，仅 --mode cli 生效）"
    )
    parser.add_argument(
        "--camera", "-c",
        action="store_true",
        help="启用摄像头模式（仅 --mode cli 生效）"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="摄像头设备 ID（默认: 0）"
    )

    parser.add_argument(
        "--output", "-o",
        help="输出视频/图片路径（默认: 仅预览）"
    )
    parser.add_argument(
        "--db-name",
        default=DEFAULT_DB_NAME,
        help="特征库名称（默认: 全部特征库）"
    )

    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.45,
        help="识别阈值，越高越严格（默认: 0.45）"
    )
    parser.add_argument(
        "--skip-frames", "-k",
        type=int,
        default=2,
        help="视频跳帧数，越大处理越快（默认: 2）"
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=3,
        help="人脸检测灵敏度，越大误检越少（默认: 3）"
    )
    parser.add_argument(
        "--rebuild", "-r",
        action="store_true",
        help="强制重建特征库"
    )
    parser.add_argument(
        "--body",
        action="store_true",
        help="启用人体姿态检测模式（绘制边界框+骨骼连线，同时识别人脸角色）"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有可用特征库"
    )

    return parser.parse_args()


# ── 入口 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import cv2   # 预先导入用于线程本地存储

    args = _parse_args()

    if args.list:
        print(f"\n{CLIColors.TITLE}可用特征库:{CLIColors.RESET}")
        db_names = sorted(scan_role_folders())
        print(f"  {CLIColors.GREEN}{DEFAULT_DB_NAME}{CLIColors.RESET}（默认）")
        for name in db_names:
            print(f"  {CLIColors.CYAN}{name}{CLIColors.RESET}")
        print("")
        sys.exit(0)

    if args.mode == "gui":
        app = MoeFaceApp()
        app.run()
    else:
        cli_args = {
            "source":       args.source,
            "output":       args.output,
            "db_name":      args.db_name,
            "threshold":    args.threshold,
            "skip_frames":  args.skip_frames,
            "min_neighbors": args.min_neighbors,
            "rebuild":      args.rebuild,
            "camera":       args.camera,
            "camera_id":    args.camera_id,
            "body":         args.body,
        }
        cli = MoeFaceCLI(cli_args)
        try:
            cli.run()
        except KeyboardInterrupt:
            CLIColors.p(f"\n{CLIColors.WARNING}⏹ 已中断{CLIColors.RESET}")
            cli.stop_evt.set()
            sys.exit(0)