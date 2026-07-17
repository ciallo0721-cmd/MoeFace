"""
MoeFace — 动漫人脸识别系统 / VTuber 二次元角色识别工具
基于 FaceNet + OpenCV 实现动漫人脸检测与特征匹配
支持 VTuber / 虚拟主播识别，本地运行保护隐私
GUI 版本 (Tkinter) + CLI 版本 (终端图形化)
"""

"""
带"～(｡•́︿•̀｡)"是严重错误
"""
# ── 抑制 MediaPipe 日志/遥测（必须在 import mediapipe 之前设置）──
import os as _os
_os.environ.setdefault('MEDIAPIPE_DISABLE_LOGGING', '1')
_os.environ.setdefault('GLOG_minloglevel', '2')
_os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
del _os

# ── 版本信息（功能9：自动更新检查）─────────────────────────────────────────
VERSION = "3.3.0"
VERSION_URL = "https://api.github.com/repos/ciallo0721-cmd/MoeFace/releases/latest"
VERSION_DOWNLOAD_URL = "https://github.com/ciallo0721-cmd/MoeFace/releases"

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

# ── AI 识别模块（2026-06-29 新增）──────────────────────────────────────
_AI_MODULES_LOADED = False
_EMOTION_MODULE = None
_SPEECH_MODULE = None
_NSFW_MODULE = None
_AI_MODULE_LOCK = threading.Lock()


def _ensure_ai_modules(light=False, log_fn=print):
    """懒加载 AI 识别模块（线程安全）"""
    global _AI_MODULES_LOADED, _EMOTION_MODULE, _SPEECH_MODULE, _NSFW_MODULE
    if _AI_MODULES_LOADED:
        return True
    with _AI_MODULE_LOCK:
        if _AI_MODULES_LOADED:
            return True
        try:
            from modules import EmotionRecognizer, SpeechRecognizer, NSFWDetector

            _EMOTION_MODULE = EmotionRecognizer()
            _EMOTION_MODULE.ensure_initialized(log_fn)
            _NSFW_MODULE = NSFWDetector()
            _NSFW_MODULE.ensure_initialized(log_fn)

            if not light:
                _SPEECH_MODULE = SpeechRecognizer(model_size="base")
                _SPEECH_MODULE.ensure_initialized(log_fn)

            _AI_MODULES_LOADED = True
            log_fn("✅ AI 识别模块已就绪（情绪/语音/NSFW）" if not light
                   else "✅ AI 轻量模块已就绪（情绪/NSFW）")
            return True
        except Exception as e:
            log_fn(f"⚠️ AI 模块加载失败: {e}")
            return False


# 通用导入（供 CLI/GUI 使用）
from modules.base import AIResult, AIResultCollection


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

CASCADE_PATH = RESOURCE_DIR / "models" / "lbpcascade_animeface.xml"
FONT_PATH    = RESOURCE_DIR / "models" / "simhei.ttf"
FEATURES_DIR = BASE_DIR / "features"
DATA_DIR     = BASE_DIR / "data"
CNAME_PATH   = RESOURCE_DIR / "cname" / "name.json"
DEFAULT_DB_NAME = "全部特征库"

# ── 负面特征库 ──────────────────────────────────────────────────────────
NEGATIVE_DIR        = DATA_DIR / "负面特征"
NEGATIVE_THRESHOLD  = 0.30   # 负面样本匹配阈值

# ── 新 .moe 文本格式的部位键名 ──────────────────────────────────────────
FEATURE_KEYS = ["eye", "eye2", "nose", "mouth", "head",
                "arm", "arm2", "hand", "hand2", "leg", "leg2"]
FACE_KEYS = FEATURE_KEYS[:5]    # eye, eye2, nose, mouth, head
LIMB_KEYS = FEATURE_KEYS[5:]    # arm, arm2, hand, hand2, leg, leg2

FEATURES_DIR.mkdir(exist_ok=True)

# ── 延迟导入重型库（避免 import 时卡死 GUI）───────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import customtkinter as ctk
from ui.theme import Colors, FONTS, apply_theme

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

# ── 识别历史记录数据库（功能4）────────────────────────────────────────────
HISTORY_DB_PATH = BASE_DIR / "history.db"

def init_history_db():
    """初始化历史记录数据库"""
    import sqlite3
    conn = sqlite3.connect(str(HISTORY_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            source_type TEXT,
            roles TEXT,
            result_json TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_history(filename: str, source_type: str, roles: list, result_json: str = ""):
    """保存识别结果到历史记录"""
    import sqlite3, json
    try:
        conn = sqlite3.connect(str(HISTORY_DB_PATH))
        conn.execute(
            "INSERT INTO history (timestamp, filename, source_type, roles, result_json) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), filename, source_type, json.dumps(roles, ensure_ascii=False), result_json)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

def load_history(limit: int = 100, search: str = "") -> list:
    """加载历史记录"""
    import sqlite3, json
    try:
        conn = sqlite3.connect(str(HISTORY_DB_PATH))
        if search:
            rows = conn.execute(
                "SELECT id, timestamp, filename, source_type, roles, result_json FROM history WHERE roles LIKE ? ORDER BY id DESC LIMIT ?",
                (f"%{search}%", limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, timestamp, filename, source_type, roles, result_json FROM history ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        conn.close()
        return [{"id": r[0], "timestamp": r[1], "filename": r[2], "source_type": r[3],
                  "roles": json.loads(r[4]) if r[4] else [], "result_json": r[5]} for r in rows]
    except Exception:
        return []

def get_history_stats() -> dict:
    """获取历史统计信息"""
    import sqlite3
    try:
        conn = sqlite3.connect(str(HISTORY_DB_PATH))
        total = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        conn.close()
        return {"total": total}
    except Exception:
        return {"total": 0}

init_history_db()

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


def save_database_to_moe(database: dict, db_name: str, negative_db: dict = None):
    """
    保存特征库为自研 .moe 文本格式
    格式：("角色名"{key1:val1:key2:val2:...keyN:valN:}"角色名2"{...})
    其中 val 为逗号分隔的浮点数（特征向量）
    若 negative_db 不为空，额外写入 "负面_<类别>" 条目（仅包含 head 键）
    """
    if not database and not negative_db:
        return None
    import numpy as np
    safe = _safe_filename(db_name)
    moe_path = FEATURES_DIR / f"{safe}.moe"
    moe_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    # 写入正面角色特征
    if database:
        for name, parts in database.items():
            content_parts = []
            for key in FEATURE_KEYS:
                if key in parts and isinstance(parts[key], np.ndarray):
                    vec_str = ",".join(f"{v:.10f}" for v in parts[key])
                    content_parts.append(f"{key}:{vec_str}")
            content = ":".join(content_parts)
            chunks.append(f'"{name}"{{{content}:}}')

    # 写入负面特征（带 "负面_" 前缀，仅使用 head 键）
    if negative_db:
        NEG_PREFIX = "负面_"
        for cat, emb in negative_db.items():
            if isinstance(emb, np.ndarray):
                vec_str = ",".join(f"{v:.10f}" for v in emb)
                chunks.append(f'"{NEG_PREFIX}{cat}"{{head:{vec_str}:}}')

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


# ── 负面特征库（非人脸样板）───────────────────────────────────────────────
def _extract_full_image_embedding(img_bgr):
    """
    将整张 BGR 图像缩放至 160x160 后提取 FaceNet 512 维嵌入，
    不经过动漫人脸检测，适用于负面样本/非人脸样板。
    """
    import cv2, torch
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rsz = cv2.resize(img_rgb, (160, 160))
    tensor = (torch.tensor(img_rsz).permute(2, 0, 1)
              .float().unsqueeze(0).to(_device) / 255.0)
    with torch.no_grad():
        return _resnet(tensor).cpu().numpy().flatten()


def build_negative_database(log_fn=print, stop_event=None):
    """
    从 data/负面特征/ 构建负面样本特征库。
    每个子文件夹为一个类别（真人、车、衣服…），
    图片不经过动漫人脸检测，直接整图提取 FaceNet 嵌入后按类别聚合平均。
    返回 {类别名: 512-dim ndarray}。
    """
    import cv2, numpy as np
    if not _ensure_models(log_fn):
        log_fn("❌ 模型未就绪，无法构建负面特征库")
        return None

    if not NEGATIVE_DIR.exists():
        log_fn("⚠️ 负面特征目录不存在喵~ 暂时跳过")
        return {}

    database = {}
    total = 0
    categories = sorted(d for d in NEGATIVE_DIR.iterdir() if d.is_dir() and not d.name.startswith("_"))

    if not categories:
        log_fn("⚠️ 负面特征目录下没有类别子文件夹喵~")
        return {}

    log_fn(f"📁 扫描到 {len(categories)} 个负面类别: {', '.join(d.name for d in categories)}")

    for subdir in categories:
        if stop_event and stop_event.is_set():
            break

        category = subdir.name
        imgs = sorted(
            p for p in subdir.iterdir()
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        )
        if not imgs:
            log_fn(f"  ⚠️ [{category}] 空文件夹，跳过")
            continue

        embeddings = []
        count = len(imgs)
        log_fn(f"  🔄 [{category}] 处理 {count} 张图...")

        for idx, img_path in enumerate(imgs):
            if stop_event and stop_event.is_set():
                break
            try:
                with open(img_path, "rb") as f:
                    data = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is None:
                    log_fn(f"    ⚠️ [{category}] 无法解码: {img_path.name}")
                    continue
                emb = _extract_full_image_embedding(img)
                embeddings.append(emb)
                total += 1
            except Exception as e:
                log_fn(f"    ⚠️ [{category}] 读取失败 {img_path.name}: {e}")

        if embeddings:
            database[category] = np.mean(embeddings, axis=0).astype(np.float32)
            log_fn(f"  ✅ [{category}] {len(embeddings)}/{count} 张有效，特征聚合完成")
        else:
            log_fn(f"  ⚠️ [{category}] 所有图片无效，跳过")

    log_fn(f"✅ 负面特征库构建完成：{len(database)} 个类别，共 {total} 张有效图")
    return database


def save_negative_database(database: dict, db_name: str):
    """
    将负面特征库保存到指定 .moe 文件中（追加写入）。
    以 "负面_" 前缀区分，负面条目仅含 head 键。
    """
    # 加载已有的正面数据库，加上负面特征一起保存
    existing = load_database_from_moe(db_name) or {}
    # 从 existing 中过滤掉已有的"负面_"条目
    positive_db = {k: v for k, v in existing.items() if not k.startswith("负面_")}
    save_database_to_moe(positive_db, db_name, negative_db=database)
    return True


def load_negative_database(db_name: str, log_fn=print):
    """
    从 .moe 文件中加载负面特征库（提取 "负面_" 前缀的条目）。
    返回 {类别名: 512-dim ndarray} 或 None。
    """
    combined = load_database_from_moe(db_name)
    if combined is None:
        return None
    negative = {}
    for name, parts in combined.items():
        if name.startswith("负面_") and "head" in parts:
            cat = name[3:]  # 去掉 "负面_" 前缀
            negative[cat] = parts["head"]
    return negative or None


def get_or_build_negative_database(db_name="全部特征库", force_rebuild=False, log_fn=print, stop_event=None):
    """
    获取或构建负面特征库。
    优先从 .moe 文件加载缓存；若强制重建则从 data/负面特征/ 重新提取。
    返回 {类别名: 512-dim ndarray}。
    """
    if not NEGATIVE_DIR.exists():
        log_fn("⚠️ 负面特征目录 data/负面特征/ 不存在喵~")
        return {}

    if not force_rebuild:
        cached = load_negative_database(db_name, log_fn)
        if cached is not None:
            return cached

    # 子文件夹检查
    has_subdirs = any(
        d.is_dir() and not d.name.startswith("_")
        for d in NEGATIVE_DIR.iterdir()
    )
    if not has_subdirs:
        log_fn("⚠️ data/负面特征/ 下没有类别文件夹喵~")
        return {}

    database = build_negative_database(log_fn, stop_event)
    if database:
        save_negative_database(database, db_name)
        log_fn(f"✅ 负面特征库已缓存到 {db_name}.moe 喵~")
    return database or {}


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
    从全身图像 + 128 关键点提取肢体特征（密集采样）
    返回 dict: {key: 512-dim embedding}
    """
    import cv2
    import numpy as np

    # 每个肢体部位对应 _BONE_SEGMENTS 的索引 → 128 关键点范围
    # arm=左大臂(seg7), arm2=右大臂(seg9), hand=左前臂(seg8), hand2=右前臂(seg10),
    # leg=左大腿(seg14), leg2=右大腿(seg16)
    LIMB_SEG_IDX = {
        "arm":  7,    # left_shoulder → left_elbow
        "arm2": 9,    # right_shoulder → right_elbow
        "hand": 8,    # left_elbow → left_wrist
        "hand2":10,   # right_elbow → right_wrist
        "leg":  14,   # left_hip → left_knee
        "leg2": 16,   # right_hip → right_knee
    }

    if not body_persons:
        return {}

    h, w = full_img_bgr.shape[:2]
    results = {}
    _, _, _, _, kps = body_persons[0]

    for key, seg_idx in LIMB_SEG_IDX.items():
        start, end = _bone_range(seg_idx)
        points = []
        for idx in range(start, end):
            if idx < len(kps) and kps[idx][2] > 0.5:
                px = int(kps[idx][0] * w)
                py = int(kps[idx][1] * h)
                points.append((px, py))

        # 也包含两端锚点
        from_idx, to_idx, _ = _BONE_SEGMENTS[seg_idx]
        for anc in (from_idx, to_idx):
            if anc < len(kps) and kps[anc][2] > 0.5:
                px = int(kps[anc][0] * w)
                py = int(kps[anc][1] * h)
                points.append((px, py))

        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # 密集点 → 更精确的包围盒
        x1 = max(0, min(xs) - 10)
        y1 = max(0, min(ys) - 10)
        x2 = min(w, max(xs) + 10)
        y2 = min(h, max(ys) + 10)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
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
            # 过滤掉"负面_"前缀的条目（它们由 get_or_build_negative_database 负责）
            positive_db = {k: v for k, v in db.items() if not k.startswith("负面_")}
            log_fn(f"✅ 从 .moe 缓存加载特征库 [{db_name}]，共 {len(positive_db)} 个角色")
            return positive_db, False
        # 兼容旧版：尝试加载 .json
        db = load_database_from_json(db_name)
        if db is not None:
            log_fn(f"✅ 从 .json 缓存加载特征库 [{db_name}]，共 {len(db)} 个角色（建议转为 .moe 格式）")
            return db, False

    if stop_event and stop_event.is_set():
        log_fn("⏹ 建库已停止")
        return {}, True

    # ⚡ 修复: 强制重建但 data/ 目录不存在时，回退到 .moe 缓存
    if force_rebuild and not DATA_DIR.exists():
        log_fn("⚠️ data/ 目录不存在，无法从训练图重建特征库喵~")
        log_fn("⏪ 回退到 .moe 缓存...")
        db = load_database_from_moe(db_name)
        if db is not None:
            positive_db = {k: v for k, v in db.items() if not k.startswith("负面_")}
            log_fn(f"✅ 从 .moe 缓存加载特征库 [{db_name}]，共 {len(positive_db)} 个角色")
            return positive_db, False
        log_fn("❌ .moe 缓存也不存在，无法加载特征库")
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
        # ── 同步构建负面特征库，一并存入 .moe ────────────────────────
        neg_db = build_negative_database(log_fn, stop_event) if NEGATIVE_DIR.exists() else {}
        save_database_to_moe(db, db_name, negative_db=neg_db)
        total_entries = len(db) + len(neg_db)
        log_fn(f"✅ 特征库已缓存为 .moe 格式喵~，共 {len(db)} 个角色 + {len(neg_db)} 个负面类别")
    return db, True


# ── 识别辅助 ──────────────────────────────────────────────────────────────
def cosine_similarity(a, b):
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

def recognize_character(face_img, database: dict, threshold=0.45,
                        full_img=None, body_persons=None,
                        negative_db=None, negative_threshold=NEGATIVE_THRESHOLD):
    """
    多部位综合识别——同时对比面部 + 肢体特征，并支持负面特征过滤
    face_img:  裁剪的人脸 RGB 图像
    database:  特征库（{name: {key: embedding}}）
    full_img:  完整 BGR 图像（用于提取肢体特征）
    body_persons: 姿态检测结果（用于提取肢体特征）
    negative_db: 负面特征库（{类别名: 512-dim embedding}），用于过滤非人脸误检测
    negative_threshold: 负面匹配阈值，超过此值且高于角色匹配分则标记为"非人脸"
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

    # 4. 检查负面特征库（防止把非人脸误识别为角色）
    if negative_db and face_emb is not None:
        for neg_cat, neg_emb in negative_db.items():
            neg_score = cosine_similarity(face_emb, neg_emb)
            if neg_score > negative_threshold and neg_score > best_score:
                return (f"非人脸({neg_cat})", neg_score)

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


# ── 现代化识别框绘制 ────────────────────────────────────────────────────────
def _box_color_by_score(score: float):
    """根据相似度返回 BGR 颜色"""
    if score >= 0.80:
        return (110, 231, 183)   # 成功绿 #6EE7B7
    elif score >= 0.50:
        return (102, 209, 255)   # 赛博蓝 #62D8FF
    else:
        return (168, 168, 200)   # 灰紫 #A8A8C8


def draw_modern_recognition_box(img, x1, y1, x2, y2, name, score):
    """绘制现代化 VTuber 风格识别框 + 标签气泡"""
    import cv2
    color = _box_color_by_score(score)

    # 圆角矩形框（主框 + 发光外框效果）
    overlay = img.copy()
    alpha = 0.15
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # 顶部标签气泡（填充背景）
    label = f"{name}" if name else "Unknown"
    pct_text = f" {score:.0%}"
    full_text = label + pct_text

    # 用 PIL 绘制中文标签
    from PIL import Image, ImageDraw, ImageFont
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw_ctx = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(str(FONT_PATH), 16) if FONT_PATH.is_file() else ImageFont.load_default()

        # 计算文字尺寸
        bbox = draw_ctx.textbbox((0, 0), full_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 6
        ly = max(0, y1 - th - pad * 2)

        # 背景 + 文字
        draw_ctx.rectangle([x1, ly, x1 + tw + pad * 2, y1],
                           fill=(color[2], color[1], color[0]))
        draw_ctx.text((x1 + pad, ly + pad), full_text, font=font,
                      fill=(18, 19, 26))  # 深色文字

        # 相似度进度条（在标签下方）
        bar_h = 3
        bar_y = y1 - bar_h - 1
        bar_w = (x2 - x1) * min(score, 1.0)
        bar_color = (110, 231, 183) if score >= 0.80 else (
            (102, 209, 255) if score >= 0.50 else (168, 168, 200))
        draw_ctx.rectangle([x1, bar_y, x1 + int(bar_w), bar_y + bar_h],
                           fill=(bar_color[2], bar_color[1], bar_color[0]))

        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # 降级：OpenCV 纯英文绘制
        cv2.rectangle(img, (x1, y1 - 24), (x1 + 180, y1), color, -1)
        cv2.putText(img, full_text, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (18, 19, 26), 1, cv2.LINE_AA)

    return img


def _is_valid_face_box(w: int, h: int) -> bool:
    """过滤非人脸区域：宽高比 0.6~1.5，最小 30x30"""
    if w < 30 or h < 30:
        return False
    ratio = w / h if h > 0 else 0
    return 0.6 <= ratio <= 1.5


# ── 人体姿态检测（Google MediaPipe Pose Landmarker）────────────────────

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

# ── 128 关键点：17 个 COCO 基础点 + 111 个骨骼插值点 ─────────────────
# (from_coco, to_coco, num_interpolated)
_BONE_SEGMENTS = [
    (0, 1, 3), (0, 2, 3), (1, 3, 4), (2, 4, 4),   # 面部
    (5, 6, 4), (0, 5, 5), (0, 6, 4),               # 肩/颈
    (5, 7, 8), (7, 9, 10),                          # 左臂
    (6, 8, 8), (8, 10, 10),                         # 右臂
    (5, 11, 6), (6, 12, 6), (11, 12, 4),            # 躯干
    (11, 13, 8), (13, 15, 8),                       # 左腿
    (12, 14, 8), (14, 16, 8),                       # 右腿
]

def _build_128_connections():
    """生成 128 关键点骨骼连线（基础 17 点 + 每条骨骼的连续线段）"""
    conns = list(BODY_CONNECTIONS)
    idx = 17
    for a, b, n in _BONE_SEGMENTS:
        prev = a
        for _ in range(n):
            conns.append((prev, idx))
            prev = idx
            idx += 1
        conns.append((prev, b))
    return conns

BODY_CONNECTIONS_128 = _build_128_connections()

# ── 肢体部位 → 128 关键点索引范围（用于精确裁剪） ──
# 各个部位的插值点在 128 数组中的 (start, end) 区间
def _bone_range(seg_index):
    """计算 _BONE_SEGMENTS[seg_index] 的插值点在 128 中的起止索引"""
    offset = 17
    for i in range(seg_index):
        offset += _BONE_SEGMENTS[i][2]
    return (offset, offset + _BONE_SEGMENTS[seg_index][2])

# ── MediaPipe Pose Landmarker 全局单例 ──
_mediapipe_landmarker = None

def _get_pose_landmarker(log_fn=print):
    """
    获取 MediaPipe Pose Landmarker 单例（懒加载）
    优先使用 pose_landmarker.task（全量），回退到 pose_landmarker_lite.task（轻量）
    如果原文件无法被 MediaPipe 直接读取（已知 zip 兼容性问题），会自动重新打包
    """
    global _mediapipe_landmarker
    if _mediapipe_landmarker is not None:
        return _mediapipe_landmarker

    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        log_fn("❌ 未安装 mediapipe，请运行: pip install mediapipe")
        return None

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'models')

    # 选模型：全量 > 轻量
    model_path = os.path.join(model_dir, 'pose_landmarker.task')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'pose_landmarker_lite.task')
        if not os.path.exists(model_path):
            log_fn("❌ 未找到 pose_landmarker.task，请从 Google MediaPipe 下载放到 models/ 目录")
            return None

    def _try_load(path):
        """尝试用 MediaPipe 加载 task 文件"""
        base_options = python.BaseOptions(model_asset_path=path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
        )
        return vision.PoseLandmarker.create_from_options(options)

    try:
        _mediapipe_landmarker = _try_load(model_path)
        log_fn(f"✅ MediaPipe Pose Landmarker 已就绪喵~（{os.path.basename(model_path)}）")
        return _mediapipe_landmarker
    except Exception as e:
        err_msg = str(e)
        if 'Unable to open zip archive' in err_msg or 'zip' in err_msg.lower():
            # 已知兼容性问题：用 Python 重新打包后加载
            repacked = os.path.join(model_dir, 'mp_pose_repacked.task')
            try:
                import zipfile
                log_fn("🔧 检测到 zip 兼容性问题，正在重新打包模型...")
                with zipfile.ZipFile(model_path, 'r') as zf:
                    entries = [(n, zf.read(n)) for n in zf.namelist()]
                with zipfile.ZipFile(repacked, 'w', zipfile.ZIP_STORED) as zf:
                    for name, data in entries:
                        zf.writestr(name, data)
                _mediapipe_landmarker = _try_load(repacked)
                log_fn(f"✅ MediaPipe Pose Landmarker 已就绪喵~（{os.path.basename(model_path)} → 重打包）")
                return _mediapipe_landmarker
            except Exception as e2:
                log_fn(f"❌ 重打包也失败了: {e2}")
                return None
        else:
            log_fn(f"❌ MediaPipe Pose Landmarker 加载失败: {e}")
            return None

def detect_body_pose(image_bgr, log_fn=print):
    """
    使用 MediaPipe Pose Landmarker 检测人体姿态
    返回列表，每个元素为:
        (x1, y1, x2, y2, keypoints)
    其中 keypoints 是 128 关键点数组：[x, y, conf] 归一化坐标
    前 17 个 = COCO 标准点，后 111 个 = 沿骨骼插值的密集点
    """
    import cv2

    landmarker = _get_pose_landmarker(log_fn)
    if landmarker is None:
        return []

    h, w = image_bgr.shape[:2]

    # MediaPipe 33 → COCO 17 基础关键点
    MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    from mediapipe import Image as MpImage, ImageFormat
    mp_image = MpImage(image_format=ImageFormat.SRGB, data=img_rgb)

    try:
        detection_result = landmarker.detect(mp_image)
    except Exception as e:
        log_fn(f"⚠️ MediaPipe 推理失败: {e}")
        return []

    if not detection_result.pose_landmarks:
        return []

    persons = []
    for pose_landmarks in detection_result.pose_landmarks:
        # ── 1. 提取 COCO 17 基础点 ──
        kps_base = []
        for mp_idx in MP_TO_COCO:
            lm = pose_landmarks[mp_idx]
            kps_base.append([float(lm.x), float(lm.y), float(lm.visibility)])

        # ── 2. 沿骨骼插值 111 个密集点 → 128 总计 ──
        kps_128 = kps_base.copy()
        for a, b, n in _BONE_SEGMENTS:
            pa = kps_base[a]
            pb = kps_base[b]
            for t_idx in range(1, n + 1):
                t = t_idx / (n + 1)
                kps_128.append([
                    pa[0] + (pb[0] - pa[0]) * t,
                    pa[1] + (pb[1] - pa[1]) * t,
                    min(pa[2], pb[2]),      # 两端都可见才可信
                ])

        # ── 3. 边界框（从 128 点计算） ──
        xs = [k[0] for k in kps_128 if k[2] > 0.5]
        ys = [k[1] for k in kps_128 if k[2] > 0.5]
        if not xs:
            xs = [k[0] for k in kps_base]
            ys = [k[1] for k in kps_base]

        pad_ratio = 0.05
        x1 = int(max(0, (min(xs) - pad_ratio) * w))
        y1 = int(max(0, (min(ys) - pad_ratio) * h))
        x2 = int(min(w, (max(xs) + pad_ratio) * w))
        y2 = int(min(h, (max(ys) + pad_ratio) * h))

        persons.append((x1, y1, x2, y2, kps_128))

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
    在图像上绘制人体矩形框 + 128 关键点骨骼连线 + COCO 17 锚点标签
    persons: list of (x1, y1, x2, y2, keypoints_128)
    """
    import cv2
    h, w = image_bgr.shape[:2]

    if not persons:
        return image_bgr

    for (x1, y1, x2, y2, landmarks) in persons:
        # 1. 人体边界框
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), bbox_color, bbox_thickness)

        # 2. 骨骼连线（128 点版本，含插值段）
        for (idx1, idx2) in BODY_CONNECTIONS_128:
            if idx1 >= len(landmarks) or idx2 >= len(landmarks):
                continue
            kp1, kp2 = landmarks[idx1], landmarks[idx2]
            if kp1[2] > 0.5 and kp2[2] > 0.5:
                pt1 = (int(kp1[0] * w), int(kp1[1] * h))
                pt2 = (int(kp2[0] * w), int(kp2[1] * h))
                cv2.line(image_bgr, pt1, pt2, line_color, line_thickness)

        # 3. 插值点圆点（小点，不标注标签）
        if len(landmarks) > 17:
            for idx in range(17, len(landmarks)):
                kp = landmarks[idx]
                if kp[2] > 0.5:
                    x, y = int(kp[0] * w), int(kp[1] * h)
                    cv2.circle(image_bgr, (x, y), 2, point_color, -1)

        # 4. COCO 17 锚点（大点 + 标签）
        for idx, label in BODY_KEYPOINT_LABELS.items():
            if idx >= len(landmarks):
                continue
            kp = landmarks[idx]
            if kp[2] > 0.5:
                x, y = int(kp[0] * w), int(kp[1] * h)
                cv2.circle(image_bgr, (x, y), point_radius, point_color, -1)
                cv2.circle(image_bgr, (x, y), point_radius + 1, line_color, 1)
                if show_labels:
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
        _safe_move(tmp_video, out_video)
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
        # 确保 moviepy/ffmpeg 完全退出再移动文件（避免 [WinError 32] 文件占用）
        _safe_move(tmp_video, out_video)


def _safe_move(src, dst, retries=5, delay=0.5):
    """带重试的安全文件移动（处理 ffmpeg 进程残留锁）"""
    import time
    for i in range(retries):
        try:
            if os.path.exists(dst):
                os.unlink(dst)
            shutil.move(src, dst)
            return
        except (PermissionError, OSError):
            if i < retries - 1:
                time.sleep(delay * (i + 1))  # 递增等待
            else:
                # 最后尝试：复制 + 删除原文件
                try:
                    shutil.copy2(src, dst)
                    try:
                        os.unlink(src)
                    except OSError:
                        pass
                except Exception:
                    warnings.warn(f"无法移动文件: {src} → {dst}")

def process_image_file(source: str, database: dict, threshold=0.45,
                       min_neighbors=5, body_mode=False,
                       log_fn=print, preview_fn=None, done_fn=None,
                       stop_event=None,
                       # ── AI 模块参数 ──────────────────────────────────
                       enable_emotion=False, enable_nsfw=False,
                       ai_results=None,
                       # ── 负面特征库 ────────────────────────────────────
                       negative_db=None, negative_threshold=NEGATIVE_THRESHOLD):
    import cv2
    import numpy as np
    from modules.base import AIResult
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

        # ── AI 增强：情绪识别 ──────────────────────────────────────────
        if enable_emotion and _EMOTION_MODULE is not None:
            log_fn("😊 情绪识别中...")
            try:
                emotions = _EMOTION_MODULE.process_frame(img, timestamp=0.0)
                if emotions:
                    for emo in emotions:
                        ai_results.add(AIResult(
                            module="emotion", event_type="face_emotion",
                            timestamp=0.0, data=emo, confidence=emo["confidence"],
                        ))
                    log_fn(f"  ✅ 检测到 {len(emotions)} 张人脸情绪")
                    # 在图片上标注情绪
                    for emo in emotions:
                        bx, by, bw, bh = emo["bbox"]
                        label = emo["emotion_label"]
                        conf = emo["confidence"]
                        img = draw_chinese_text(
                            img, f"[{label}] {conf:.0%}",
                            (bx, by + bh + 5), 14, (255, 200, 0)
                        )
            except Exception as e:
                log_fn(f"  ⚠️ 情绪识别异常: {e}")

        # ── AI 增强：NSFW 检测 ─────────────────────────────────────────
        if enable_nsfw and _NSFW_MODULE is not None:
            log_fn("🔞 NSFW 检测中...")
            try:
                nsfw_result = _NSFW_MODULE.process_frame(img, timestamp=0.0)
                if nsfw_result:
                    ai_results.add(nsfw_result)
                    nsfw_label = nsfw_result.data.get("label_cn", "未知")
                    nsfw_score = nsfw_result.data.get("nsfw_score", 0.0)
                    log_fn(f"  {nsfw_label}: {nsfw_score:.1%}")
                    img = draw_chinese_text(
                        img, f"NSFW: {nsfw_label} ({nsfw_score:.1%})",
                        (10, 30), 16, (255, 100, 100)
                    )
            except Exception as e:
                log_fn(f"  ⚠️ NSFW 检测异常: {e}")

        # 人脸识别（始终执行，无论 body_mode 是否开启，实现"同时检测"）
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
            if not _is_valid_face_box(w, h):
                continue
            x1 = max(0, x); y1 = max(0, y)
            x2 = min(img.shape[1], x + w)
            y2 = min(img.shape[0], y + h)
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            name, score = recognize_character(face_rgb, database, threshold,
                                              full_img=img, body_persons=persons,
                                              negative_db=negative_db, negative_threshold=negative_threshold)
            if not name or score < threshold:
                continue
            img = draw_modern_recognition_box(img, x1, y1, x2, y2, name, score)
            log_fn(f"  {'✅' if name else '❓'} {name} ({score:.2f})")

        if preview_fn:
            preview_fn(img)
    except Exception:
        log_fn(f"❌ 呜呜呜,(｡•́︿•̀｡)处理图片出错了:\n{traceback.format_exc()}")
    finally:
        if done_fn: done_fn()


def process_video_file(source: str, database: dict, output_path=None,
                       threshold=0.45, skip_frames=2, min_neighbors=5,
                       body_mode=False,
                       log_fn=print, preview_fn=None,
                       stop_event: threading.Event = None, done_fn=None,
                       # ── AI 模块参数 ──────────────────────────────────
                       enable_emotion=False, enable_speech=False,
                       enable_nsfw=False, ai_results=None,
                       # ── 负面特征库 ────────────────────────────────────
                       negative_db=None, negative_threshold=NEGATIVE_THRESHOLD):
    import cv2
    from modules.base import AIResult
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log_fn(f"❌ 无法打开视频喵~: {source}")
        if done_fn: done_fn()
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 限制输出分辨率，防止 x264 编码器内存分配失败
    MAX_DIM = 1920
    out_w, out_h = w, h
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        out_w, out_h = int(w * scale), int(h * scale)
        # 确保偶数像素（避免某些编码器报错）
        out_w -= out_w % 2
        out_h -= out_h % 2
        log_fn(f"⚠️ 原视频分辨率过大 ({w}×{h})，已限制输出为 ({out_w}×{out_h})")

    log_fn(f"视频信息: {w}×{h}  {fps:.1f}fps  共 {total} 帧")
    if body_mode:
        log_fn("🔍 同时开启: 人体姿态检测(框+骨骼) + 人脸角色识别")

    out = tmp_out = None
    if output_path:
        tmp_dir = Path(output_path).parent / "temp"
        tmp_dir.mkdir(exist_ok=True)
        tmp_fd, tmp_out = tempfile.mkstemp(suffix=".mp4", dir=str(tmp_dir))
        os.close(tmp_fd)
        out = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    frame_idx  = 0
    found_names: set = set()
    persons = []  # 用于跨循环传递 body_persons
    need_resize = (out_w != w or out_h != h)
    # ── 画面角色时间线（用于语音转文字角色关联）──────────────────────
    _face_timeline: list = []

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
                    if not _is_valid_face_box(fw, fh):
                        continue
                    x1 = max(0, x); y1 = max(0, y)
                    x2 = min(frame.shape[1], x + fw)
                    y2 = min(frame.shape[0], y + fh)
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    name, score = recognize_character(face_rgb, database, threshold,
                                                      full_img=frame, body_persons=persons,
                                                      negative_db=negative_db, negative_threshold=negative_threshold)
                    if not name or score < threshold:
                        continue
                    frame = draw_modern_recognition_box(frame, x1, y1, x2, y2, name, score)
                    if name:
                        found_names.add(name)
                        # 收集画面角色时间线（供语音角色关联使用）
                        if enable_speech:
                            _face_timeline.append({
                                "timestamp": frame_idx / fps if fps > 0 else 0.0,
                                "names": [name],
                            })

                # ── AI 增强：帧级情绪识别 ──────────────────────────────
                if enable_emotion and _EMOTION_MODULE is not None:
                    try:
                        timestamp = frame_idx / fps
                        emotions = _EMOTION_MODULE.process_frame(
                            frame, timestamp,
                            face_boxes=faces if len(faces) > 0 else None,
                        )
                        for emo in emotions:
                            if ai_results is not None:
                                ai_results.add(AIResult(
                                    module="emotion", event_type="face_emotion",
                                    timestamp=timestamp, data=emo,
                                    confidence=emo["confidence"],
                                ))
                            # 在画面上标注情绪
                            bx, by, bw, bh = emo["bbox"]
                            img_label = emo["emotion_label"]
                            frame = draw_chinese_text(
                                frame, f"{img_label}",
                                (bx, by + bh + 2), 12, (255, 200, 0)
                            )
                    except Exception:
                        pass

                # ── AI 增强：帧级 NSFW 检测 ─────────────────────────────
                if enable_nsfw and _NSFW_MODULE is not None:
                    try:
                        timestamp = frame_idx / fps
                        nsfw_result = _NSFW_MODULE.process_frame(frame, timestamp)
                        if nsfw_result and ai_results is not None:
                            ai_results.add(nsfw_result)
                    except Exception:
                        pass

            if preview_fn and frame_idx % skip_frames == 0:
                preview_fn(frame.copy())

            if out:
                if need_resize:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
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

    # ── AI 增强：语音转文字 + 说话人分离 ────────────────────────────
    if enable_speech and _SPEECH_MODULE is not None and ai_results is not None:
        log_fn("🎤 语音转文字 + 说话人分离中...")
        log_fn("ℹ️ 首次运行可能需要下载模型，请耐心等待")
        try:
            # 将已知角色名作为说话人候选
            known_speakers = None
            if database:
                known_speakers = {}
                for i, name in enumerate(database.keys()):
                    known_speakers[f"speaker_{i}"] = name

            # 视频路径传给语音模块，它会自动提取音频
            speech_results = _SPEECH_MODULE.process_video(
                video_path=source,
                language="auto",
                enable_diarization=True,
                known_speakers=known_speakers,
                log_fn=log_fn,
                stop_event=stop_event,
                face_timeline=_face_timeline,
            )
            # 合并结果
            ai_results.extend(speech_results.results)

            # 打印摘要
            utterances = [r for r in speech_results.results
                          if r.get("event_type") == "utterance"]
            log_fn(f"  ✅ 共转录 {len(utterances)} 句话")
            for utt in utterances[:10]:  # 只打印前 10 句
                d = utt.get("data", {})
                log_fn(f"  [{d.get('speaker_name','?')}] {d.get('text','')[:60]}")
            if len(utterances) > 10:
                log_fn(f"  ...还有 {len(utterances) - 10} 句（详见 JSON 结果）")

            # 语义 NSFW：分析转录文本
            if enable_nsfw and _NSFW_MODULE is not None:
                log_fn("🔞 语义 NSFW 检测（分析语音内容）...")
                for utt in utterances[:]:  # 使用切片避免重复
                    d = utt.get("data", {})
                    text = d.get("text", "")
                    if text.strip():
                        nsfw_text_result = _NSFW_MODULE.process_text(
                            text=text,
                            timestamp=d.get("start", 0.0),
                            source="transcript",
                        )
                        if nsfw_text_result and nsfw_text_result.confidence > 0:
                            ai_results.add(nsfw_text_result)
                            log_fn(f"  ⚠️ NSFW 语义检测: {nsfw_text_result.data.get('nsfw_score', 0):.1%}")
        except Exception as e:
            log_fn(f"  ⚠️ 语音识别异常: {e}")
            import traceback
            traceback.print_exc()

    # ── AI 结果保存 ─────────────────────────────────────────────────
    if ai_results is not None and len(ai_results) > 0:
        try:
            out_base = Path(source).stem
            save_analysis_result(ai_results, str(Path(source).parent / out_base), log_fn)
        except Exception as e:
            log_fn(f"  ⚠️ 保存 AI 结果失败: {e}")

    if done_fn:
        done_fn()


# ══════════════════════════════════════════════════════════════════════════════
#  AI 结果输出工具
# ══════════════════════════════════════════════════════════════════════════════

def save_analysis_result(collection, output_base: str, log_fn=print):
    """
    保存 AI 模块分析结果为 JSON 文件。
    若已启用多个模块，结果会合并到一个 JSON 中。
    """
    if not collection or len(collection) == 0:
        log_fn("ℹ️ 无分析结果可保存")
        return None

    output_json = f"{output_base}.ai_result.json"
    try:
        with open(output_json, "w", encoding="utf-8") as f:
            f.write(collection.to_json())
        log_fn(f"💾 分析结果已保存: {output_json}")
        return output_json
    except Exception as e:
        log_fn(f"⚠️ 保存分析结果失败: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Tkinter GUI
# ══════════════════════════════════════════════════════════════════════════════
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}


class MoeFaceApp:
    WIN_W, WIN_H = 1100, 780

    def __init__(self):
        apply_theme()
        ctk.set_appearance_mode("dark")

        if DND_AVAILABLE:
            self.root = tkdnd.Tk()
        else:
            self.root = tk.Tk()

        self.root.title("MoeFace — AI Anime Face Recognition")
        self.root.geometry(f"{self.WIN_W}x{self.WIN_H}")
        self.root.minsize(900, 650)
        self.root.configure(bg=Colors.BG)

        # 状态
        self._database: dict   = {}
        self._db_name:  str    = ""
        self._negative_db: dict = {}
        self._stop_evt: threading.Event = threading.Event()
        self._busy      = False
        self._preview_img = None
        self._models_ready = False
        # ── 新功能状态（2026-07-16 新增）───────────────────────────────────
        self._last_results: list = []           # 最近一次识别结果（功能1）
        self._queue: list = []                  # 批量处理队列（功能2）
        self._queue_idx = 0
        self._api_running = False               # API 服务状态（功能6）
        self._api_server_thread = None
        self._multi_role_mode = False           # 多角色模式（功能8）
        self._heatmap_mode = False              # 热力图模式（功能14）
        self._update_available = False          # 更新可用（功能9）
        self._new_version = ""
        self._performance_mode = "standard"     # 性能模式（功能13）
        self._appearance_stats: dict = {}       # 出场统计（功能11）
        self._live_overlay = None               # 直播输出（功能12）
        # ── 模型版本管理（功能16）───────────────────────────────────────────
        self._model_config = {}
        self._current_model = "facenet-vggface2"
        self._current_model_path = FEATURES_DIR

        self._auto_shutdown_var = tk.BooleanVar(value=False)
        self._body_mode_var = tk.BooleanVar(value=False)
        self._emotion_mode_var = tk.BooleanVar(value=False)
        self._speech_mode_var = tk.BooleanVar(value=False)
        self._nsfw_mode_var = tk.BooleanVar(value=False)
        # ── 新功能开关（2026-07-16 新增）───────────────────────────────────
        self._multi_role_var = tk.BooleanVar(value=False)
        self._heatmap_var = tk.BooleanVar(value=False)
        self._performance_var = tk.StringVar(value="standard")

        self._build_ui()
        self._load_models_async()
        # ── 启动时后台检查（功能3/9）─────────────────────────────────────
        self.root.after(2000, lambda: threading.Thread(target=self._init_tray, daemon=True).start())
        self.root.after(3000, self._check_update_auto)

    # ── UI 构建 ────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root

        # Header
        hdr = ctk.CTkFrame(root, height=52, fg_color=Colors.CARD, corner_radius=0)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        logo_frame = ctk.CTkFrame(hdr, fg_color="transparent")
        logo_frame.pack(side="left", padx=16, pady=4)
        ctk.CTkLabel(logo_frame, text="◉", font=("Segoe UI", 20),
                     text_color=Colors.PRIMARY).pack(side="left", padx=(0, 4))
        ctk.CTkLabel(logo_frame, text="MoeFace",
                     font=FONTS["title"], text_color=Colors.TEXT).pack(side="left")
        ctk.CTkLabel(logo_frame, text="  AI Anime Vision",
                     font=FONTS["small"], text_color=Colors.TEXT_MUTED).pack(side="left", padx=(4, 0))

        self._status_lbl = ctk.CTkLabel(hdr, text=" 正在初始化…",
                                        font=FONTS["body"], text_color=Colors.ACCENT)
        self._status_lbl.pack(side="right", padx=16)

        # Body
        body = ctk.CTkFrame(root, fg_color=Colors.BG, corner_radius=0)
        body.pack(fill="both", expand=True)

        left = ctk.CTkScrollableFrame(body, width=250, fg_color=Colors.BG, corner_radius=0)
        left.pack(side="left", fill="y", padx=(0, 0), pady=0)

        right = ctk.CTkFrame(body, fg_color=Colors.BG, corner_radius=0)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self._build_left(left)
        self._build_right(right)

    def _build_card_header(self, parent, title):
        """卡片区域标题行"""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=(12, 4))
        ctk.CTkLabel(row, text=title, font=FONTS["card_title"],
                     text_color=Colors.PRIMARY).pack(side="left")

    def _btn(self, parent, text, cmd, fg_color=Colors.CARD,
             hover_color=None, border_width=1, border_color=None, **kw):
        return ctk.CTkButton(
            parent, text=text, command=cmd,
            fg_color=fg_color,
            hover_color=hover_color or Colors.CARD_HOVER,
            border_width=border_width,
            border_color=border_color or Colors.BORDER,
            font=FONTS["body"], corner_radius=6, **kw
        )

    def _build_left(self, parent):
        pad = {"padx": 4, "pady": 2, "anchor": "w"}

        # ── 🎯 识别源 ──
        self._build_card_header(parent, "🎯 识别源")
        self._open_btn = ctk.CTkButton(
            parent, text=" 打开图片 / 视频",
            command=self._open_file,
            fg_color=Colors.PRIMARY, hover_color=Colors.PRIMARY_HOVER,
            font=FONTS["body"], corner_radius=6
        )
        self._open_btn.pack(fill="x", padx=4, pady=2)

        # ── 🧠 AI 增强 ──
        self._build_card_header(parent, "🧠 AI 增强")
        ai_frame = ctk.CTkFrame(parent, fg_color=Colors.CARD, corner_radius=6)
        ai_frame.pack(fill="x", padx=4, pady=4)

        self._emotion_cb = ctk.CTkCheckBox(
            ai_frame, text="😊 情绪识别", variable=self._emotion_mode_var,
            font=FONTS["body"], text_color=Colors.TEXT_MUTED, fg_color=Colors.PRIMARY)
        self._emotion_cb.pack(anchor="w", padx=8, pady=(8, 2))

        self._speech_cb = ctk.CTkCheckBox(
            ai_frame, text="🎤 语音转文字", variable=self._speech_mode_var,
            font=FONTS["body"], text_color=Colors.TEXT_MUTED, fg_color=Colors.PRIMARY)
        self._speech_cb.pack(anchor="w", padx=8, pady=2)

        self._nsfw_cb = ctk.CTkCheckBox(
            ai_frame, text="🔞 NSFW 检测", variable=self._nsfw_mode_var,
            font=FONTS["body"], text_color=Colors.TEXT_MUTED, fg_color=Colors.PRIMARY)
        self._nsfw_cb.pack(anchor="w", padx=8, pady=(2, 8))

        # ── ⚙ 识别设置 ──
        self._build_card_header(parent, "⚙ 识别设置")

        # 特征库选择
        self._db_var = tk.StringVar(value=DEFAULT_DB_NAME)
        db_names = [DEFAULT_DB_NAME] + sorted(scan_role_folders())
        self._db_combo = ctk.CTkOptionMenu(
            parent, values=db_names, variable=self._db_var,
            font=FONTS["body"], fg_color=Colors.BORDER,
            button_color=Colors.PRIMARY, button_hover_color=Colors.PRIMARY_HOVER,
            corner_radius=6)
        self._db_combo.pack(fill="x", padx=4, pady=2)

        # 阈值
        self._threshold_var = tk.DoubleVar(value=0.45)
        self._build_slider(parent, "阈值", self._threshold_var, 0.1, 0.95, 0.01)

        # 灵敏度 (默认 5，减少误检)
        self._min_neighbors_var = tk.IntVar(value=5)
        self._build_slider(parent, "灵敏度", self._min_neighbors_var, 1, 10, 1)

        # 跳帧
        self._skip_var = tk.IntVar(value=2)
        self._build_slider(parent, "跳帧", self._skip_var, 1, 30, 1)

        # 人体姿态
        self._body_mode_cb = ctk.CTkCheckBox(
            parent, text="🧍 人体姿态检测", variable=self._body_mode_var,
            font=FONTS["body"], text_color=Colors.TEXT_MUTED, fg_color=Colors.PRIMARY)
        self._body_mode_cb.pack(anchor="w", padx=4, pady=2)

        # ── 新功能：多角色模式 + 热力图（功能8/14）─────────────────────────
        self._multi_role_cb = ctk.CTkCheckBox(
            parent, text="👥 多角色模式", variable=self._multi_role_var,
            font=FONTS["body"], text_color=Colors.TEXT_MUTED, fg_color=Colors.PRIMARY)
        self._multi_role_cb.pack(anchor="w", padx=4, pady=2)

        self._heatmap_cb = ctk.CTkCheckBox(
            parent, text="🔍 显示热力图", variable=self._heatmap_var,
            font=FONTS["body"], text_color=Colors.TEXT_MUTED, fg_color=Colors.PRIMARY)
        self._heatmap_cb.pack(anchor="w", padx=4, pady=2)

        # ── 新功能：性能模式（功能13）──────────────────────────────────────
        perf_frame = ctk.CTkFrame(parent, fg_color="transparent")
        perf_frame.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(perf_frame, text="⚡ 性能", font=FONTS["small"],
                     text_color=Colors.TEXT_MUTED, width=36).pack(side="left")
        self._perf_combo = ctk.CTkOptionMenu(
            perf_frame, values=["标准", "快速", "极致"],
            variable=self._performance_var,
            font=FONTS["body"], fg_color=Colors.BORDER,
            button_color=Colors.PRIMARY, button_hover_color=Colors.PRIMARY_HOVER,
            corner_radius=6, width=100)
        self._perf_combo.pack(side="left", padx=6)

        # ── 💾 输出 ──
        self._build_card_header(parent, "💾 输出（可选）")
        out_frame = ctk.CTkFrame(parent, fg_color="transparent")
        out_frame.pack(fill="x", padx=4, pady=2)
        self._out_var = tk.StringVar()
        ctk.CTkEntry(out_frame, textvariable=self._out_var,
                     placeholder_text="保存路径...",
                     font=FONTS["body"], corner_radius=6).pack(
                     side="left", fill="x", expand=True)
        ctk.CTkButton(out_frame, text="⋯", command=self._pick_output,
                      width=30, corner_radius=6,
                      fg_color=Colors.BORDER, hover_color=Colors.CARD_HOVER,
                      font=FONTS["body"]).pack(side="left", padx=4)

        # ── 新功能1：导出报告按钮 ──
        export_frame = ctk.CTkFrame(parent, fg_color="transparent")
        export_frame.pack(fill="x", padx=4, pady=2)
        self._export_html_btn = self._btn(export_frame, "📄 HTML", self._export_report_html,
                                          fg_color=Colors.CARD, width=70)
        self._export_html_btn.pack(side="left", padx=(0, 2))
        self._export_json_btn = self._btn(export_frame, "📋 JSON", lambda: self._export_report("json"),
                                          fg_color=Colors.CARD, width=60)
        self._export_json_btn.pack(side="left", padx=2)
        self._export_csv_btn = self._btn(export_frame, "📊 CSV", lambda: self._export_report("csv"),
                                         fg_color=Colors.CARD, width=60)
        self._export_csv_btn.pack(side="left", padx=2)

        # ── ⚡ 操作 ──
        self._build_card_header(parent, "⚡ 操作")
        self._rebuild_btn = self._btn(parent, "重建特征库", self._rebuild_db,
                                      fg_color=Colors.CARD)
        self._rebuild_btn.pack(fill="x", padx=4, pady=2)

        self._load_btn = self._btn(parent, "加载特征库", self._load_db_now,
                                   fg_color=Colors.CARD)
        self._load_btn.pack(fill="x", padx=4, pady=2)

        self._stop_btn = ctk.CTkButton(
            parent, text="⏹ 停止",
            command=self._stop_processing,
            fg_color=Colors.ERROR, hover_color="#E55A78",
            font=FONTS["body"], corner_radius=6
        )
        self._stop_btn.pack(fill="x", padx=4, pady=2)

        # 进度条
        self._progress_bar = ctk.CTkProgressBar(parent, mode="determinate", width=220)
        self._progress_bar.pack(fill="x", padx=4, pady=(8, 0))
        self._progress_bar.set(0)
        self._progress_label = ctk.CTkLabel(parent, text="等待加载特征库...",
                                            font=FONTS["small"],
                                            text_color=Colors.TEXT_MUTED)
        self._progress_label.pack(anchor="w", padx=4, pady=2)

        # 杂项
        self._btn(parent, "管理角色别名", self._open_alias_editor,
                  fg_color=Colors.CARD).pack(fill="x", padx=4, pady=2)

        # ── 新功能2：批量处理 ──
        self._build_card_header(parent, "📦 批量处理")
        batch_frame = ctk.CTkFrame(parent, fg_color="transparent")
        batch_frame.pack(fill="x", padx=4, pady=2)
        self._batch_add_btn = self._btn(batch_frame, "添加文件", self._batch_add_files,
                                        fg_color=Colors.CARD, width=80)
        self._batch_add_btn.pack(side="left", padx=(0, 2))
        self._batch_folder_btn = self._btn(batch_frame, "添加文件夹", self._batch_add_folder,
                                           fg_color=Colors.CARD, width=80)
        self._batch_folder_btn.pack(side="left", padx=2)
        self._batch_clear_btn = self._btn(batch_frame, "清空", self._batch_clear,
                                          fg_color=Colors.CARD, width=50)
        self._batch_clear_btn.pack(side="left", padx=2)
        self._batch_queue_label = ctk.CTkLabel(parent, text="队列: 0 个文件",
                                               font=FONTS["small"], text_color=Colors.TEXT_MUTED)
        self._batch_queue_label.pack(anchor="w", padx=4, pady=1)
        self._batch_start_btn = ctk.CTkButton(
            parent, text="▶ 开始批量处理", command=self._batch_process,
            fg_color=Colors.SUCCESS, hover_color="#4CAF50",
            font=FONTS["body"], corner_radius=6)
        self._batch_start_btn.pack(fill="x", padx=4, pady=2)

        # ── 新功能4：历史记录 ──
        self._build_card_header(parent, "📋 历史记录")
        self._btn(parent, "查看历史", self._open_history_viewer,
                  fg_color=Colors.CARD).pack(fill="x", padx=4, pady=2)

        # ── 新功能6：API 服务 ──
        self._build_card_header(parent, "🌐 API 服务")
        self._api_btn = ctk.CTkButton(
            parent, text="启动 API 服务 (端口 5000)", command=self._toggle_api,
            fg_color=Colors.CARD, hover_color=Colors.CARD_HOVER,
            font=FONTS["body"], corner_radius=6, border_width=1, border_color=Colors.BORDER)
        self._api_btn.pack(fill="x", padx=4, pady=2)

        # ── 新功能7：特征库导入导出 ──
        self._build_card_header(parent, "🏪 特征库管理")
        feat_frame = ctk.CTkFrame(parent, fg_color="transparent")
        feat_frame.pack(fill="x", padx=4, pady=2)
        self._export_feat_btn = self._btn(feat_frame, "导出特征库", self._export_features,
                                          fg_color=Colors.CARD, width=90)
        self._export_feat_btn.pack(side="left", padx=(0, 2))
        self._import_feat_btn = self._btn(feat_frame, "导入特征库", self._import_features,
                                          fg_color=Colors.CARD, width=90)
        self._import_feat_btn.pack(side="left", padx=2)
        self._btn(parent, "浏览社区特征库", self._browse_market,
                  fg_color=Colors.CARD).pack(fill="x", padx=4, pady=2)

        # ── 新功能5：角色图片管理器 ──
        self._btn(parent, "角色库管理", self._open_role_manager,
                  fg_color=Colors.CARD).pack(fill="x", padx=4, pady=2)

        # ── 新功能9：更新检查 ──
        self._update_btn = self._btn(parent, "🔄 检查更新", self._check_update_manual,
                                     fg_color=Colors.CARD)
        self._update_btn.pack(fill="x", padx=4, pady=2)

        # ── 新功能12：直播输出 ──
        self._build_card_header(parent, "📺 直播输出")
        self._live_btn = ctk.CTkButton(
            parent, text="启动虚拟摄像头", command=self._toggle_live,
            fg_color=Colors.CARD, hover_color=Colors.CARD_HOVER,
            font=FONTS["body"], corner_radius=6, border_width=1, border_color=Colors.BORDER)
        self._live_btn.pack(fill="x", padx=4, pady=2)

        # ── 新功能15：语言切换 ──
        self._build_card_header(parent, "🌍 语言")
        lang_frame = ctk.CTkFrame(parent, fg_color="transparent")
        lang_frame.pack(fill="x", padx=4, pady=2)
        self._lang_combo = ctk.CTkOptionMenu(
            lang_frame, values=["简体中文", "English", "日本語"],
            command=self._switch_language,
            font=FONTS["body"], fg_color=Colors.BORDER,
            button_color=Colors.PRIMARY, button_hover_color=Colors.PRIMARY_HOVER,
            corner_radius=6)
        self._lang_combo.pack(fill="x")

        # ── 新功能16：模型版本管理 ──
        self._build_card_header(parent, "🧬 特征模型")
        model_frame = ctk.CTkFrame(parent, fg_color="transparent")
        model_frame.pack(fill="x", padx=4, pady=2)
        self._model_combo = ctk.CTkOptionMenu(
            model_frame, values=["FaceNet-VGGFace2"],
            command=self._switch_model,
            font=FONTS["body"], fg_color=Colors.BORDER,
            button_color=Colors.PRIMARY, button_hover_color=Colors.PRIMARY_HOVER,
            corner_radius=6)
        self._model_combo.pack(fill="x")
        self._btn(parent, "清空日志", self._clear_log,
                  fg_color=Colors.CARD).pack(fill="x", padx=4, pady=2)

    def _build_slider(self, parent, label, var, from_val, to_val, step):
        """滑动条行：标签 + 滑块 + 数值"""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=2)

        ctk.CTkLabel(row, text=label, font=FONTS["small"],
                     text_color=Colors.TEXT_MUTED, width=36).pack(side="left")

        slider = ctk.CTkSlider(
            row, from_=from_val, to=to_val,
            number_of_steps=int((to_val - from_val) / step),
            variable=var, progress_color=Colors.PRIMARY,
            button_color=Colors.PRIMARY,
            button_hover_color=Colors.PRIMARY_HOVER
        )
        slider.pack(side="left", fill="x", expand=True, padx=6)

        val_label = ctk.CTkLabel(
            row, text=f"{var.get():.2f}" if isinstance(var.get(), float) else str(var.get()),
            font=FONTS["small"], text_color=Colors.ACCENT, width=34
        )
        val_label.pack(side="left")

        def _upd(*_):
            v = var.get()
            val_label.configure(
                text=f"{v:.2f}" if isinstance(v, float) else str(v))
        var.trace_add("write", _upd)

    def _build_right(self, parent):
        # Preview area with dashed border
        preview_frame = ctk.CTkFrame(
            parent, fg_color=Colors.CANVAS_BG, corner_radius=8,
            border_width=2, border_color=Colors.DROP_BORDER
        )
        preview_frame.pack(fill="both", expand=True, pady=(0, 6))

        self._drop_label = tk.Label(
            preview_frame,
            text="✦\n\n拖入图片或视频开始识别\n\n支持 PNG · JPG · MP4",
            font=("Microsoft YaHei UI", 12),
            bg=self._hexify(Colors.CANVAS_BG),
            fg=self._hexify(Colors.TEXT_MUTED),
            justify="center", bd=0, highlightthickness=0
        )
        self._drop_label.place(relx=0.5, rely=0.5, anchor="center")

        self._preview_canvas = tk.Canvas(
            preview_frame, bg=self._hexify(Colors.CANVAS_BG),
            highlightthickness=0, bd=0
        )
        self._preview_canvas.pack(fill="both", expand=True)

        if DND_AVAILABLE:
            self._preview_canvas.drop_target_register(tkdnd.DND_FILES)
            self._preview_canvas.dnd_bind("<<Drop>>", self._on_drop)
        else:
            self._drop_label.config(
                text="主人，将图片拖拽到此处喵~\n（需安装 tkinterdnd2）\n\n✦\n\n或点击左侧「打开文件」"
            )

        # Log area
        log_frame = ctk.CTkFrame(parent, fg_color=Colors.CARD, corner_radius=8)
        log_frame.pack(fill="x", ipady=2)

        log_hdr = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_hdr.pack(fill="x", padx=8, pady=(4, 0))
        ctk.CTkLabel(log_hdr, text="运行日志", font=FONTS["small"],
                     text_color=Colors.TEXT_MUTED).pack(side="left")

        self._log_box = scrolledtext.ScrolledText(
            log_frame, height=6, bg=self._hexify(Colors.LOG_BG),
            fg=self._hexify(Colors.SUCCESS),
            font=("Consolas", 9), relief="flat",
            state="disabled", wrap="word"
        )
        self._log_box.pack(fill="x", padx=6, pady=4)

        # ── 新功能2：队列列表（在日志下方）───────────────────────────────────
        self._batch_list_frame = ctk.CTkFrame(parent, fg_color=Colors.CARD, corner_radius=8, height=80)
        self._batch_list_box = scrolledtext.ScrolledText(
            self._batch_list_frame, height=3, bg=self._hexify(Colors.LOG_BG),
            fg=self._hexify(Colors.TEXT_MUTED),
            font=("Consolas", 8), relief="flat",
            state="disabled", wrap="none"
        )
        self._batch_list_box.pack(fill="both", expand=True, padx=4, pady=4)

        # ── 新功能11：统计面板 ──────────────────────────────────────────────
        self._stats_frame = ctk.CTkFrame(parent, fg_color=Colors.CARD, corner_radius=8)
        self._stats_label = ctk.CTkLabel(self._stats_frame, text="📊 出场统计",
                                         font=FONTS["small"], text_color=Colors.TEXT_MUTED)
        self._stats_label.pack(anchor="w", padx=8, pady=(4, 2))
        self._stats_text = scrolledtext.ScrolledText(
            self._stats_frame, height=3, bg=self._hexify(Colors.LOG_BG),
            fg=self._hexify(Colors.TEXT),
            font=("Consolas", 8), relief="flat",
            state="disabled", wrap="word"
        )
        self._stats_text.pack(fill="x", padx=6, pady=4)

    @staticmethod
    def _hexify(hex_color):
        """#RRGGBB 转 Tk 兼容颜色字符串"""
        return hex_color

    # ── 模型初始化（异步）─────────────────────────────────────
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
        self._open_btn.configure(state="normal")
        self._rebuild_btn.configure(state="normal")
        self._load_btn.configure(state="normal")

    # ── 日志 ────────────────────────────────────────────────
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
        self.root.after(0, lambda: self._status_lbl.configure(text=text))

    # ── 预览帧更新 ──────────────────────────────────────────
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

    # ── 特征库操作 ──────────────────────────────────────────
    def _progress_update(self, current: int, total: int, elapsed: float, db_name: str):
        if total <= 0:
            return
        pct = min(current / total, 1.0)
        self.root.after(0, lambda: self._progress_bar.set(pct))
        name_str = f" [{db_name}]" if db_name else ""
        text = f"构建中{name_str} {current}/{total}"
        if current > 0 and current < total:
            eta_sec = (elapsed / current) * (total - current)
            if eta_sec >= 60:
                eta_str = f"剩余 {int(eta_sec // 60)}分{int(eta_sec % 60)}秒"
            else:
                eta_str = f"剩余 {int(eta_sec)}秒"
            text += f" {eta_str}"
        self.root.after(0, lambda t=text: self._progress_label.configure(text=t))

    def _load_db_now(self):
        if not self._models_ready:
            self._log("请等待模型加载完成后再操作")
            return
        db_name = self._db_var.get()
        self._log(f"加载特征库: {db_name}")
        self._progress_bar.set(0)
        self._progress_label.configure(text="准备加载...")

        def _run():
            db, built = get_or_build_database(db_name, force_rebuild=False,
                                              log_fn=self._log, progress_fn=self._progress_update)
            self._database = db
            self._db_name  = db_name
            cnt = len(db)
            self._log(f"特征库就绪: {cnt} 个角色")
            ndb = get_or_build_negative_database(log_fn=self._log)
            self._negative_db = ndb
            if ndb:
                self._log(f"✅ 负面特征库就绪：{len(ndb)} 个类别")
            self._set_status(f"✅ 特征库已加载（{cnt} 角色）")
            self._progress_bar.set(1.0)
            self._progress_label.configure(text=f"完成！共 {cnt} 个角色")
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
        self._progress_bar.set(0)
        self._progress_label.configure(text="准备重建...")

        def _run():
            db, built = get_or_build_database(db_name, force_rebuild=True,
                                              log_fn=self._log, progress_fn=self._progress_update)
            self._database = db
            self._db_name  = db_name
            self._log(f"✅ 重建完成: {len(db)} 个角色")
            ndb = get_or_build_negative_database(force_rebuild=True, log_fn=self._log)
            self._negative_db = ndb
            if ndb:
                self._log(f"✅ 负面特征库已重建：{len(ndb)} 个类别")
            self._set_status(f"✅ 特征库已重建（{len(db)} 角色）")
            self._progress_bar.set(1.0)
            self._progress_label.configure(text=f"完成！共 {len(db)} 个角色")
            if built and self._auto_shutdown_var.get():
                self._log("训练完成，即将关机...")
                self._shutdown_system()
        threading.Thread(target=_run, daemon=True).start()

    # ── 关机功能 ───────────────────────────────────────────
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
                self._log(f"不支持自动关机的系统: {system}")
                return
            self._log("关机命令已执行，系统即将关闭...")
        except Exception as e:
            self._log(f"关机失败: {e}，请手动关机或检查权限")

    # ── 文件处理 ───────────────────────────────────────────
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
            from modules.base import AIResultCollection

            db = self._database
            db_name = self._db_name
            if suggested != db_name or not db:
                self._log(f" 自动选择特征库: {suggested}")
                self.root.after(0, lambda: self._db_var.set(suggested))
                db, _ = get_or_build_database(suggested, log_fn=self._log, stop_event=self._stop_evt)
                self._database = db
                self._db_name  = suggested
                ndb = get_or_build_negative_database(log_fn=self._log)
                self._negative_db = ndb
                if ndb:
                    self._log(f"✅ 负面特征库已加载：{len(ndb)} 个类别")

            if not db:
                self._log("❌ 特征库为空，无法识别。请先建库或检查 ./data 文件夹")
                self._set_busy(False)
                return

            enable_emotion = self._emotion_mode_var.get()
            enable_speech = self._speech_mode_var.get()
            enable_nsfw = self._nsfw_mode_var.get()
            has_ai_modules = enable_emotion or enable_speech or enable_nsfw

            if has_ai_modules:
                light = not enable_speech
                if not _ensure_ai_modules(light=light, log_fn=self._log):
                    self._log("⚠️ AI 模块加载失败，跳过增强功能")
                    enable_emotion = enable_speech = enable_nsfw = False
                    has_ai_modules = False

            ai_results = AIResultCollection(path, {
                "emotion": enable_emotion,
                "speech": enable_speech,
                "nsfw": enable_nsfw,
            })

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
                    negative_db=self._negative_db,
                    enable_emotion=enable_emotion,
                    enable_nsfw=enable_nsfw,
                    ai_results=ai_results,
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
                    enable_emotion=enable_emotion,
                    enable_speech=enable_speech,
                    enable_nsfw=enable_nsfw,
                    negative_db=self._negative_db,
                    ai_results=ai_results,
                )
            else:
                self._log(f"⚠️  不支持的文件格式: {suffix}")
                self._set_busy(False)

            # Save AI results after main processing
            if has_ai_modules and ai_results and len(ai_results) > 0:
                out_base = Path(path).stem
                out_dir = Path(path).parent
                save_analysis_result(ai_results, str(out_dir / out_base), self._log)

            # ── 保存到历史记录（功能4）和 _last_results（功能1）───────────────
            source_type = "video" if suffix in VIDEO_EXTS else "image"
            save_to_history(path, source_type, [])
            # 保存结果用于导出
            self._last_results.append({
                "source": path,
                "timestamp": datetime.now().isoformat(),
                "source_type": source_type,
                "results": [],
            })

        threading.Thread(target=_load_and_run, daemon=True).start()

    def _stop_processing(self):
        self._stop_evt.set()
        self._log("⏹ 已发送停止信号…")

    def _set_busy(self, busy: bool):
        self._busy = busy
        def _upd():
            state = "normal" if busy else "disabled"
            self._stop_btn.configure(state=state)
        self.root.after(0, _upd)

    # ── 别名编辑器 ──────────────────────────────────────────
    def _open_alias_editor(self):
        win = tk.Toplevel(self.root)
        win.title("角色别名管理")
        win.geometry("600x500")
        win.configure(bg=Colors.BG)

        tk.Label(win, text="编辑 cname/name.json（每行一个别名，逗号分隔）",
                 bg=Colors.BG, fg=Colors.TEXT,
                 font=("Microsoft YaHei UI", 10)).pack(pady=8)

        txt = scrolledtext.ScrolledText(win, bg=self._hexify(Colors.LOG_BG),
                                        fg=Colors.TEXT,
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

        btn_frame = tk.Frame(win, bg=Colors.BG)
        btn_frame.pack(pady=6)
        tk.Button(btn_frame, text="💾 保存", command=_save,
                  bg=Colors.PRIMARY, fg="white", relief="flat",
                  font=("Microsoft YaHei UI", 10), cursor="hand2",
                  padx=16, pady=4).pack(side="left", padx=8)
        tk.Button(btn_frame, text="取消", command=win.destroy,
                  bg=Colors.BORDER, fg=Colors.TEXT, relief="flat",
                  font=("Microsoft YaHei UI", 10), cursor="hand2",
                  padx=16, pady=4).pack(side="left", padx=8)

    # ── 新功能1：导出报告 ──────────────────────────────────────────
    def _export_report_html(self):
        """导出 HTML 格式报告"""
        if not self._last_results:
            self._log("⚠️ 没有可导出的识别结果")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML 报告", "*.html"), ("所有文件", "*.*")]
        )
        if not path:
            return
        self._log("正在导出 HTML 报告...")
        try:
            from report_generator import ReportGenerator
            rg = ReportGenerator("MoeFace 识别报告")
            rg.source_file = self._last_results[0].get("source", "")
            for r in self._last_results:
                roles = r.get("results", [])
                for role in roles:
                    rg.unique_roles.add(role.get("name", "未知"))
            rg.total_processed = len(self._last_results)
            rg.generate_html(path)
            self._log(f"✅ HTML 报告已保存: {path}")
        except Exception as e:
            self._log(f"❌ 导出失败: {e}")

    def _export_report(self, fmt: str):
        """导出 JSON/CSV 报告"""
        if not self._last_results:
            self._log("⚠️ 没有可导出的识别结果")
            return
        ext = {"json": ".json", "csv": ".csv"}
        path = filedialog.asksaveasfilename(
            defaultextension=ext.get(fmt, ".json"),
            filetypes=[(f"{fmt.upper()} 文件", f"*.{fmt}"), ("所有文件", "*.*")]
        )
        if not path:
            return
        self._log(f"正在导出 {fmt.upper()}...")
        try:
            if fmt == "json":
                import json
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self._last_results, f, ensure_ascii=False, indent=2)
            elif fmt == "csv":
                import csv
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["文件名", "角色名", "相似度", "时间戳"])
                    for r in self._last_results:
                        source = r.get("source", "")
                        ts = r.get("timestamp", "")
                        for role in r.get("results", []):
                            writer.writerow([source, role.get("name", ""),
                                             role.get("score", 0), ts])
            self._log(f"✅ {fmt.upper()} 已保存: {path}")
        except Exception as e:
            self._log(f"❌ 导出失败: {e}")

    # ── 新功能2：批量处理 ──────────────────────────────────────────
    def _update_batch_queue_label(self):
        self._batch_queue_label.configure(text=f"队列: {len(self._queue)} 个文件")

    def _batch_add_files(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("图片/视频", " ".join(f"*{e}" for e in IMAGE_EXTS | VIDEO_EXTS)),
                       ("图片", " ".join(f"*{e}" for e in IMAGE_EXTS)),
                       ("视频", " ".join(f"*{e}" for e in VIDEO_EXTS))]
        )
        for p in paths:
            if p not in self._queue:
                self._queue.append(p)
        self._update_batch_queue_label()
        self._refresh_batch_list()
        self._log(f"📦 已添加 {len(paths)} 个文件到队列")

    def _batch_add_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        count = 0
        for ext in IMAGE_EXTS | VIDEO_EXTS:
            for f in Path(folder).rglob(f"*{ext}"):
                if str(f) not in self._queue:
                    self._queue.append(str(f))
                    count += 1
        self._update_batch_queue_label()
        self._refresh_batch_list()
        self._log(f"📦 已添加 {count} 个文件到队列")

    def _batch_clear(self):
        self._queue.clear()
        self._queue_idx = 0
        self._update_batch_queue_label()
        self._refresh_batch_list()
        self._log("🗑 已清空队列")

    def _refresh_batch_list(self):
        self._batch_list_box.config(state="normal")
        self._batch_list_box.delete("1.0", "end")
        for i, p in enumerate(self._queue[self._queue_idx:], self._queue_idx + 1):
            self._batch_list_box.insert("end", f"  {i}. {Path(p).name}\n")
        self._batch_list_box.config(state="disabled")

    def _batch_process(self):
        if not self._queue:
            self._log("⚠️ 队列为空，请先添加文件")
            return
        if self._busy:
            self._log("⚠️ 正在处理中")
            return

        def _run():
            self._set_busy(True)
            self._queue_idx = 0
            total = len(self._queue)
            while self._queue_idx < total:
                if self._stop_evt.is_set():
                    self._log("⏹ 批量处理已停止")
                    break
                path = self._queue[self._queue_idx]
                self._log(f"\n📦 [{self._queue_idx + 1}/{total}] {Path(path).name}")
                # 直接调用 dispatch 的内部逻辑
                self._dispatch_file_internal(path)
                self._queue_idx += 1
                self.root.after(0, self._refresh_batch_list)

            self._set_busy(False)
            self._log(f"✅ 批量处理完成: {self._queue_idx}/{total}")
            self.root.after(0, self._update_batch_queue_label)

        threading.Thread(target=_run, daemon=True).start()

    def _dispatch_file_internal(self, path: str):
        """内部文件处理（不检查 busy 状态，供批量处理使用）"""
        from modules.base import AIResultCollection
        suffix = Path(path).suffix.lower()
        suggested = get_db_name_from_filename(Path(path).name)

        self._stop_evt.clear()

        db = self._database
        db_name = self._db_name
        if suggested != db_name or not db:
            db, _ = get_or_build_database(suggested, log_fn=self._log, stop_event=self._stop_evt)
            self._database = db
            self._db_name = suggested
            ndb = get_or_build_negative_database(log_fn=self._log)
            self._negative_db = ndb

        if not db:
            self._log(f"❌ 特征库为空，跳过: {path}")
            return

        enable_emotion = self._emotion_mode_var.get()
        enable_speech = self._speech_mode_var.get()
        enable_nsfw = self._nsfw_mode_var.get()
        has_ai = enable_emotion or enable_speech or enable_nsfw
        if has_ai:
            _ensure_ai_modules(light=not enable_speech, log_fn=self._log)

        ai_results = AIResultCollection(path, {"emotion": enable_emotion, "speech": enable_speech, "nsfw": enable_nsfw})
        mn = self._min_neighbors_var.get()
        body_mode = self._body_mode_var.get()

        if suffix in IMAGE_EXTS:
            process_image_file(source=path, database=db, threshold=self._threshold_var.get(),
                               min_neighbors=mn, body_mode=body_mode, log_fn=self._log,
                               preview_fn=self._show_frame_cv, stop_event=self._stop_evt,
                               negative_db=self._negative_db, enable_emotion=enable_emotion,
                               enable_nsfw=enable_nsfw, ai_results=ai_results)
        elif suffix in VIDEO_EXTS:
            out = self._out_var.get().strip() or None
            process_video_file(source=path, database=db, output_path=out,
                               threshold=self._threshold_var.get(), skip_frames=self._skip_var.get(),
                               min_neighbors=mn, body_mode=body_mode, log_fn=self._log,
                               preview_fn=self._show_frame_cv, stop_event=self._stop_evt,
                               enable_emotion=enable_emotion, enable_speech=enable_speech,
                               enable_nsfw=enable_nsfw, negative_db=self._negative_db, ai_results=ai_results)
        else:
            self._log(f"⚠️ 不支持的文件格式: {suffix}")

    # ── 新功能3：系统托盘 + 全局快捷键（静默加载）──────────────────────
    def _init_tray(self):
        try:
            import pystray
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (64, 64), (108, 92, 231))
            draw = ImageDraw.Draw(img)
            draw.ellipse([8, 8, 56, 56], fill=(162, 155, 254))
            menu = pystray.Menu(
                pystray.MenuItem("显示窗口", lambda: self.root.deiconify()),
                pystray.MenuItem("截屏识别", self._tray_screenshot),
                pystray.MenuItem("切换特征库", self._tray_switch_db),
                pystray.MenuItem("退出", self._tray_quit),
            )
            self._tray_icon = pystray.Icon("MoeFace", img, "MoeFace", menu)
            threading.Thread(target=self._tray_icon.run, daemon=True).start()
        except Exception:
            pass  # pystray 不可用则静默跳过

    def _tray_screenshot(self):
        """托盘菜单：截屏识别"""
        def _run():
            try:
                from PIL import ImageGrab
                screenshot = ImageGrab.grab()
                import cv2, numpy as np, tempfile
                path = tempfile.mktemp(suffix=".png")
                screenshot.save(path)
                self.root.after(0, lambda: self._dispatch_file(path))
            except Exception as e:
                self._log(f"📸 截屏失败: {e}")
        threading.Thread(target=_run, daemon=True).start()

    def _tray_switch_db(self):
        self.root.after(0, lambda: self._db_combo.open_dropdown_menu())

    def _tray_quit(self):
        self.root.after(0, self.root.quit)

    # ── 新功能4：历史记录查看器 ──────────────────────────────────
    def _open_history_viewer(self):
        win = tk.Toplevel(self.root)
        win.title("识别历史记录")
        win.geometry("700x500")
        win.configure(bg=Colors.BG)

        # 搜索框
        search_frame = tk.Frame(win, bg=Colors.BG)
        search_frame.pack(fill="x", padx=10, pady=8)
        tk.Label(search_frame, text="搜索角色:", bg=Colors.BG, fg=Colors.TEXT,
                 font=("Microsoft YaHei UI", 9)).pack(side="left", padx=(0, 8))
        search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=search_var, width=30,
                                bg=Colors.LOG_BG, fg=Colors.TEXT, relief="flat")
        search_entry.pack(side="left", padx=(0, 8))

        # 表格
        cols = ("时间", "文件名", "类型", "角色")
        tree = ttk.Treeview(win, columns=cols, show="headings", height=15)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=150 if c == "时间" else 100)
        tree.pack(fill="both", expand=True, padx=10, pady=4)

        scrollbar = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        scrollbar.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scrollbar.set)

        def _refresh(search_text=""):
            tree.delete(*tree.get_children())
            records = load_history(search=search_text)
            for r in records:
                roles_str = ", ".join(str(name) for name in r.get("roles", [])[:3])
                if len(r.get("roles", [])) > 3:
                    roles_str += "..."
                tree.insert("", "end", values=(
                    r["timestamp"][:19], Path(r["filename"]).name,
                    r["source_type"], roles_str
                ))

        def _on_search(*_):
            _refresh(search_var.get().strip())

        search_var.trace_add("write", _on_search)
        _refresh()

        # 双击查看详情
        def _on_double_click(event):
            sel = tree.selection()
            if sel:
                values = tree.item(sel[0], "values")
                messagebox.showinfo("历史详情",
                                    f"时间: {values[0]}\n文件: {values[1]}\n类型: {values[2]}\n角色: {values[3]}",
                                    parent=win)

        tree.bind("<Double-1>", _on_double_click)

        # 关闭按钮
        tk.Button(win, text="关闭", command=win.destroy,
                  bg=Colors.PRIMARY, fg="white", relief="flat",
                  font=("Microsoft YaHei UI", 10), padx=16, pady=4).pack(pady=8)

        # 统计
        stats = get_history_stats()
        tk.Label(win, text=f"共 {stats['total']} 条记录", bg=Colors.BG,
                 fg=Colors.TEXT_MUTED, font=("Microsoft YaHei UI", 9)).pack()

    # ── 新功能5：角色图片管理器 ──────────────────────────────────
    def _open_role_manager(self):
        win = tk.Toplevel(self.root)
        win.title("角色训练图片管理器")
        win.geometry("700x500")
        win.configure(bg=Colors.BG)

        left_frame = tk.Frame(win, bg=Colors.BG, width=200)
        left_frame.pack(side="left", fill="y", padx=8, pady=8)
        left_frame.pack_propagate(False)

        right_frame = tk.Frame(win, bg=Colors.BG)
        right_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        tk.Label(left_frame, text="角色列表", bg=Colors.BG, fg=Colors.TEXT,
                 font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w")

        roles = scan_role_folders()
        listbox = tk.Listbox(left_frame, bg=Colors.LOG_BG, fg=Colors.TEXT,
                             selectbackground=Colors.PRIMARY, relief="flat")
        listbox.pack(fill="both", expand=True)
        for r in roles:
            listbox.insert("end", r)

        # 预览区域
        preview_label = tk.Label(right_frame, text="选择角色查看图片",
                                 bg=Colors.BG, fg=Colors.TEXT_MUTED)
        preview_label.pack()

        img_frame = tk.Frame(right_frame, bg=Colors.BG)
        img_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(img_frame, bg=Colors.LOG_BG, highlightthickness=0)
        scrollbar_v = tk.Scrollbar(img_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=Colors.LOG_BG)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_v.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_v.pack(side="right", fill="y")

        def _on_select(event):
            for w in scrollable_frame.winfo_children():
                w.destroy()
            sel = listbox.curselection()
            if not sel:
                return
            role = listbox.get(sel[0])
            role_path = _find_role_path(role)
            if not role_path:
                return
            imgs = list(role_path.glob("*"))[:50]
            preview_label.config(text=f"{role} — {len(imgs)} 张图片")
            from PIL import Image, ImageTk
            for i, img_path in enumerate(imgs[:20]):
                try:
                    pil_img = Image.open(img_path)
                    pil_img.thumbnail((120, 120))
                    tk_img = ImageTk.PhotoImage(pil_img)
                    lbl = tk.Label(scrollable_frame, image=tk_img, bg=Colors.LOG_BG)
                    lbl.image = tk_img
                    lbl.grid(row=i // 4, column=i % 4, padx=4, pady=4)
                except Exception:
                    pass

        listbox.bind("<<ListboxSelect>>", _on_select)

    # ── 新功能6：API 服务控制 ──────────────────────────────────
    def _toggle_api(self):
        if self._api_running:
            self._log("ℹ️ 请关闭终端窗口来停止 API 服务")
            return
        try:
            from api_server import start_api_server
            ok = start_api_server(port=5000, db_name=self._db_name or DEFAULT_DB_NAME,
                                  log_fn=self._log)
            if ok:
                self._api_running = True
                self._api_btn.configure(text="🟢 API 运行中 (端口 5000)", fg_color=Colors.SUCCESS)
                self._log("🌐 API 服务已启动: http://localhost:5000")
        except Exception as e:
            self._log(f"❌ API 启动失败: {e}")

    # ── 新功能7：特征库导入导出 ────────────────────────────────
    def _export_features(self):
        if not self._database:
            self._log("⚠️ 特征库为空，无法导出")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".moe",
            filetypes=[("MoeFace 特征库", "*.moe"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            ndb = self._negative_db or {}
            save_database_to_moe(self._database, Path(path).stem, negative_db=ndb)
            # 复制到目标路径
            src = FEATURES_DIR / f"{_safe_filename(Path(path).stem)}.moe"
            if src.exists():
                shutil.copy2(str(src), path)
                self._log(f"✅ 特征库已导出: {path}")
            else:
                self._log("✅ 特征库已保存")
        except Exception as e:
            self._log(f"❌ 导出失败: {e}")

    def _import_features(self):
        path = filedialog.askopenfilename(
            filetypes=[("MoeFace 特征库", "*.moe"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            dst = FEATURES_DIR / Path(path).name
            shutil.copy2(path, str(dst))
            self._log(f"✅ 特征库已导入: {dst.name}")
            self._log("💡 请通过「加载特征库」使用新导入的特征库")
        except Exception as e:
            self._log(f"❌ 导入失败: {e}")

    def _browse_market(self):
        import webbrowser
        webbrowser.open("https://github.com/ciallo0721-cmd/MoeFace/releases")
        self._log("🌐 已打开浏览器前往社区特征库页面")

    # ── 新功能9：更新检查 ─────────────────────────────────────
    def _check_update_auto(self):
        """后台自动检查更新"""
        def _run():
            try:
                import requests
                resp = requests.get(VERSION_URL, timeout=5)
                data = resp.json()
                latest = data.get("tag_name", "").lstrip("v")
                if latest and latest > VERSION:
                    self._update_available = True
                    self._new_version = latest
                    self.root.after(0, lambda: self._update_btn.configure(
                        text=f"🔄 新版本 {latest} 可用!", fg_color=Colors.SUCCESS))
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()

    def _check_update_manual(self):
        """手动检查更新"""
        if self._update_available:
            import webbrowser
            webbrowser.open(VERSION_DOWNLOAD_URL)
            self._log(f"🌐 正在前往 {VERSION_DOWNLOAD_URL}")
            return
        self._log("正在检查更新...")
        def _run():
            try:
                import requests
                resp = requests.get(VERSION_URL, timeout=5)
                data = resp.json()
                latest = data.get("tag_name", "").lstrip("v")
                if latest and latest > VERSION:
                    self.root.after(0, lambda: self._log(f"🔄 新版本 {latest} 可用！点击按钮下载"))
                    self.root.after(0, lambda: self._update_btn.configure(
                        text=f"🔄 新版本 {latest} 可用!", fg_color=Colors.SUCCESS))
                    self._update_available = True
                    self._new_version = latest
                else:
                    self.root.after(0, lambda: self._log("✅ 当前已是最新版本"))
            except Exception as e:
                self.root.after(0, lambda: self._log(f"⚠️ 检查更新失败: {e}"))
        threading.Thread(target=_run, daemon=True).start()

    # ── 新功能12：直播输出控制 ──────────────────────────────────
    def _toggle_live(self):
        try:
            from live_overlay import LiveOverlay
            if self._live_overlay is None:
                self._live_overlay = LiveOverlay()
            if self._live_overlay._running:
                self._live_overlay.stop()
                self._live_btn.configure(text="启动虚拟摄像头", fg_color=Colors.CARD)
                self._log("📺 虚拟摄像头已停止")
            else:
                ok = self._live_overlay.start()
                if ok:
                    self._live_btn.configure(text="🟢 虚拟摄像头运行中", fg_color=Colors.SUCCESS)
                    self._log("📺 虚拟摄像头已启动")
                else:
                    self._log("⚠️ 虚拟摄像头不可用，请安装 pyvirtualcam 和 OBS")
        except Exception as e:
            self._log(f"⚠️ 直播功能不可用: {e}")

    # ── 新功能15：语言切换 ──────────────────────────────────────
    def _switch_language(self, lang: str):
        try:
            from i18n.i18n import set_language
            lang_map = {"简体中文": "zh-CN", "English": "en", "日本語": "ja"}
            code = lang_map.get(lang, "zh-CN")
            set_language(code)
            self._log(f"🌍 语言已切换为: {lang}")
            self._log("💡 部分界面需重启应用后生效")
        except Exception as e:
            self._log(f"⚠️ 语言切换失败: {e}")

    # ── 新功能16：模型版本切换 ────────────────────────────────
    def _switch_model(self, model_name: str):
        self._log(f"🧬 切换模型: {model_name}")
        self._log("💡 切换后请重建特征库以生成新的特征向量")
        if model_name == "FaceNet-VGGFace2":
            self._current_model = "facenet-vggface2"
        else:
            self._current_model = "facenet-vggface2"
        # 重建需要重新加载模型
        global _models_ready
        _models_ready = False
        self._models_ready = False
        self._load_models_async()
        self.root.mainloop()


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
        self.negative_db = {}  # 负面特征库（非人脸样板）
        self.stats     = {"total_frames": 0, "processed": 0,
                          "found_names": {}, "errors": 0}
        self.stop_evt  = threading.Event()
        self._console_height = 0
        self._last_pct = -1.0
        self._last_eta_str = ""
        # ── AI 模块状态 ────────────────────────────────────────────────
        self._ai_collection = None

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
        
        # ── 加载负面特征库 ────────────────────────────────────────────
        self._header("① 加载负面特征库")
        self.negative_db = get_or_build_negative_database(
            force_rebuild=force_rebuild, log_fn=self._log
        )
        if self.negative_db:
            self._log(f"✅ 负面特征库就绪：{len(self.negative_db)} 个类别")
        
        CLIColors.p("")
        if not self.db:
            self._log("特征库为空，请检查 ./data 文件夹或添加角色图片", "error")
            sys.exit(1)
        
        CLIColors.p(f"\n  {CLIColors.SUCCESS}✨ 已加载 {len(self.db)} 个角色特征，就绪~ ✨{CLIColors.RESET}")

        # ── AI 模块初始化（如需）───────────────────────────────────────
        enable_emotion = self.args.get("emotion", False)
        enable_speech = self.args.get("speech", False)
        enable_nsfw = self.args.get("nsfw", False)
        self._has_ai = enable_emotion or enable_speech or enable_nsfw

        if self._has_ai:
            from modules.base import AIResultCollection
            self._ai_collection = AIResultCollection(
                self.args.get("source", "cli_input"),
                {"emotion": enable_emotion, "speech": enable_speech, "nsfw": enable_nsfw},
            )
            self._header("③ AI 识别模块")
            self._log(f"{'😊 情绪' if enable_emotion else ''}"
                       f"{' 🎤 语音' if enable_speech else ''}"
                       f"{' 🔞 NSFW' if enable_nsfw else ''}", "info")
            light = not enable_speech
            if not _ensure_ai_modules(light=light, log_fn=self._log):
                self._log("⚠️ AI 模块加载失败，跳过增强功能", "warn")
                self._has_ai = False

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
        min_neighbors = self.args.get("min_neighbors", 5)
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
                                              full_img=img, body_persons=persons,
                                              negative_db=self.negative_db)
            tag = CLIColors.SUCCESS if name else CLIColors.MUTED
            self._log(f"  {tag}{name or '未知'}{CLIColors.RESET} "
                      f"{CLIColors.MUTED}({score:.2f}){CLIColors.RESET} "
                      f"@ ({x1},{y1},{w}×{h})", "ok" if name else "skip")
            if name:
                self.stats["found_names"][name] = \
                    self.stats["found_names"].get(name, 0) + 1

        # ── AI 增强：情绪识别 ──────────────────────────────────────────
        if self._has_ai and self.args.get("emotion", False) and _EMOTION_MODULE is not None:
            self._log("😊 情绪识别中...")
            try:
                emotions = _EMOTION_MODULE.process_frame(img, timestamp=0.0)
                if emotions:
                    for emo in emotions:
                        self._ai_collection.add(AIResult(
                            module="emotion", event_type="face_emotion",
                            timestamp=0.0, data=emo, confidence=emo["confidence"],
                        ))
                        bx, by, bw, bh = emo["bbox"]
                        img = draw_chinese_text(
                            img, f"[{emo['emotion_label']}] {emo['confidence']:.0%}",
                            (bx, by + bh + 5), 14, (255, 200, 0)
                        )
                    self._log(f"  ✅ 检测到 {len(emotions)} 张人脸情绪")
            except Exception as e:
                self._log(f"  ⚠️ 情绪识别异常: {e}")

        # ── AI 增强：NSFW 检测 ─────────────────────────────────────────
        if self._has_ai and self.args.get("nsfw", False) and _NSFW_MODULE is not None:
            self._log("🔞 NSFW 检测中...")
            try:
                nsfw_result = _NSFW_MODULE.process_frame(img, timestamp=0.0)
                if nsfw_result:
                    self._ai_collection.add(nsfw_result)
                    nsfw_label = nsfw_result.data.get("label_cn", "未知")
                    nsfw_score = nsfw_result.data.get("nsfw_score", 0.0)
                    self._log(f"  {nsfw_label}: {nsfw_score:.1%}")
                    img = draw_chinese_text(
                        img, f"NSFW: {nsfw_label} ({nsfw_score:.1%})",
                        (10, 30), 16, (255, 100, 100)
                    )
            except Exception as e:
                self._log(f"  ⚠️ NSFW 检测异常: {e}")

        output = self.args.get("output")
        if output:
            cv2.imwrite(output, img)
            self._log(f"已保存标注图: {output}", "ok")

        # 保存 AI 结果
        if self._ai_collection and len(self._ai_collection) > 0:
            save_analysis_result(self._ai_collection, path, self._log)

    def _run_video(self, source: str, output_path: str):
        """识别视频文件（同时人体框+骨骼 + 人脸识别）"""
        threshold    = self.args.get("threshold", 0.45)
        skip_frames  = self.args.get("skip_frames", 2)
        min_neighbors = self.args.get("min_neighbors", 5)
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
                        if not _is_valid_face_box(fw, fh):
                            continue
                        x1, y1 = max(0,x), max(0,y)
                        x2, y2 = min(frame.shape[1],x+fw), min(frame.shape[0],y+fh)
                        face = frame[y1:y2, x1:x2]
                        if face.size == 0:
                            continue
                        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        name, score = recognize_character(face_rgb, self.db, threshold,
                                                          full_img=frame, body_persons=persons,
                                                          negative_db=self.negative_db)
                        if not name or score < threshold:
                            continue
                        frame = draw_modern_recognition_box(frame, x1, y1, x2, y2, name, score)
                        if name:
                            self.stats["found_names"][name] = \
                                self.stats["found_names"].get(name, 0) + 1
                            # 收集画面角色时间线（供语音角色关联）
                            if self._has_ai and self.args.get("speech", False):
                                ts = frame_idx / fps if fps > 0 else 0.0
                                if not hasattr(self, '_face_timeline') or self._face_timeline is None:
                                    self._face_timeline = []
                                self._face_timeline.append({
                                    "timestamp": ts,
                                    "names": [name],
                                })

                    self.stats["processed"] += 1

                # ── AI 增强：帧级情绪 + NSFW ──────────────────────────
                if frame_idx % skip_frames == 0:
                    timestamp = frame_idx / fps if fps > 0 else 0.0

                    if self._has_ai and self.args.get("emotion", False) and _EMOTION_MODULE is not None:
                        try:
                            emotions = _EMOTION_MODULE.process_frame(frame, timestamp)
                            for emo in emotions:
                                self._ai_collection.add(AIResult(
                                    module="emotion", event_type="face_emotion",
                                    timestamp=timestamp, data=emo, confidence=emo["confidence"],
                                ))
                                bx, by, _, _ = emo["bbox"]
                                frame = draw_chinese_text(
                                    frame, f"{emo['emotion_label']}",
                                    (bx, by - 12), 11, (255, 200, 0)
                                )
                        except Exception:
                            pass

                    if self._has_ai and self.args.get("nsfw", False) and _NSFW_MODULE is not None:
                        try:
                            nsfw_result = _NSFW_MODULE.process_frame(frame, timestamp)
                            if nsfw_result:
                                self._ai_collection.add(nsfw_result)
                        except Exception:
                            pass

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

        # ── AI 增强：语音转文字 + 说话人分离 ────────────────────────────
        if self._has_ai and self.args.get("speech", False) and _SPEECH_MODULE is not None:
            self._log("🎤 语音转文字 + 说话人分离中...")
            try:
                known_speakers = None
                if self.db:
                    known_speakers = {}
                    for i, name in enumerate(self.db.keys()):
                        known_speakers[f"speaker_{i}"] = name

                speech_results = _SPEECH_MODULE.process_video(
                    video_path=source,
                    language="auto",
                    enable_diarization=True,
                    known_speakers=None,
                    log_fn=self._log,
                    stop_event=self.stop_evt,
                    face_timeline=getattr(self, '_face_timeline', None),
                )
                self._ai_collection.extend(speech_results.results)

                utterances = [r for r in speech_results.results
                              if r.get("event_type") == "utterance"]
                self._log(f"  ✅ 共转录 {len(utterances)} 句话")
                for utt in utterances[:5]:
                    d = utt.get("data", {})
                    self._log(f"  [{d.get('speaker_name','?')}] {d.get('text','')[:60]}")
                if len(utterances) > 5:
                    self._log(f"  ...还有 {len(utterances) - 5} 句")

                # 语义 NSFW
                if self.args.get("nsfw", False) and _NSFW_MODULE is not None:
                    for utt in utterances[:]:
                        d = utt.get("data", {})
                        text = d.get("text", "")
                        if text.strip():
                            nsfw_text = _NSFW_MODULE.process_text(
                                text=text, timestamp=d.get("start", 0.0), source="transcript",
                            )
                            if nsfw_text and nsfw_text.confidence > 0:
                                self._ai_collection.add(nsfw_text)
            except Exception as e:
                self._log(f"  ⚠️ 语音识别异常: {e}")

        # 保存 AI 结果
        if self._ai_collection and len(self._ai_collection) > 0:
            save_analysis_result(self._ai_collection, source, self._log)

    def _run_camera(self):
        """摄像头实时识别（同时人体框+骨骼 + 人脸识别）"""
        threshold    = self.args.get("threshold", 0.45)
        min_neighbors = self.args.get("min_neighbors", 5)
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
                    if not _is_valid_face_box(fw, fh):
                        continue
                    x1, y1 = max(0,x), max(0,y)
                    x2, y2 = min(frame.shape[1],x+fw), min(frame.shape[0],y+fh)
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    name, score = recognize_character(face_rgb, self.db, threshold,
                                                      full_img=frame, body_persons=persons,
                                                      negative_db=self.negative_db)
                    if not name or score < threshold:
                        continue
                    frame = draw_modern_recognition_box(frame, x1, y1, x2, y2, name, score)

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
  {CLIColors.LGREEN}--emotion{CLIColors.RESET}      😊 启用表情与情绪识别（实验性）
  {CLIColors.LGREEN}--speech{CLIColors.RESET}       🎤 启用语音转文字+说话人分离（实验性）
  {CLIColors.LGREEN}--nsfw{CLIColors.RESET}         🔞 启用 NSFW 内容检测（仅报告，实验性）
  {CLIColors.LGREEN}--list{CLIColors.RESET}         列出所有可用特征库

{CLIColors.TITLE}示例:{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode cli --source 视频.mp4 --output out.mp4 --body{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode cli --camera --threshold 0.6 --body{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode cli --source 图片.jpg --output annotated.jpg --body{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode cli --source 视频.mp4 --db-name 永雏塔菲 --body{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode cli --source 视频.mp4 --emotion --nsfw --body{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode cli --source 视频.mp4 --speech --nsfw{CLIColors.RESET}
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
        default=5,
        help="人脸检测灵敏度，越大误检越少（默认: 5）"
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
    # ── AI 识别增强模块（2026-06-29 新增）─────────────────────────────
    parser.add_argument(
        "--emotion",
        action="store_true",
        help="😊 启用表情与情绪识别（实时检测面部情绪并标注变化节点）"
    )
    parser.add_argument(
        "--speech",
        action="store_true",
        help="🎤 启用语音转文字 + 说话人分离（需安装 faster-whisper，首次运行会下载模型）"
    )
    parser.add_argument(
        "--nsfw",
        action="store_true",
        help="🔞 启用 NSFW 内容检测（视觉+语义双通道，仅报告不屏蔽）"
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
            # ── AI 模块标志 ────────────────────────────────────────
            "emotion":      args.emotion,
            "speech":       args.speech,
            "nsfw":         args.nsfw,
        }
        cli = MoeFaceCLI(cli_args)
        try:
            cli.run()
        except KeyboardInterrupt:
            CLIColors.p(f"\n{CLIColors.WARNING}⏹ 已中断{CLIColors.RESET}")
            cli.stop_evt.set()
            sys.exit(0)