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

CASCADE_PATH = RESOURCE_DIR / "lbpcascade_animeface.xml"
FONT_PATH    = RESOURCE_DIR / "simhei.ttf"
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

    # 选模型：全量 > 轻量
    model_path = os.path.join(base_dir, 'pose_landmarker.task')
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, 'pose_landmarker_lite.task')
        if not os.path.exists(model_path):
            log_fn("❌ 未找到 pose_landmarker.task，请从 Google MediaPipe 下载放到项目目录")
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
            repacked = os.path.join(base_dir, 'mp_pose_repacked.task')
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
                       min_neighbors=3, body_mode=False,
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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    txt = f"{name} ({score:.2f})" if name else f"Unknown ({score:.2f})"
                    frame = draw_chinese_text(frame, txt, (x1, max(0, y1 - 25)), 16, (0, 255, 0))
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
        self._negative_db: dict = {}  # 负面特征库
        self._stop_evt: threading.Event = threading.Event()
        self._busy      = False
        self._preview_img = None
        self._models_ready = False

        self._auto_shutdown_var = tk.BooleanVar(value=False)
        self._body_mode_var = tk.BooleanVar(value=False)  # 人体姿态检测模式
        # ── AI 识别模块开关（2026-06-29 新增）─────────────────────────
        self._emotion_mode_var = tk.BooleanVar(value=False)  # 情绪识别
        self._speech_mode_var = tk.BooleanVar(value=False)   # 语音转文字
        self._nsfw_mode_var = tk.BooleanVar(value=False)     # NSFW 检测

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

        # ── AI 识别模块复选框 ──────────────────────────────────────────
        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=4)
        self._lbl(parent, "AI 识别增强（实验性功能）", PANEL, MUTED,
                  font=("微软雅黑", 8)).pack(**pad)

        self._emotion_cb = tk.Checkbutton(
            parent, text="😊 表情 & 情绪识别",
            variable=self._emotion_mode_var,
            bg=PANEL, fg=TEXT, selectcolor=PANEL, font=("微软雅黑", 9),
            activebackground=PANEL, activeforeground="#f59e0b"
        )
        self._emotion_cb.pack(anchor="w", padx=10, pady=1)

        self._speech_cb = tk.Checkbutton(
            parent, text="🎤 语音转文字 + 说话人分离",
            variable=self._speech_mode_var,
            bg=PANEL, fg=TEXT, selectcolor=PANEL, font=("微软雅黑", 9),
            activebackground=PANEL, activeforeground="#f59e0b"
        )
        self._speech_cb.pack(anchor="w", padx=10, pady=1)

        self._nsfw_cb = tk.Checkbutton(
            parent, text="🔞 NSFW 内容检测（仅报告）",
            variable=self._nsfw_mode_var,
            bg=PANEL, fg=TEXT, selectcolor=PANEL, font=("微软雅黑", 9),
            activebackground=PANEL, activeforeground="#f59e0b"
        )
        self._nsfw_cb.pack(anchor="w", padx=10, pady=1)

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
            # ── 加载负面特征库 ────────────────────────────────────────
            ndb = get_or_build_negative_database(log_fn=self._log)
            self._negative_db = ndb
            if ndb:
                self._log(f"✅ 负面特征库就绪：{len(ndb)} 个类别")
            # ──────────────────────────────────────────────────────────
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
            # ── 重建负面特征库 ────────────────────────────────────────
            ndb = get_or_build_negative_database(force_rebuild=True, log_fn=self._log)
            self._negative_db = ndb
            if ndb:
                self._log(f"✅ 负面特征库已重建：{len(ndb)} 个类别")
            # ──────────────────────────────────────────────────────────
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
            from modules.base import AIResultCollection

            db = self._database
            db_name = self._db_name
            if suggested != db_name or not db:
                self._log(f" 自动选择特征库: {suggested}")
                self.root.after(0, lambda: self._db_var.set(suggested))
                db, _ = get_or_build_database(suggested, log_fn=self._log, stop_event=self._stop_evt)
                self._database = db
                self._db_name  = suggested
                # ── 加载负面特征库 ────────────────────────────────────
                ndb = get_or_build_negative_database(log_fn=self._log)
                self._negative_db = ndb
                if ndb:
                    self._log(f"✅ 负面特征库已加载：{len(ndb)} 个类别")
                # ──────────────────────────────────────────────────────

            if not db:
                self._log("❌ 特征库为空，无法识别。请先建库或检查 ./data 文件夹")
                self._set_busy(False)
                return

            # ── AI 模块配置 ────────────────────────────────────────────
            enable_emotion = self._emotion_mode_var.get()
            enable_speech = self._speech_mode_var.get()
            enable_nsfw = self._nsfw_mode_var.get()
            has_ai_modules = enable_emotion or enable_speech or enable_nsfw

            if has_ai_modules:
                light = not enable_speech  # 如果不需要语音，轻量加载
                if not _ensure_ai_modules(light=light, log_fn=self._log):
                    self._log("⚠️ AI 模块加载失败，跳过增强功能")
                    enable_emotion = enable_speech = enable_nsfw = False
                    has_ai_modules = False

            # ── 结果收集器 ─────────────────────────────────────────────
            ai_results = AIResultCollection(path, {
                "emotion": enable_emotion,
                "speech": enable_speech,
                "nsfw": enable_nsfw,
            })

            mn = self._min_neighbors_var.get()
            body_mode = self._body_mode_var.get()

            def _ai_done():
                """主线处理完成后的 AI 回调"""
                # 保存 JSON 结果
                if len(ai_results) > 0:
                    out_base = Path(path).stem
                    out_dir = Path(path).parent
                    save_analysis_result(ai_results, str(out_dir / out_base), self._log)
                self._set_busy(False)

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
                )
            elif suffix in VIDEO_EXTS:
                self._log(f"\n🎬  视频: {path}")
                out = self._out_var.get().strip() or None

                # 运行主识别
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
                    # ── AI 增强模块参数 ────────────────────────────────
                    enable_emotion=enable_emotion,
                    enable_speech=enable_speech,
                    enable_nsfw=enable_nsfw,
                    # ── 负面特征库 ────────────────────────────────────
                    negative_db=self._negative_db,
                    ai_results=ai_results,
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
                                                          full_img=frame, body_persons=persons,
                                                          negative_db=self.negative_db)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        frame = draw_chinese_text(frame,
                            f"{name or '?'} ({score:.2f})",
                            (x1, max(0,y1-25)), 16, (0,255,0))
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
                                                      full_img=frame, body_persons=persons,
                                                      negative_db=self.negative_db)
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