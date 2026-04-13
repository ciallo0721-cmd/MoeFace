"""
MoeFace — 动漫人脸识别系统
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

# ── 确保以脚本所在目录为基准路径 ────────────────────────────────────────────
import sys as _sys

def _resource_base() -> Path:
    """兼容 PyInstaller 打包：打包后资源在 _MEIPASS，开发时在脚本目录"""
    if getattr(_sys, "frozen", False) and hasattr(_sys, "_MEIPASS"):
        return Path(_sys._MEIPASS)
    return Path(__file__).resolve().parent

BASE_DIR = Path(__file__).resolve().parent  # 数据/特征库写在可执行文件旁边
RESOURCE_DIR = _resource_base()             # XML/字体等只读资源

os.chdir(BASE_DIR)

CASCADE_PATH = RESOURCE_DIR / "lbpcascade_animeface.xml"
FONT_PATH    = RESOURCE_DIR / "simhei.ttf"
FEATURES_DIR = BASE_DIR / "features"
DATA_DIR     = BASE_DIR / "data"
CNAME_PATH   = RESOURCE_DIR / "cname" / "name.json"
DEFAULT_DB_NAME = "全部特征库"

FEATURES_DIR.mkdir(exist_ok=True)

# ── 延迟导入重型库（避免 import 时卡死 GUI）───────────────────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

try:
    import tkinterdnd2 as tkdnd
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

# ── 别名模块：从 cname/name.json 加载 ─────────────────────────────────────
def load_alias_map(path: Path = CNAME_PATH):
    """
    返回 list[dict]，每项格式：
        {"db_name": "角色名", "aliases": ["别名1", "别名2", ...]}
    """
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
    """根据文件名中的关键词匹配最合适的角色特征库名称"""
    name_lower = filename.lower()
    for entry in ALIAS_MAP:
        for alias in entry.get("aliases", []):
            if alias.lower() in name_lower:
                return entry["db_name"]
    return DEFAULT_DB_NAME


# ── JSON 特征库管理 ────────────────────────────────────────────────────────
def _safe_json_name(name: str) -> str:
    """生成安全文件名（保留汉字、字母、数字、常用符号）"""
    keep = set("._- ·•")
    return "".join(c for c in name if c.isalnum() or c in keep or "\u4e00" <= c <= "\u9fff")

def save_database_to_json(database: dict, json_name: str):
    if not database:
        return None
    safe = _safe_json_name(json_name)
    json_path = FEATURES_DIR / f"{safe}.json"
    serializable = {n: emb.tolist() for n, emb in database.items()}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return json_path

def load_database_from_json(json_name: str):
    import numpy as np
    safe = _safe_json_name(json_name)
    json_path = FEATURES_DIR / f"{safe}.json"
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {n: np.array(e) for n, e in raw.items()}
    except Exception as e:
        warnings.warn(f"(｡•́︿•̀｡),主人,加载特征库失败了: {e}")
        return None


# ── 核心识别逻辑（懒加载，首次使用时初始化）──────────────────────────────────
_model_lock = threading.Lock()
_models_ready = False
_anime_cascade = None
_resnet = None
_device = None

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
    import cv2, torch
    import numpy as np
    
    # 每个线程加载自己的 CascadeClassifier（线程安全）
    cascade_path = Path(__file__).parent / "lbpcascade_animeface.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    
    try:
        with open(image_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
    except Exception:
        return None

    # 限制最大分辨率，防止内存溢出
    MAX_DIM = 4096
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.02, minNeighbors=3,
        minSize=(20, 20), maxSize=(800, 800)
    )
    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    x, y, w, h = faces[0]
    face      = img[y:y+h, x:x+w]
    face_rgb  = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_rsz  = cv2.resize(face_rgb, (160, 160))
    tensor    = (torch.tensor(face_rsz).permute(2, 0, 1)
                 .float().unsqueeze(0).to(_device) / 255.0)
    with torch.no_grad():
        emb = _resnet(tensor).cpu().numpy().flatten()
    return emb


def build_database(data_root: Path, log_fn=print, progress_fn=None):
    """
    构建特征库，支持进度回调和多线程加速。
    
    progress_fn(current, total, elapsed_sec, db_name) -> None
        current: 已处理图片数
        total: 总图片数
        elapsed_sec: 已耗时（秒）
        db_name: 当前角色名
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    database = {}
    if not data_root.exists():
        log_fn(f"❌呜呜呜,主人,路径不存在喵～(｡•́︿•̀｡): {data_root}")
        return database

    subdirs = [d for d in data_root.iterdir() if d.is_dir()]
    exts    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def _gather(folder):
        files = []
        for ext in exts:
            files += list(folder.glob(f"*{ext}"))
            files += list(folder.glob(f"*{ext.upper()}"))
        return files

    def _extract_single(img_path):
        """提取单张图片特征，用于多线程"""
        e = extract_features_from_image(str(img_path), lambda *_: None)
        return (str(img_path), e)
    
    # 统计总图片数和总角色数（每角色最多50张）
    MAX_PER_PERSON = 50
    total_images = 0
    person_images = {}  # {person_name: [img_paths]}
    
    if subdirs:
        for person_dir in subdirs:
            imgs = _gather(person_dir)
            if imgs:
                person_images[person_dir.name] = imgs[:MAX_PER_PERSON]
                total_images += min(len(imgs), MAX_PER_PERSON)
    else:
        imgs = _gather(data_root)
        if imgs:
            person_images[data_root.name] = imgs[:MAX_PER_PERSON]
            total_images = min(len(imgs), MAX_PER_PERSON)
    
    log_fn(f"多角色模式，共 {len(person_images)} 个角色，{total_images} 张图片")
    
    # 多线程配置：限制为 2 线程，避免内存爆炸
    import os as _os
    max_workers = min(2, total_images)
    
    start_time = time.time()
    processed = [0]  # 用列表包装以便在闭包中修改
    
    def _process_person(person_dir: Path):
        import gc
        imgs = person_images[person_dir.name][:50]  # 每角色最多50张，避免内存爆炸
        if not imgs:
            return None
        
        # 进度回调：角色开始
        if progress_fn:
            progress_fn(processed[0], total_images, time.time() - start_time, person_dir.name)
        
        all_embs = []
        for i, p in enumerate(imgs):
            e = extract_features_from_image(str(p), lambda *_: None)
            if e is not None:
                all_embs.append(e)
            processed[0] += 1
            
            # 每处理 5 张图更新一次进度
            if progress_fn and (i + 1) % 5 == 0:
                progress_fn(processed[0], total_images, time.time() - start_time, person_dir.name)
        
        # 强制垃圾回收，释放内存
        gc.collect()
        
        if all_embs:
            return (person_dir.name, np.mean(all_embs, axis=0))
        return None
    
    # 多线程处理各角色
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_person, d): d for d in 
                   [Path(data_root / name) for name in person_images.keys()]}
        
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                name, avg = result
                database[name] = avg
                log_fn(f"  ✅ {name} 完成")
            else:
                person_name = futures[future].name
                log_fn(f"  ⚠️  {person_name} 无有效人脸喵～(｡•́︿•̀｡)")
            
            # 角色完成时更新进度
            if progress_fn:
                progress_fn(processed[0], total_images, time.time() - start_time, "")

    # 最终进度 100%
    if progress_fn:
        progress_fn(total_images, total_images, time.time() - start_time, "")
    
    return database


def get_or_build_database(db_name: str, force_rebuild=False, log_fn=print, progress_fn=None):
    if not force_rebuild:
        db = load_database_from_json(db_name)
        if db is not None:
            log_fn(f"✅ 从缓存加载特征库 [{db_name}]，共 {len(db)} 个角色")
            return db

    if db_name == DEFAULT_DB_NAME:
        log_fn("构建全部特征库喵...")
        db = build_database(DATA_DIR, log_fn, progress_fn)
    else:
        db_path = DATA_DIR / db_name
        if not db_path.exists():
            log_fn(f"❌ 文件夹不存在: {db_path}")
            return {}
        log_fn(f"构建特征库喵~: {db_name}")
        db = build_database(db_path, log_fn, progress_fn)

    if db:
        save_database_to_json(db, db_name)
        log_fn(f"✅ 特征库已缓存喵~，共 {len(db)} 个角色")
    return db


# ── 识别辅助 ──────────────────────────────────────────────────────────────
def cosine_similarity(a, b):
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

def recognize_face(face_img, database: dict, threshold=0.45):
    import cv2, torch
    face   = cv2.resize(face_img, (160, 160))
    tensor = (torch.tensor(face).permute(2, 0, 1)
              .float().unsqueeze(0).to(_device) / 255.0)
    with torch.no_grad():
        emb = _resnet(tensor).cpu().numpy().flatten()
    best_name, best_score = None, 0.0
    for name, ref in database.items():
        s = cosine_similarity(emb, ref)
        if s > best_score:
            best_score, best_name = s, name
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


# ── 视频/图片处理（运行于子线程）──────────────────────────────────────────────
try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


def _add_audio(src_video, out_video, tmp_video):
    if not MOVIEPY_AVAILABLE:
        shutil.move(tmp_video, out_video)
        return
    try:
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
                       min_neighbors=3,
                       log_fn=print, preview_fn=None, done_fn=None):
    """识别单张图片"""
    import cv2
    import numpy as np
    try:
        with open(source, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            log_fn("❌ 无法读取图片喵~")
            if done_fn: done_fn()
            return

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _anime_cascade.detectMultiScale(
            gray, scaleFactor=1.02, minNeighbors=min_neighbors,
            minSize=(20, 20), maxSize=(800, 800)
        )
        log_fn(f"检测到 {len(faces)} 张人脸")

        for (x, y, w, h) in faces:
            x1 = max(0, x); y1 = max(0, y)
            x2 = min(img.shape[1], x + w)
            y2 = min(img.shape[0], y + h)
            face     = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            name, score = recognize_face(face_rgb, database, threshold)
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
                       log_fn=print, preview_fn=None,
                       stop_event: threading.Event = None, done_fn=None):
    """处理视频文件，支持逐帧预览"""
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

    out = tmp_out = None
    if output_path:
        tmp_dir = Path(output_path).parent / "temp"
        tmp_dir.mkdir(exist_ok=True)
        tmp_fd, tmp_out = tempfile.mkstemp(suffix=".mp4", dir=str(tmp_dir))
        os.close(tmp_fd)
        out = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx  = 0
    found_names: set = set()

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
                    face_rgb       = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    name, score    = recognize_face(face_rgb, database, threshold)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    txt = f"{name} ({score:.2f})" if name else f"Unknown ({score:.2f})"
                    frame = draw_chinese_text(frame, txt, (x1, max(0, y1 - 25)), 16, (0, 255, 0))
                    if name:
                        found_names.add(name)

                if preview_fn:
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
        self._preview_img = None   # 当前预览 PIL Image

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

        # ── 顶部标题栏 ───────────────────────────────────────────────────
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

        # ── 主体区域（左 + 右）──────────────────────────────────────────
        body = tk.Frame(root, bg=DARK)
        body.pack(fill="both", expand=True)

        # 左侧控制面板
        left = tk.Frame(body, bg=PANEL, width=260)
        left.pack(side="left", fill="y", padx=(8, 0), pady=8)
        left.pack_propagate(False)

        # 右侧：预览 + 日志
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

        # — 特征库选择 —
        self._lbl(parent, "特征库", PANEL, MUTED,
                  font=("微软雅黑", 8)).pack(**pad)

        db_frame = tk.Frame(parent, bg=PANEL)
        db_frame.pack(fill="x", padx=10, pady=2)

        self._db_var = tk.StringVar(value=DEFAULT_DB_NAME)
        db_names = [DEFAULT_DB_NAME] + sorted(
            {e["db_name"] for e in ALIAS_MAP}
        )
        self._db_combo = ttk.Combobox(db_frame, textvariable=self._db_var,
                                      values=db_names, state="readonly", width=20)
        self._db_combo.pack(side="left", fill="x", expand=True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
                         fieldbackground="#2a2a3e", background="#2a2a3e",
                         foreground="#cdd6f4", bordercolor="#7c3aed",
                         arrowcolor="#cdd6f4")

        # — 进度条框架 —
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
        
        # — 重建特征库按钮 —
        self._btn(parent, " 重建特征库", self._rebuild_db,
                  ACCENT).pack(fill="x", padx=10, pady=(4, 2))
        self._btn(parent, " 加载特征库", self._load_db_now,
                  "#374151").pack(fill="x", padx=10, pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # — 阈值 —
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

        # — 跳帧数 —
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

        # — 检测灵敏度（minNeighbors，越大越严格/误检越少）—
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

        # — 输出视频路径 —
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

        # — 打开文件按钮 —
        self._btn(parent, "  打开图片/视频", self._open_file,
                  ACCENT).pack(fill="x", padx=10, pady=2)
        self._stop_btn = self._btn(parent, "⏹  停止处理", self._stop_processing,
                                   "#dc2626", state="disabled")
        self._stop_btn.pack(fill="x", padx=10, pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # — 别名管理入口 —
        self._btn(parent, " 管理角色别名", self._open_alias_editor,
                  "#374151").pack(fill="x", padx=10, pady=2)
        self._btn(parent, " 清空日志", self._clear_log,
                  "#374151").pack(fill="x", padx=10, pady=2)

    def _build_right(self, parent, DARK, PANEL, TEXT, MUTED):
        # 上方：拖拽/预览区
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

        # 绑定拖拽
        if DND_AVAILABLE:
            self._preview_canvas.drop_target_register(tkdnd.DND_FILES)
            self._preview_canvas.dnd_bind("<<Drop>>", self._on_drop)
        else:
            # 无 tkinterdnd2 时显示提示
            self._drop_label.config(
                text="主人,将图片或视频拖拽到此处喵~（需安装 tkinterdnd2）\n或点击左侧「打开图片/视频」按钮\n\n支持 JPG / PNG / MP4 / AVI / MKV 等"
            )

        # 下方：日志
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
            if ok:
                self._set_status("✅ 就绪，请拖入文件")
            else:
                self._set_status("❌ 模型加载失败(｡•́︿•̀｡)，请检查依赖")
        threading.Thread(target=_run, daemon=True).start()

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
        """将 OpenCV BGR 图像显示到 Canvas（在主线程执行）"""
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
            self._preview_canvas._tk_img = tk_img  # 防止 GC
            self._drop_label.place_forget()
        self.root.after(0, _upd)

    # ── 特征库操作 ────────────────────────────────────────────────────────
    def _progress_update(self, current: int, total: int, elapsed: float, db_name: str):
        """进度回调：更新进度条和剩余时间"""
        if total <= 0:
            return
        
        pct = min(current / total * 100, 100)
        # 估算剩余时间
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
        db_name = self._db_var.get()
        self._log(f"加载特征库: {db_name}")
        
        # 重置进度条
        def _reset():
            self._progress_bar["value"] = 0
            self._progress_label.config(text="准备加载...")
        self.root.after(0, _reset)
        
        def _run():
            db = get_or_build_database(db_name, force_rebuild=False, 
                                       log_fn=self._log, progress_fn=self._progress_update)
            self._database = db
            self._db_name  = db_name
            cnt = len(db)
            self._log(f"特征库就绪: {cnt} 个角色")
            self._set_status(f"✅ 特征库已加载（{cnt} 角色）")
            
            # 完成时重置进度条
            def _done():
                self._progress_bar["value"] = 100
                self._progress_label.config(text=f"完成！共 {cnt} 个角色")
            self.root.after(0, _done)
        threading.Thread(target=_run, daemon=True).start()

    def _rebuild_db(self):
        db_name = self._db_var.get()
        self._log(f" 强制重建: {db_name}")
        
        # 重置进度条
        def _reset():
            self._progress_bar["value"] = 0
            self._progress_label.config(text="准备重建...")
        self.root.after(0, _reset)
        
        def _run():
            db = get_or_build_database(db_name, force_rebuild=True, 
                                       log_fn=self._log, progress_fn=self._progress_update)
            self._database = db
            self._db_name  = db_name
            self._log(f"✅ 重建完成: {len(db)} 个角色")
            self._set_status(f"✅ 特征库已重建（{len(db)} 角色）")
            
            # 完成时重置进度条
            def _done():
                self._progress_bar["value"] = 100
                self._progress_label.config(text=f"完成！共 {len(db)} 个角色")
            self.root.after(0, _done)
        threading.Thread(target=_run, daemon=True).start()

    # ── 文件处理 ─────────────────────────────────────────────────────────
    def _pick_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 视频", "*.mp4"), ("AVI 视频", "*.avi")]
        )
        if path:
            self._out_var.set(path)

    def _open_file(self):
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
        """接收拖拽文件，兼容含空格路径和多文件"""
        import re
        raw = event.data.strip()
        # tkinterdnd2 对含空格的路径用花括号包裹，多文件间用空格分隔
        # 提取第一个路径（花括号内 或 第一个空格前）
        matches = re.findall(r'\{([^}]+)\}|(\S+)', raw)
        if matches:
            path = (matches[0][0] or matches[0][1]).strip()
        else:
            path = raw
        self._dispatch_file(path)

    def _dispatch_file(self, path: str):
        if not _models_ready:
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
            # 特征库加载在子线程中执行，避免阻塞 GUI 主线程
            db = self._database
            db_name = self._db_name
            if suggested != db_name or not db:
                self._log(f" 自动选择特征库: {suggested}")
                self.root.after(0, lambda: self._db_var.set(suggested))
                db = get_or_build_database(suggested, log_fn=self._log)
                self._database = db
                self._db_name  = suggested

            if not db:
                self._log("❌ 特征库为空，无法识别。请先建库或检查 ./data 文件夹")
                self._set_busy(False)
                return

            mn = self._min_neighbors_var.get()

            if suffix in IMAGE_EXTS:
                self._log(f"\n🖼  图片: {path}")
                process_image_file(
                    source=path,
                    database=db,
                    threshold=self._threshold_var.get(),
                    min_neighbors=mn,
                    log_fn=self._log,
                    preview_fn=self._show_frame_cv,
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

        # 读取现有内容
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
                # 重载别名
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
#  CLI 模式（终端图形化界面）
# ══════════════════════════════════════════════════════════════════════════════

class CLIColors:
    """ANSI 终端颜色"""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    # 前景色
    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    # 亮色
    LGRAY   = "\033[90m"
    LRED    = "\033[91m"
    LGREEN  = "\033[92m"
    LYELLOW = "\033[93m"
    LBLUE   = "\033[94m"
    LMAGENTA= "\033[95m"
    LCYAN   = "\033[96m"
    LWHITE  = "\033[97m"

    # 背景色
    BGBLACK  = "\033[40m"
    BGRED    = "\033[41m"
    BGGREEN  = "\033[42m"
    BGYELLOW = "\033[43m"
    BGBLUE   = "\033[44m"
    BGMAGENTA= "\033[45m"
    BGCYAN   = "\033[46m"
    BGWHITE  = "\033[47m"

    # 合成
    HEADER  = BOLD + LGRAY
    TITLE   = BOLD + MAGENTA
    SUCCESS = BOLD + LGREEN
    WARNING = BOLD + LYELLOW
    ERROR   = BOLD + LRED
    INFO    = BOLD + LCYAN
    MUTED   = DIM + LGRAY
    PROGRESS_BG = BGWHITE
    PROGRESS_FG = BGGREEN

    @staticmethod
    def p(text, color="", end="\n"):
        """打印彩色文本"""
        if color:
            print(f"{color}{text}{CLIColors.RESET}", end=end)
        else:
            print(text, end=end)

    @staticmethod
    def clear_line():
        """清除当前行"""
        sys.stdout.write("\033[2K\r")
        sys.stdout.flush()

    @staticmethod
    def progress_bar(percent: float, width: int = 40,
                     fg_color=None, bg_color=None):
        """生成进度条字符串"""
        filled = int(width * percent)
        empty  = width - filled
        bar = "█" * filled + "░" * empty
        percent_str = f"{percent * 100:5.1f}%"
        fg = fg_color or CLIColors.PROGRESS_FG
        bg = bg_color or CLIColors.PROGRESS_BG
        return f"{fg}{CLIColors.BOLD}{bar}{CLIColors.RESET} {fg}{percent_str}{CLIColors.RESET}"

#═══════════════════
    #                   ║
class MoeFaceCLI:
    """CLI 模式主类"""

    BANNER = rf"""{CLIColors.TITLE}
╔══════════════════════════════════════════════════╗
║                                                  ║
║   ███╗   ███╗  ██████╗ ███████╗                  ║
║   ████╗ ████║ ██╔═══██╗██╔════╝                  ║
║   ██╔████╔██║ ██║   ██║█████╗                    ║
║   ██║╚██╔╝██║ ██║   ██║██╔══╝                    ║
║   ██║ ╚═╝ ██║ ╚██████╔╝███████╗                  ║
║   ╚═╝     ╚═╝  ╚═════╝ ╚══════╝                  ║
║                                                  ║
║            Face Recognition System               ║
║                                                  ║
╚══════════════════════════════════════════════════╝
    {CLIColors.RESET}"""

    def __init__(self, args):
        self.args      = args
        self.db        = {}
        self.stats     = {"total_frames": 0, "processed": 0,
                          "found_names": {}, "errors": 0}
        self.stop_evt  = threading.Event()
        self._console_height = 0
        self._last_pct = -1.0

    # ── 工具方法 ─────────────────────────────────────────────────────────
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
        CLIColors.p(f"\n{CLIColors.HEADER}{'─' * 50}{CLIColors.RESET}", end="")
        CLIColors.p(f"{CLIColors.HEADER} {title}{CLIColors.RESET}")

    def _progress(self, current: int, total: int, elapsed: float = 0, label: str = ""):
        """更新进度条（同一行覆盖）"""
        if total <= 0:
            return
        pct = min(current / total, 1.0)
        bar = CLIColors.progress_bar(pct, width=28)
        label_text = f" {label}" if label else ""
        # 估算剩余时间
        if elapsed > 0 and current > 0:
            eta = elapsed * (total - current) / current
            eta_text = f" 剩余: {int(eta // 60)}分{int(eta % 60)}秒"
        else:
            eta_text = ""
        line = f"\r  {bar}{label_text}{eta_text}"
        CLIColors.clear_line()
        print(line, end="", flush=True)
        self._last_pct = pct

    def _stat_row(self, name: str, count: int, color: str = ""):
        """统计行"""
        fg = color or CLIColors.SUCCESS
        CLIColors.p(f"  {CLIColors.MUTED}├─{CLIColors.RESET} {fg}{name}{CLIColors.RESET}"
                    f" {CLIColors.MUTED}({count} 次){CLIColors.RESET}")

    # ── 主流程 ───────────────────────────────────────────────────────────
    def run(self):
        # 打印 Banner
        print(self.BANNER)

        # 步骤 1: 加载模型
        self._header("① 加载模型")
        CLIColors.p(f"  设备: {CLIColors.INFO}"
                    f"{'NVIDIA GPU (CUDA)' if os.path.exists('/dev/nvidia0') or shutil.which('nvidia-smi') else 'CPU'}"
                    f"{CLIColors.RESET}")
        if not _ensure_models(lambda m: self._log(m, "info" if "✓" in m or "✅" in m else "warn")):
            self._log("模型加载失败，主人～(｡•́︿•̀｡)", "error")
            sys.exit(1)

        # 步骤 2: 加载特征库
        self._header("② 加载特征库")
        db_name = self.args.get("db_name") or DEFAULT_DB_NAME
        CLIColors.p(f"  特征库: {CLIColors.INFO}{db_name}{CLIColors.RESET}")
        force_rebuild = self.args.get("rebuild", False)
        self.db = get_or_build_database(
            db_name, force_rebuild=force_rebuild,
            log_fn=self._log,
            progress_fn=lambda cur, tot, elapsed=0, lbl="": self._progress(cur, tot, elapsed, lbl)
        )
        if not self.db:
            self._log("特征库为空，请检查 ./data 文件夹或添加角色图片", "error")
            sys.exit(1)
        CLIColors.p(f"  已加载 {CLIColors.SUCCESS}{len(self.db)}{CLIColors.RESET} "
                    f"个角色特征{CLIColors.MUTED}，就绪~{CLIColors.RESET}")

        # 步骤 3: 执行识别
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
                self._log(f"输入: {source}")
                self._log(f"输出: {output or '仅预览（无保存）'}")
                self._run_video(source, output)
            else:
                self._log(f"不支持的文件格式: {suffix}", "error")
                sys.exit(1)
        else:
            self._log("请指定 --source（文件路径）或 --camera（摄像头模式）", "error")
            self._print_help()
            sys.exit(1)

        # 步骤 4: 统计报告
        self._print_report()

    def _run_image(self, path: str):
        """识别单张图片"""
        self._log(f"文件: {CLIColors.INFO}{path}{CLIColors.RESET}")
        threshold    = self.args.get("threshold", 0.45)
        min_neighbors = self.args.get("min_neighbors", 3)

        results = []
        import cv2, numpy as np

        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            self._log("无法读取图片喵~", "error")
            return

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _anime_cascade.detectMultiScale(
            gray, scaleFactor=1.02, minNeighbors=min_neighbors,
            minSize=(20, 20), maxSize=(800, 800)
        )
        self._log(f"检测到 {CLIColors.SUCCESS}{len(faces)}{CLIColors.RESET} 张人脸")

        for (x, y, w, h) in faces:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img.shape[1], x+w), min(img.shape[0], y+h)
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            name, score = recognize_face(face_rgb, self.db, threshold)
            tag = CLIColors.SUCCESS if name else CLIColors.MUTED
            self._log(f"  {tag}{name or '未知'}{CLIColors.RESET} "
                      f"{CLIColors.MUTED}({score:.2f}){CLIColors.RESET} "
                      f"@ ({x1},{y1},{w}×{h})", "ok" if name else "skip")
            results.append((name, score))
            if name:
                self.stats["found_names"][name] = \
                    self.stats["found_names"].get(name, 0) + 1

        # 保存结果图
        output = self.args.get("output")
        if output and results:
            draw = img.copy()
            for (x, y, w, h), (name, score) in zip(faces, results):
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(img.shape[1], x+w), min(img.shape[0], y+h)
                cv2.rectangle(draw, (x1,y1),(x2,y2),(0,255,0),2)
                color = (0,255,0) if name else (128,128,128)
                draw = draw_chinese_text(draw,
                    f"{name or '未知'} ({score:.2f})",
                    (x1, max(0,y1-25)), 18, color)
            cv2.imwrite(output, draw)
            self._log(f"已保存标注图: {output}", "ok")
        elif not results:
            self._log("未检测到任何人脸", "warn")

    def _run_video(self, source: str, output_path: str):
        """识别视频文件"""
        threshold    = self.args.get("threshold", 0.45)
        skip_frames  = self.args.get("skip_frames", 2)
        min_neighbors = self.args.get("min_neighbors", 3)

        import cv2, numpy as np

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self._log(f"无法打开视频喵~: {source}", "error")
            return

        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._log(f"视频: {w}×{h}  {fps:.1f}fps  共 {total} 帧")
        CLIColors.p("")  # 换行，开始进度条

        out = tmp_out = None
        if output_path:
            tmp_dir = Path(output_path).parent / "temp"
            tmp_dir.mkdir(exist_ok=True)
            tmp_fd, tmp_out = tempfile.mkstemp(suffix=".mp4", dir=str(tmp_dir))
            os.close(tmp_fd)
            out = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, (w, h))

        frame_idx = 0
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
                    self._progress(frame_idx, total,
                                   f"帧 {frame_idx}/{total}")

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
                        name, score = recognize_face(face_rgb, self.db, threshold)
                        if name:
                            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                            frame = draw_chinese_text(frame,
                                f"{name} ({score:.2f})",
                                (x1, max(0,y1-25)), 16, (0,255,0))
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
        CLIColors.p(f"  {CLIColors.SUCCESS}✓ 处理完成{CLIColors.RESET}："
                    f" {frame_idx} 帧，识别了 {self.stats['processed']} 个关键帧")

        if output_path and tmp_out and os.path.exists(tmp_out):
            self._log("正在合并音频...")
            _add_audio(source, output_path, tmp_out)
            self._log(f"已保存: {output_path}", "ok")

    def _run_camera(self):
        """摄像头实时识别"""
        threshold    = self.args.get("threshold", 0.45)
        min_neighbors = self.args.get("min_neighbors", 3)

        import cv2
        camera_id = self.args.get("camera_id", 0)
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            self._log(f"无法打开摄像头 {camera_id}喵~", "error")
            return

        self._log(f"摄像头 {camera_id} 已启动，按 Q 退出")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay_ms = max(int(1000 / fps), 1)

        try:
            while not self.stop_evt.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue

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
                    name, score = recognize_face(face_rgb, self.db, threshold)
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
        CLIColors.p(f"  总帧数: {CLIColors.INFO}{self.stats['total_frames']}"
                    f"{CLIColors.RESET}，"
                    f"关键帧: {CLIColors.INFO}{self.stats['processed']}"
                    f"{CLIColors.RESET}，"
                    f"识别次数: {CLIColors.SUCCESS}"
                    f"{sum(self.stats['found_names'].values())}"
                    f"{CLIColors.RESET}")
        CLIColors.p(f"  角色分布:")
        sorted_names = sorted(self.stats["found_names"].items(),
                             key=lambda x: x[1], reverse=True)
        for name, count in sorted_names:
            pct = count / sum(self.stats["found_names"].values()) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            CLIColors.p(f"  {CLIColors.SUCCESS}{name:<15}{CLIColors.RESET}"
                        f" {bar} {CLIColors.MUTED}{count} 次 ({pct:4.1f}%)"
                        f"{CLIColors.RESET}")
        CLIColors.p("")
        self._log("识别完成~ 主人再见喵！(｡•́︿•̀｡)", "done")

    def _print_help(self):
        """打印帮助"""
        print(f"""
{CLIColors.TITLE}用法:{CLIColors.RESET}
  {CLIColors.CYAN}python recognize.py --mode cli --source <文件>{CLIColors.RESET}
  {CLIColors.CYAN}python recognize.py --mode gui{CLIColors.RESET}

{CLIColors.TITLE}CLI 参数:{CLIColors.RESET}
  {CLIColors.LGREEN}--mode{CLIColors.RESET}         运行模式: gui / cli (默认: cli)
  {CLIColors.LGREEN}--source{CLIColors.RESET}       视频或图片路径
  {CLIColors.LGREEN}--camera{CLIColors.RESET}       启用摄像头模式
  {CLIColors.LGREEN}--camera-id{CLIColors.RESET}    摄像头 ID (默认: 0)
  {CLIColors.LGREEN}--output{CLIColors.RESET}        输出视频/图片路径
  {CLIColors.LGREEN}--db-name{CLIColors.RESET}       特征库名称 (默认: 全部特征库)
  {CLIColors.LGREEN}--threshold{CLIColors.RESET}     识别阈值 0.1~0.95 (默认: 0.45)
  {CLIColors.LGREEN}--skip-frames{CLIColors.RESET}   视频跳帧数 (默认: 2)
  {CLIColors.LGREEN}--min-neighbors{CLIColors.RESET} 检测灵敏度 1~10 (默认: 3)
  {CLIColors.LGREEN}--rebuild{CLOLors.RESET}         强制重建特征库

{CLIColors.TITLE}示例:{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode cli --source 视频.mp4 --output out.mp4{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode cli --camera --threshold 0.6{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode cli --source 图片.jpg --output annotated.jpg{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode cli --source 视频.mp4 --db-name 永雏塔菲{CLIColors.RESET}
  {CLIColors.MAGENTA}python recognize.py --mode gui{CLIColors.RESET}
""")


def _parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        prog="MoeFace",
        description="MoeFace 动漫人脸识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例: python recognize.py --mode cli --source video.mp4 --output out.mp4"
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["gui", "cli"],
        default="gui",
        help="运行模式: gui=图形界面(默认), cli=终端界面"
    )

    # 文件/流输入
    parser.add_argument(
        "--source", "-s",
        help="视频或图片文件路径（支持 JPG/PNG/MP4/AVI/MKV 等）"
    )
    parser.add_argument(
        "--camera", "-c",
        action="store_true",
        help="启用摄像头模式"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="摄像头设备 ID（默认: 0）"
    )

    # 输出
    parser.add_argument(
        "--output", "-o",
        help="输出视频/图片路径（默认: 仅预览）"
    )
    parser.add_argument(
        "--db-name",
        default=DEFAULT_DB_NAME,
        help="特征库名称（默认: 全部特征库）"
    )

    # 识别参数
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
        "--list", "-l",
        action="store_true",
        help="列出所有可用特征库"
    )

    return parser.parse_args()


# ── 入口 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = _parse_args()

    # 列出特征库
    if args.list:
        print(f"\n{CLIColors.TITLE}可用特征库:{CLIColors.RESET}")
        db_names = sorted({e["db_name"] for e in ALIAS_MAP})
        print(f"  {CLIColors.GREEN}{DEFAULT_DB_NAME}{CLIColors.RESET}（默认）")
        for name in db_names:
            print(f"  {CLIColors.CYAN}{name}{CLIColors.RESET}")
        print("")
        sys.exit(0)

    # 启动对应模式
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
        }
        cli = MoeFaceCLI(cli_args)
        try:
            cli.run()
        except KeyboardInterrupt:
            CLIColors.p(f"\n{CLIColors.WARNING}⏹ 已中断{CLIColors.RESET}")
            cli.stop_evt.set()
            sys.exit(0)
