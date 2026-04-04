"""
MoeFace — 动漫人脸识别系统
GUI 版本 (Tkinter)
"""
import os
import sys
import json
import threading
import warnings
import shutil
import tempfile
import traceback
from pathlib import Path

# ── 确保以脚本所在目录为基准路径 ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

CASCADE_PATH = BASE_DIR / "lbpcascade_animeface.xml"
FONT_PATH    = BASE_DIR / "simhei.ttf"
FEATURES_DIR = BASE_DIR / "features"
DATA_DIR     = BASE_DIR / "data"
CNAME_PATH   = BASE_DIR / "cname" / "name.json"
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
        warnings.warn(f"别名配置文件不存在: {path}")
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        warnings.warn(f"加载别名配置失败: {e}")
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
        warnings.warn(f"加载特征库失败: {e}")
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
                log_fn("❌ 找不到 lbpcascade_animeface.xml")
                return False
            clf = cv2.CascadeClassifier(cascade_p)
            if clf.empty():
                log_fn("❌ CascadeClassifier 加载失败")
                return False
            _anime_cascade = clf

            log_fn("正在加载 FaceNet 模型（首次可能需要几秒）...")
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _resnet = InceptionResnetV1(pretrained="vggface2").eval().to(_device)
            log_fn(f"✅ 模型加载完成（{'GPU' if _device.type == 'cuda' else 'CPU'}）")

            _models_ready = True
            return True
        except Exception as e:
            log_fn(f"❌ 模型初始化失败: {e}")
            traceback.print_exc()
            return False


def extract_features_from_image(image_path: str, log_fn=print):
    import cv2, torch
    import numpy as np
    try:
        with open(image_path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
    except Exception:
        return None

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _anime_cascade.detectMultiScale(
        gray, scaleFactor=1.02, minNeighbors=1,
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


def build_database(data_root: Path, log_fn=print):
    import numpy as np
    database = {}
    if not data_root.exists():
        log_fn(f"❌ 路径不存在: {data_root}")
        return database

    subdirs = [d for d in data_root.iterdir() if d.is_dir()]
    exts    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def _gather(folder):
        files = []
        for ext in exts:
            files += list(folder.glob(f"*{ext}"))
            files += list(folder.glob(f"*{ext.upper()}"))
        return files

    def _process_person(person_dir: Path):
        imgs = _gather(person_dir)
        if not imgs:
            return None
        embs = []
        for i, p in enumerate(imgs):
            log_fn(f"  [{i+1}/{len(imgs)}] {p.name}")
            e = extract_features_from_image(str(p), log_fn)
            if e is not None:
                embs.append(e)
        if embs:
            return np.mean(embs, axis=0)
        return None

    if subdirs:
        log_fn(f"多角色模式，共 {len(subdirs)} 个子文件夹")
        for person_dir in subdirs:
            log_fn(f"\n处理角色: {person_dir.name}")
            avg = _process_person(person_dir)
            if avg is not None:
                database[person_dir.name] = avg
                log_fn(f"  ✅ {person_dir.name}")
            else:
                log_fn(f"  ⚠️  无有效人脸")
    else:
        imgs = _gather(data_root)
        if not imgs:
            log_fn(f"⚠️  {data_root} 下没有图片")
            return database
        log_fn(f"单角色模式: {data_root.name}")
        avg = _process_person(data_root)
        if avg is not None:
            database[data_root.name] = avg

    return database


def get_or_build_database(db_name: str, force_rebuild=False, log_fn=print):
    if not force_rebuild:
        db = load_database_from_json(db_name)
        if db is not None:
            log_fn(f"✅ 从缓存加载特征库 [{db_name}]，共 {len(db)} 个角色")
            return db

    if db_name == DEFAULT_DB_NAME:
        log_fn("构建全部特征库...")
        db = build_database(DATA_DIR, log_fn)
    else:
        db_path = DATA_DIR / db_name
        if not db_path.exists():
            log_fn(f"❌ 文件夹不存在: {db_path}")
            return {}
        log_fn(f"构建特征库: {db_name}")
        db = build_database(db_path, log_fn)

    if db:
        save_database_to_json(db, db_name)
        log_fn(f"✅ 特征库已缓存，共 {len(db)} 个角色")
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
                       log_fn=print, preview_fn=None, done_fn=None):
    """识别单张图片"""
    import cv2
    import numpy as np
    try:
        with open(source, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            log_fn("❌ 无法读取图片")
            if done_fn: done_fn()
            return

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = _anime_cascade.detectMultiScale(
            gray, scaleFactor=1.02, minNeighbors=1,
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
        log_fn(f"❌ 处理图片出错:\n{traceback.format_exc()}")
    finally:
        if done_fn: done_fn()


def process_video_file(source: str, database: dict, output_path=None,
                       threshold=0.45, skip_frames=2,
                       log_fn=print, preview_fn=None,
                       stop_event: threading.Event = None, done_fn=None):
    """处理视频文件，支持逐帧预览"""
    import cv2
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log_fn(f"❌ 无法打开视频: {source}")
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
                    gray, scaleFactor=1.02, minNeighbors=1,
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
        log_fn(f"📁 已保存: {output_path}")

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
        tk.Label(hdr, text="🌸  MoeFace  动漫人脸识别",
                 font=("微软雅黑", 14, "bold"),
                 bg=ACCENT, fg="white").pack(side="left", padx=16, pady=8)

        self._status_lbl = tk.Label(hdr, text="⏳ 正在初始化…",
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

        # — 重建特征库按钮 —
        self._btn(parent, "🔨 重建特征库", self._rebuild_db,
                  ACCENT).pack(fill="x", padx=10, pady=(4, 2))
        self._btn(parent, "📂 加载特征库", self._load_db_now,
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
        tk.Spinbox(parent, from_=1, to=30, textvariable=self._skip_var,
                   bg="#374151", fg=TEXT, buttonbackground="#374151",
                   relief="flat", font=("微软雅黑", 9), width=6
                   ).pack(padx=10, pady=2, anchor="w")

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
        self._btn(parent, "🖼  打开图片/视频", self._open_file,
                  ACCENT).pack(fill="x", padx=10, pady=2)
        self._stop_btn = self._btn(parent, "⏹  停止处理", self._stop_processing,
                                   "#dc2626", state="disabled")
        self._stop_btn.pack(fill="x", padx=10, pady=2)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # — 别名管理入口 —
        self._btn(parent, "⚙  管理角色别名", self._open_alias_editor,
                  "#374151").pack(fill="x", padx=10, pady=2)
        self._btn(parent, "🗑  清空日志", self._clear_log,
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
                text="将图片或视频拖拽到此处（需安装 tkinterdnd2）\n或点击左侧「打开图片/视频」按钮\n\n支持 JPG / PNG / MP4 / AVI / MKV 等"
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
                self._set_status("❌ 模型加载失败，请检查依赖")
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
    def _load_db_now(self):
        db_name = self._db_var.get()
        self._log(f"加载特征库: {db_name}")
        def _run():
            db = get_or_build_database(db_name, force_rebuild=False, log_fn=self._log)
            self._database = db
            self._db_name  = db_name
            cnt = len(db)
            self._log(f"特征库就绪: {cnt} 个角色")
            self._set_status(f"✅ 特征库已加载（{cnt} 角色）")
        threading.Thread(target=_run, daemon=True).start()

    def _rebuild_db(self):
        db_name = self._db_var.get()
        self._log(f"🔨 强制重建: {db_name}")
        def _run():
            db = get_or_build_database(db_name, force_rebuild=True, log_fn=self._log)
            self._database = db
            self._db_name  = db_name
            self._log(f"✅ 重建完成: {len(db)} 个角色")
            self._set_status(f"✅ 特征库已重建（{len(db)} 角色）")
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
        """接收拖拽文件"""
        # tkinterdnd2 返回的路径可能被花括号包裹（路径含空格时）
        raw = event.data.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        # 多文件只取第一个
        path = raw.split("} {")[0].strip("{} ")
        self._dispatch_file(path)

    def _dispatch_file(self, path: str):
        if not _models_ready:
            self._log("⏳ 模型尚未加载完成，请稍候…")
            return
        if self._busy:
            self._log("⚠️  正在处理中，请等待完成或点击停止")
            return

        suffix = Path(path).suffix.lower()
        # 自动根据文件名切换特征库
        suggested = get_db_name_from_filename(Path(path).name)
        if suggested != self._db_name or not self._database:
            self._log(f"🔍 自动选择特征库: {suggested}")
            self._db_var.set(suggested)
            db = get_or_build_database(suggested, log_fn=self._log)
            self._database = db
            self._db_name  = suggested

        if not self._database:
            self._log("❌ 特征库为空，无法识别。请先建库或检查 ./data 文件夹")
            return

        self._set_busy(True)
        self._stop_evt.clear()

        if suffix in IMAGE_EXTS:
            self._log(f"\n🖼  图片: {path}")
            threading.Thread(
                target=process_image_file,
                kwargs=dict(
                    source=path,
                    database=self._database,
                    threshold=self._threshold_var.get(),
                    log_fn=self._log,
                    preview_fn=self._show_frame_cv,
                    done_fn=lambda: self._set_busy(False),
                ),
                daemon=True
            ).start()

        elif suffix in VIDEO_EXTS:
            self._log(f"\n🎬  视频: {path}")
            out = self._out_var.get().strip() or None
            threading.Thread(
                target=process_video_file,
                kwargs=dict(
                    source=path,
                    database=self._database,
                    output_path=out,
                    threshold=self._threshold_var.get(),
                    skip_frames=self._skip_var.get(),
                    log_fn=self._log,
                    preview_fn=self._show_frame_cv,
                    stop_event=self._stop_evt,
                    done_fn=lambda: self._set_busy(False),
                ),
                daemon=True
            ).start()
        else:
            self._log(f"⚠️  不支持的文件格式: {suffix}")
            self._set_busy(False)

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


# ── 入口 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = MoeFaceApp()
    app.run()
