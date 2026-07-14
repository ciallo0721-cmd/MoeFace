"""
MoeFace EXE 构建脚本
用法: python build_exe.py
输出: dist/MoeFace/ 目录（onedir 模式）
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# ── 项目路径 ──
PROJ_DIR = Path(__file__).parent.resolve()
DIST_DIR = PROJ_DIR / "dist"
BUILD_DIR = PROJ_DIR / "build"
SPEC_FILE = PROJ_DIR / "MoeFace.spec"
VENV_PYTHON = PROJ_DIR / ".venv" / "Scripts" / "python.exe"

# ── 资源文件列表（需要打包进 EXE 目录的） ──
RESOURCE_FILES = [
    # 核心识别资源
    "lbpcascade_animeface.xml",
    "simhei.ttf",
    "cname/name.json",

    # 姿态检测（用于 .moe 特征提取）
    "pose_landmarker.task",
    "pose_landmarker_lite.task",

    # Python 包
    "modules/",
    "ui/",
]

# ── 需要排除的大文件（普通模式不需要） ──
EXCLUDE_PATTERNS = [
    "data/",              # 训练图片（4.6GB）
    "models/",            # Whisper 语音模型
    "emotion-ferplus.onnx",
    "nsfw_mobilenet2.224x224.h5",
    "yolo11n-pose.onnx",
]

# ── 特征库（.moe 文件） ──
FEATURES_DIR = PROJ_DIR / "features"


def get_pyinstaller_cmd():
    """选择正确的 PyInstaller"""
    if VENV_PYTHON.exists():
        return [str(VENV_PYTHON), "-m", "PyInstaller"]
    return ["pyinstaller"]


def ensure_temp_dir():
    """确保临时目录在 G 盘（C 盘已满）"""
    tmp = PROJ_DIR / ".venv" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def build():
    tmp_dir = ensure_temp_dir()

    # 确保 features 目录存在
    if not FEATURES_DIR.exists() or not any(FEATURES_DIR.iterdir()):
        print("⚠️  警告: features/ 为空，请先运行特征提取喵！")

    # 准备 PyInstaller 参数
    pyi = get_pyinstaller_cmd()

    # ── 隐藏导入（PyTorch 等动态加载的模块） ──
    hidden_imports = [
        "torch",
        "torch._C",
        "torch.storage",
        "torch.nn",
        "torch.nn.modules",
        "torch.nn.functional",
        "torch.utils.data",
        "torchvision",
        "facenet_pytorch",
        "PIL",
        "PIL.Image",
        "PIL.ImageDraw",
        "PIL.ImageFont",
        "cv2",
        "numpy",
        "mediapipe",
        "mediapipe.python",
        "mediapipe.python.solutions",
        "mediapipe.tasks.python",
        "mediapipe.tasks.python.vision",
        "mediapipe.tasks.python.vision.pose_landmarker",
        "google.protobuf",
        "google.protobuf.internal",
        "flask",
        "customtkinter",
        "requests",
        "bs4",
        "moviepy",
        "matplotlib",
        "matplotlib.backends",
        "matplotlib.backends.backend_agg",
        "PIL._tkinter_finder",
        "tkinter",
        "tkinter.filedialog",
        "tkinter.messagebox",
        "tkinter.scrolledtext",
        "tkinter.ttk",
        "json",
        "pathlib",
        "threading",
        "warnings",
        "tempfile",
        "shutil",
        "traceback",
        "datetime",
        "typing",
        "typing_extensions",
        "collections",
        "functools",
        "itertools",
        "re",
        "math",
        "hashlib",
        "struct",
        "io",
        "base64",
        "uuid",
        "html",
        "urllib",
        "http",
        "socketserver",
        "smtplib",
        "email",
        "email.mime",
        "email.mime.multipart",
        "email.mime.text",
        "email.mime.base",
        "webbrowser",
        "subprocess",
        "csv",
        "textwrap",
        "dataclasses",
        "enum",
        "inspect",
        "copy",
        "gc",
        "logging",
        "os",
        "sys",
        "time",
        "ctypes",
        "ssl",
        "socket",
        "queue",
        "pickle",
        "configparser",
        "argparse",
        "platform",
        "random",
        "string",
    ]

    # ── 资源数据路径（相对于项目根目录） ──
    datas = [
        ("lbpcascade_animeface.xml", "."),
        ("simhei.ttf", "."),
        ("cname/name.json", "cname"),
        ("features", "features"),
        ("pose_landmarker.task", "."),
        ("pose_landmarker_lite.task", "."),
        ("modules", "modules"),
        ("ui", "ui"),
    ]

    # ── 构建命令 ──
    cmd = pyi + [
        str(PROJ_DIR / "recognize.py"),
        "--name", "MoeFace",
        "--onedir",
        "--noconfirm",
        "--clean",
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
        "--specpath", str(PROJ_DIR),
    ]

    # 添加隐藏导入
    for mod in hidden_imports:
        cmd += ["--hidden-import", mod]

    # 添加资源文件
    for src, dst in datas:
        cmd += ["--add-data", f"{src}{os.pathsep}{dst}"]

    # 添加 PyTorch 相关收集
    cmd += [
        "--collect-all", "torch",
        "--collect-all", "torchvision",
        "--collect-all", "facenet_pytorch",
        "--collect-all", "mediapipe",
        "--collect-all", "google.protobuf",
        "--collect-submodules", "PIL",
    ]

    # 排除不需要的大模块
    cmd += [
        "--exclude-module", "torch.distributed",
        "--exclude-module", "torch.testing",
        "--exclude-module", "torch.utils.tensorboard",
        "--exclude-module", "torch.jit",
        "--exclude-module", "torch.onnx",
        "--exclude-module", "torch.backends.cudnn",
        "--exclude-module", "torch.backends.mps",
        "--exclude-module", "torchvision.datasets",
        "--exclude-module", "torchvision.io",
        "--exclude-module", "torchvision.ops",
        "--exclude-module", "torchvision.models",
        "--exclude-module", "torchvision.transforms",
        "--exclude-module", "matplotlib.tests",
        "--exclude-module", "PIL.ImageQt",
        "--exclude-module", "PIL.TiffImagePlugin",
        "--exclude-module", "PIL.FpxImagePlugin",
        "--exclude-module", "PIL.MicImagePlugin",
        "--exclude-module", "PIL.MpegImagePlugin",
        "--exclude-module", "PIL.PcdImagePlugin",
        "--exclude-module", "PIL.PcfFontFile",
        "--exclude-module", "PIL.PixarImagePlugin",
        "--exclude-module", "PIL.SgiImagePlugin",
        "--exclude-module", "PIL.TgaImagePlugin",
        "--exclude-module", "PIL.XbmImagePlugin",
        "--exclude-module", "PIL.XpmImagePlugin",
        "--exclude-module", "tests",
        "--exclude-module", "test",
    ]

    # ── 执行构建 ──
    print("=" * 60)
    print("  MoeFace EXE 构建")
    print(f"  Python: {VENV_PYTHON}")
    print(f"  输出: {DIST_DIR / 'MoeFace/'}")
    print("=" * 60)
    print()

    cache_dir = PROJ_DIR / ".venv" / "pyinstaller-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TMP"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["PYINSTALLER_CONFIG_DIR"] = str(cache_dir)
    env["PYTHONHASHSEED"] = "42"

    print("🚀 开始 PyInstaller 构建（这可能需要 10-30 分钟喵）...")
    print()

    result = subprocess.run(cmd, env=env, cwd=str(PROJ_DIR))

    if result.returncode == 0:
        print()
        print("✅ 构建成功！")
        print(f"  EXE 位置: {DIST_DIR / 'MoeFace' / 'MoeFace.exe'}")
        print()

        # 复制特征库
        features_dst = DIST_DIR / "MoeFace" / "features"
        if FEATURES_DIR.exists():
            if not features_dst.exists():
                print("📦 复制特征库到输出目录...")
                shutil.copytree(str(FEATURES_DIR), str(features_dst),
                                ignore=shutil.ignore_patterns('__pycache__'))

        # 检查输出大小
        total_size = sum(f.stat().st_size for f in Path(DIST_DIR / "MoeFace").rglob('*')
                        if f.is_file())
        print(f"📊 输出目录大小: {total_size / 1024**3:.1f} GB")
        print()
        print("💡 提示: 需要分发时，把 dist/MoeFace/ 整个目录打包即可喵～")
        print("   也可以运行: dist/MoeFace/MoeFace.exe 来启动")
    else:
        print()
        print(f"❌ 构建失败 (exit code: {result.returncode})")
        print("   检查 build/ 目录下的 PyInstaller 日志喵")
        sys.exit(1)


if __name__ == "__main__":
    build()
