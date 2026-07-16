#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
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
║                   统一启动入口 · v3.3.0                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

MoeFace 动漫人脸识别系统 — 统一启动入口

用法:
    python app.py                         启动 Tkinter GUI 界面（默认）
    python app.py --mode gui              启动 Tkinter GUI 界面
    python app.py --mode cli [参数...]     终端 CLI 模式
    python app.py --mode web              启动 Flask Web 服务
    python app.py --mode api               启动 REST API 服务
    python app.py --mode batch <目录>      批量处理目录中的图片/视频
    python app.py --list                   列出可用特征库
    python app.py --version                显示版本信息
"""

import os
import sys
import argparse
from pathlib import Path


# ── 确保以项目目录为工作目录 ──
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

VERSION = "3.3.0"

BANNER = rf"""
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
║                   统一启动入口 · v{VERSION}                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


def print_banner():
    """打印启动 Banner"""
    # 检查终端是否支持颜色
    if sys.platform == "win32":
        os.system("")  # 启用 Windows 终端 ANSI 支持
    print(BANNER)


def start_gui(args):
    """启动 GUI 模式（默认）"""
    from recognize import MoeFaceApp
    app = MoeFaceApp()
    app.run()


def start_cli(args):
    """启动 CLI 模式"""
    from recognize import MoeFaceCLI, DEFAULT_DB_NAME

    cli_args = {
        "source":        args.source,
        "output":        args.output,
        "db_name":       args.db_name or DEFAULT_DB_NAME,
        "threshold":     args.threshold,
        "skip_frames":   args.skip_frames,
        "min_neighbors": args.min_neighbors,
        "rebuild":       args.rebuild,
        "camera":        args.camera,
        "camera_id":     args.camera_id,
        "body":          args.body,
        "emotion":       args.emotion,
        "speech":        args.speech,
        "nsfw":          args.nsfw,
    }

    cli = MoeFaceCLI(cli_args)
    try:
        cli.run()
    except KeyboardInterrupt:
        from recognize import CLIColors
        CLIColors.p(f"\n  {CLIColors.WARNING}⏹ 已中断{CLIColors.RESET}")
        cli.stop_evt.set()
        sys.exit(0)


def start_web(args):
    """启动 Web 模式（基于 api_server 的 Flask 实例）"""
    from api_server import app, init_api

    db_name = args.db_name or "全部特征库"
    host = args.host or "127.0.0.1"
    port = args.port or 5000

    if not init_api(db_name, log_fn=print):
        print("❌ Web 服务初始化失败")
        sys.exit(1)

    print(f"\n🌐 Web 界面: http://{host}:{port}")
    print(f"   GET  /api/health               — 健康检查")
    print(f"   GET  /api/roles                — 角色列表")
    print(f"   POST /api/recognize            — 识别图片")
    print(f"   POST /api/recognize/video      — 视频识别")
    print(f"   GET  /api/tasks/{{id}}         — 查询任务状态")
    print("   按 Ctrl+C 停止服务\n")

    try:
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n⏹ Web 服务已停止")


def start_api(args):
    """启动 API 服务模式"""
    from api_server import start_api_server, init_api

    db_name = args.db_name or "全部特征库"
    host = args.host or "0.0.0.0"
    port = args.port or 5000

    if not init_api(db_name, log_fn=print):
        print("❌ API 初始化失败")
        sys.exit(1)

    print(f"\n🌐 API 服务启动: http://{host}:{port}")
    print(f"   GET  /api/health       — 健康检查")
    print(f"   GET  /api/roles        — 角色列表")
    print(f"   POST /api/recognize    — 识别图片")
    print(f"   POST /api/recognize/video — 视频识别（异步）")
    print(f"   GET  /api/tasks/{{id}} — 查询任务状态")
    print("   按 Ctrl+C 停止服务\n")

    try:
        from api_server import app
        app.run(host=host, port=port, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n⏹ API 服务已停止")


def start_batch(args):
    """批量处理模式"""
    from recognize import (
        MoeFaceCLI, CLIColors, DEFAULT_DB_NAME, get_or_build_database,
        get_or_build_negative_database, _ensure_models,
        process_image_file, process_video_file, IMAGE_EXTS, VIDEO_EXTS,
    )

    target = args.batch
    if not target or not os.path.exists(target):
        print("❌ 请指定有效的目录路径")
        sys.exit(1)

    target_path = Path(target)
    if not target_path.is_dir():
        print("❌ 批量模式需要一个目录，而不是文件")
        sys.exit(1)

    print_banner()
    print(f"📦 批量处理模式")
    print(f"   目录: {target_path.resolve()}")
    print()

    # 收集文件
    files = []
    for ext in IMAGE_EXTS | VIDEO_EXTS:
        for f in target_path.rglob(f"*{ext}"):
            files.append(str(f))
        for f in target_path.rglob(f"*{ext.upper()}"):
            if str(f) not in files:
                files.append(str(f))

    if not files:
        print("❌ 目录中没有找到图片或视频文件")
        return

    print(f"📊 找到 {len(files)} 个文件")
    print()

    # 加载模型
    print("🔧 加载模型...")
    if not _ensure_models(print):
        print("❌ 模型加载失败")
        sys.exit(1)

    # 加载特征库
    db_name = args.db_name or DEFAULT_DB_NAME
    print(f"📚 加载特征库: {db_name}")
    db, _ = get_or_build_database(db_name, log_fn=print)
    ndb = get_or_build_negative_database(log_fn=print)

    if not db:
        print("❌ 特征库为空，无法识别")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"开始批量处理 {len(files)} 个文件...")
    print(f"{'='*60}\n")

    success = 0
    failed = 0

    for i, filepath in enumerate(files, 1):
        suffix = Path(filepath).suffix.lower()
        name = Path(filepath).name
        print(f"\n[{i}/{len(files)}] {name}")

        try:
            if suffix in IMAGE_EXTS:
                process_image_file(
                    source=filepath, database=db, threshold=args.threshold,
                    log_fn=lambda m: print(f"  {m}"), negative_db=ndb,
                )
                success += 1
            elif suffix in VIDEO_EXTS:
                output = str(Path(filepath).parent / f"{Path(filepath).stem}_annotated.mp4")
                process_video_file(
                    source=filepath, database=db, output_path=output,
                    threshold=args.threshold, skip_frames=args.skip_frames,
                    log_fn=lambda m: print(f"  {m}"), negative_db=ndb,
                )
                success += 1
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"✅ 批量处理完成: 成功 {success}, 失败 {failed}")
    print(f"{'='*60}")


def list_databases():
    """列出可用特征库"""
    from recognize import CLIColors, scan_role_folders, DEFAULT_DB_NAME
    print(f"\n  {CLIColors.TITLE}可用特征库:{CLIColors.RESET}")
    print(f"    {CLIColors.GREEN}{DEFAULT_DB_NAME}{CLIColors.RESET}（默认）")
    for name in scan_role_folders():
        print(f"    {CLIColors.CYAN}{name}{CLIColors.RESET}")
    print("")


def main():
    parser = argparse.ArgumentParser(
        prog="MoeFace",
        description="MoeFace 动漫人脸识别系统 — 统一启动入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python app.py                             启动 GUI 界面
  python app.py --mode cli --source 视频.mp4 --output out.mp4
  python app.py --mode web --port 8080
  python app.py --mode api
  python app.py --mode batch ./data/测试集
  python app.py --list
        """,
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["gui", "cli", "web", "api", "batch"],
        default="gui",
        help="运行模式: gui(默认) / cli / web / api / batch",
    )

    # ── 通用参数 ──
    parser.add_argument("--source", "-s", help="输入文件路径（CLI 模式）")
    parser.add_argument("--output", "-o", help="输出文件路径（CLI 模式）")
    parser.add_argument("--db-name", help="特征库名称（默认: 全部特征库）")
    parser.add_argument("--threshold", "-t", type=float, default=0.45, help="识别阈值 (默认: 0.45)")
    parser.add_argument("--skip-frames", "-k", type=int, default=2, help="视频跳帧数 (默认: 2)")
    parser.add_argument("--min-neighbors", type=int, default=5, help="检测灵敏度 (默认: 5)")
    parser.add_argument("--rebuild", "-r", action="store_true", help="强制重建特征库")

    # ── 人体姿态 ──
    parser.add_argument("--body", action="store_true", help="启用人体姿态检测")

    # ── AI 增强 ──
    parser.add_argument("--emotion", action="store_true", help="启用情绪识别")
    parser.add_argument("--speech", action="store_true", help="启用语音转文字")
    parser.add_argument("--nsfw", action="store_true", help="启用 NSFW 检测")

    # ── 摄像头 ──
    parser.add_argument("--camera", action="store_true", help="启用摄像头模式")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头 ID (默认: 0)")

    # ── 网络服务 ──
    parser.add_argument("--host", default="127.0.0.1", help="Web/API 服务监听地址 (默认: 127.0.0.1)")
    parser.add_argument("--port", "-p", type=int, default=5000, help="Web/API 服务端口 (默认: 5000)")

    # ── 批量处理 ──
    parser.add_argument("--batch", help="批量处理目录路径")

    # ── 其他 ──
    parser.add_argument("--list", "-l", action="store_true", help="列出可用特征库")
    parser.add_argument("--version", "-v", action="store_true", help="显示版本信息")

    args = parser.parse_args()

    # 优先处理 --list 和 --version
    if args.version:
        print(f"MoeFace v{VERSION}")
        return

    if args.list:
        print_banner()
        list_databases()
        return

    # 按模式启动
    mode_handlers = {
        "gui":   start_gui,
        "cli":   start_cli,
        "web":   start_web,
        "api":   start_api,
        "batch": start_batch,
    }

    handler = mode_handlers.get(args.mode)
    if handler:
        if args.mode != "gui":
            print_banner()
        handler(args)
    else:
        print(f"❌ 未知模式: {args.mode}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
