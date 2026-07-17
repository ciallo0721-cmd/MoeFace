"""
clean_hanime.py — 扫描 data/test/hanime 目录，
用项目的 NSFWDetector（规则分析+TF 模型）检测 NSFW 图片并删除。

判断条件（颜色规则分析）:
  - 肉色+白色+粉色 >= 80%  → NSFW
  - 肉色 >= 90%            → NSFW
  - 粉色 >= 70%            → NSFW
"""

import os
import sys
import time
import cv2
import numpy as np

# 把项目根目录加入 sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

TARGET_DIR = os.path.join(BASE_DIR, "data", "test", "hanime")
THRESHOLD = 0.70  # 新规则下安全分(≤0.60)与NSFW分(≥0.80)之间有明确分界

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def init_detector():
    """初始化 NSFWDetector，如果 TF 或 ONNX 模型不可用则降级到规则分析"""
    from modules.nsfw import NSFWDetector
    detector = NSFWDetector()
    detector.ensure_initialized(log_fn=log)
    return detector


def main():
    if not os.path.isdir(TARGET_DIR):
        log(f"❌ 目录不存在: {TARGET_DIR}")
        sys.exit(1)

    # 收集图片
    images = sorted([f for f in os.listdir(TARGET_DIR)
                     if f.lower().endswith(IMAGE_EXTS)])
    if not images:
        log("📂 目录中没有图片")
        return

    log(f"📂 共找到 {len(images)} 张图片")
    log(f"🔞 阈值: NSFW >= {THRESHOLD:.0%}")
    log("")

    # 初始化检测器
    log("🔄 加载 NSFW 检测器...")
    detector = init_detector()
    log("✅ 检测器就绪")
    log("")

    deleted = 0
    kept = 0
    skipped = 0

    for idx, fname in enumerate(images, 1):
        path = os.path.join(TARGET_DIR, fname)
        try:
            img = cv2.imread(path)
            if img is None:
                skipped += 1
                log(f"  ⚠️ [{idx}/{len(images)}] {fname}  无法读取")
                continue

            score, label = detector._detect_visual_nsfw(img)

            if score >= THRESHOLD:
                os.remove(path)
                deleted += 1
                log(f"  🗑️ [{idx}/{len(images)}] {fname}  score={score:.2f} ({label})  → 已删除")
            else:
                kept += 1
                log(f"  ✅ [{idx}/{len(images)}] {fname}  score={score:.2f} ({label})  → 保留")

        except Exception as e:
            skipped += 1
            log(f"  ⚠️ [{idx}/{len(images)}] {fname}  检测出错: {e}")

    log("")
    log(f"{'='*40}")
    log(f"📊 汇总:")
    log(f"   总计: {len(images)} 张")
    log(f"   删除: {deleted} 张")
    log(f"   保留: {kept} 张")
    log(f"   跳过: {skipped} 张")
    log(f"{'='*40}")


if __name__ == "__main__":
    main()
