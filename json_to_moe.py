"""
json_to_moe.py - 将旧版 .json 特征库批量转换为新 .moe 文本格式
MoeFace 动漫人脸识别项目：角色特征库转换工具
支持 VTuber/二次元角色特征向量 (.json) → 文本 .moe 格式
用法：python json_to_moe.py

新 .moe 文本格式：
("角色名"{key1:浮点,浮点,...:key2:浮点,...:keyN:浮点,...:}"角色名2"{...})

每个角色存储 11 个部位特征：
  面部：eye, eye2, nose, mouth, head
  肢体：arm, arm2, hand, hand2, leg, leg2
旧 .json 只有 1 个面部特征，转换时复制到所有 11 个键
"""

import os
import sys
import json
from pathlib import Path

import numpy as np

FEATURES_DIR = Path(__file__).resolve().parent / "features"

FEATURE_KEYS = ["eye", "eye2", "nose", "mouth", "head",
                "arm", "arm2", "hand", "hand2", "leg", "leg2"]


def _safe_filename(name: str) -> str:
    keep = set("._- ·•/")
    return "".join(c for c in name if c.isalnum() or c in keep or "\u4e00" <= c <= "\u9fff")


def json_to_moe(json_path: Path) -> bool:
    """
    将单个 .json 特征库文件转换为 .moe 文本格式
    旧 .json = {"角色名": [512 个浮点数]}
    新 .moe = ("角色名"{eye:0.1,0.2,...:eye2:0.1,0.2,...:...leg2:...:})
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"  [ERROR] 读取 JSON 失败: {json_path.name} → {e}")
        return False

    if not isinstance(raw, dict) or len(raw) == 0:
        print(f"  [SKIP]  空文件或格式不对: {json_path.name}")
        return False

    moe_path = json_path.with_suffix(".moe")

    try:
        chunks = []
        for name, emb in raw.items():
            # 旧 .json 只有 1 个面部特征向量，复制到所有键
            vec = np.array(emb, dtype=np.float32)
            if vec.ndim == 0 or vec.shape[0] != 512:
                print(f"  [WARN]  {name} 特征维度异常: {vec.shape}，跳过")
                continue

            content_parts = []
            for key in FEATURE_KEYS:
                vec_str = ",".join(f"{v:.10f}" for v in vec)
                content_parts.append(f"{key}:{vec_str}")
            content = ":".join(content_parts)
            chunks.append(f'"{name}"{{{content}:}}')

        all_text = "(" + "".join(chunks) + ")"
        moe_path.write_text(all_text, encoding="utf-8")
        return True

    except Exception as e:
        print(f"  [ERROR] 写入 .moe 失败: {moe_path.name} → {e}")
        if moe_path.exists():
            moe_path.unlink()
        return False


def verify_moe(moe_path: Path) -> bool:
    """验证 .moe 文本文件是否可正常读取"""
    try:
        text = moe_path.read_text(encoding="utf-8").strip()
        if not text.startswith("(") or not text.endswith(")"):
            return False
        inner = text[1:-1]
        parts = inner.split('"')
        i = 1
        while i + 1 < len(parts):
            name = parts[i]
            content_block = parts[i + 1]
            i += 2
            if not content_block.startswith("{") or not content_block.endswith("}"):
                return False
            content = content_block[1:-1]
            if not content:
                continue
            # 验证每个键都有值
            kv_pairs = content.split(":")
            if len(kv_pairs) < 2:
                return False
        return True
    except Exception:
        return False


def main():
    print(f"📂 扫描特征库目录: {FEATURES_DIR}")

    json_files = list(FEATURES_DIR.glob("*.json"))
    if not json_files:
        print("没有找到任何 .json 文件喵～")
        return

    total = len(json_files)
    ok = 0
    skipped = 0

    for json_path in json_files:
        moe_path = json_path.with_suffix(".moe")
        print(f"\n→ {json_path.name}")

        # 已存在且可验证就跳过
        if moe_path.exists() and verify_moe(moe_path):
            print(f"  [SKIP]  .moe 已存在且验证通过，跳过")
            skipped += 1
            continue

        success = json_to_moe(json_path)
        if success and verify_moe(moe_path):
            size_json = json_path.stat().st_size
            size_moe  = moe_path.stat().st_size
            ratio = (1 - size_moe / size_json) * 100 if size_json else 0
            print(f"  [OK]    {json_path.name} → {moe_path.name}")
            print(f"          体积: {size_json/1024:.1f} KB → {size_moe/1024:.1f} KB  (变化 {ratio:+.1f}%)")
            ok += 1
        else:
            print(f"  [FAIL]  转换或验证失败: {json_path.name}")

    print(f"\n✅ 完成！共 {total} 个文件，成功转换 {ok} 个，跳过 {skipped} 个")


if __name__ == "__main__":
    main()
