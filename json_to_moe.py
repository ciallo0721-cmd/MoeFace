"""
json_to_moe.py - 将旧版 .json 特征库批量转换为自研 .moe 二进制格式
MoeFace 动漫人脸识别项目：角色特征库转换工具
支持 VTuber/二次元角色特征向量 (.json) → 紧凑 .moe 格式
用法：python json_to_moe.py
"""

import os
import sys
import json
import struct
import warnings
from pathlib import Path

import numpy as np

FEATURES_DIR = Path(__file__).resolve().parent / "features"


def _safe_filename(name: str) -> str:
    keep = set("._- ·•/")
    return "".join(c for c in name if c.isalnum() or c in keep or "\u4e00" <= c <= "\u9fff")


def json_to_moe(json_path: Path) -> bool:
    """
    将单个 .json 特征库文件转换为 .moe 格式
    .moe 格式说明：
      0x00-0x02  魔数 "MOE" (3 bytes)
      0x03       版本号 (1 byte, =1)
      0x04-0x07  角色数量 (uint32, little-endian)
      接着每个角色：
        名称长度 (uint32) + 名称 (UTF-8) + 特征向量 (512×float32 = 2048 bytes)
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
        with open(moe_path, "wb") as f:
            # 魔数 + 版本
            f.write(b"MOE")
            f.write(bytes([1]))
            # 角色数量
            f.write(struct.pack("<I", len(raw)))
            for name, emb in raw.items():
                # 角色名
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                # 特征向量
                vec = np.array(emb, dtype=np.float32)
                if vec.shape[0] != 512:
                    print(f"  [WARN]  {name} 特征维度异常: {vec.shape}，跳过")
                    continue
                f.write(vec.tobytes())
        return True
    except Exception as e:
        print(f"  [ERROR] 写入 .moe 失败: {moe_path.name} → {e}")
        if moe_path.exists():
            moe_path.unlink()
        return False


def verify_moe(moe_path: Path) -> bool:
    """验证 .moe 文件是否可正常读取"""
    try:
        with open(moe_path, "rb") as f:
            magic = f.read(3)
            if magic != b"MOE":
                return False
            version = ord(f.read(1))
            if version != 1:
                return False
            num = struct.unpack("<I", f.read(4))[0]
            for _ in range(num):
                name_len = struct.unpack("<I", f.read(4))[0]
                f.read(name_len)
                vec_bytes = f.read(512 * 4)
                if len(vec_bytes) != 512 * 4:
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
            print(f"          体积: {size_json/1024:.1f} KB → {size_moe/1024:.1f} KB  (减少 {ratio:.1f}%)")
            ok += 1
        else:
            print(f"  [FAIL]  转换或验证失败: {json_path.name}")

    print(f"\n✅ 完成！共 {total} 个文件，成功转换 {ok} 个，跳过 {skipped} 个")


if __name__ == "__main__":
    main()
