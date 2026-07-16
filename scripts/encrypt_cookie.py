#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B站 Cookie 加密工具 - 本地运行，加密后把结果放到 txt 文件中"""
import base64
import sys

# ⚠️ 这个 KEY 和 fetch_vtuber_news.py 里的必须一致！
_KEY = b"MoeFace_2024_Bili_Cookie_Key_X"


def encrypt(raw: str) -> str:
    """三层加密: base64 → XOR → hex → base64（外层防爬虫）"""
    # 第一层：Base64
    _b64 = base64.b64encode(raw.encode("utf-8")).decode("utf-8")
    # 第二层：XOR 混淆
    _kb = _KEY
    _xor = bytes(ord(_b64[i]) ^ _kb[i % len(_kb)] for i in range(len(_b64)))
    # 第三层：hex → 外层 base64（防简单爬虫直接读 hex）
    return base64.b64encode(_xor.hex().encode()).decode()


if __name__ == "__main__":
    print("=" * 50)
    print("  B站 Cookie 加密工具")
    print("  把加密结果放到 ciallo0721-cmd.top 的")
    print("  /css/css/1/2/3/4/5/6/7/8.txt")
    print("=" * 50)
    cookie = input("\n请粘贴 B站 Cookie: ").strip()
    if not cookie:
        print("❌ Cookie 为空！")
        sys.exit(1)
    result = encrypt(cookie)
    print(f"\n✅ 加密完成！复制下面这行放到 txt 文件里：")
    print("-" * 50)
    print(result)
    print("-" * 50)
    print(f"\n长度: {len(cookie)} → {len(result)} 字符")
