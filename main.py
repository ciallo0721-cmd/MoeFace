#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MoeFace 动漫人脸识别系统 — 兼容入口

等效于 python app.py [参数...]
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

if __name__ == "__main__":
    from app import main
    main()
