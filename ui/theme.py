"""
MoeFace v3.2 设计系统
配色方案：VTuber 科技风（萌系 + AI + 未来感）
"""

import customtkinter as ctk


# ── 配色 ──────────────────────────────────────────────────────────────────
class Colors:
    """MoeFace v3.2 调色板"""
    # 背景
    BG = "#12131A"              # 主背景（深黑蓝）
    CARD = "#1E2130"            # 卡片/面板
    CARD_HOVER = "#252838"      # 卡片悬浮

    # 主色 & 辅助色
    PRIMARY = "#B47CFF"         # 柔和紫（主操作按钮）
    PRIMARY_HOVER = "#9B5FE6"   # 紫色深色
    ACCENT = "#62D8FF"          # 赛博蓝（AI 状态/高亮）
    ACCENT_HOVER = "#4AC0E6"    # 蓝色深色

    # 状态色
    SUCCESS = "#6EE7B7"         # 成功（识别高分）
    WARNING = "#FFD166"         # 警告（中分）
    ERROR = "#FF6B8A"           # 错误

    # 文字
    TEXT = "#E0E0FF"            # 主文字（浅紫白）
    TEXT_MUTED = "#A8A8C8"      # 副文字（灰紫）
    TEXT_DIM = "#6C6C8A"        # 更暗文字

    # 杂项
    BORDER = "#2A2D3E"          # 边框
    DIVIDER = "#2A2D3E"         # 分割线
    CANVAS_BG = "#0F0F1A"       # 预览区背景
    LOG_BG = "#0B0B14"          # 日志背景
    DROP_BORDER = "#62D8FF"     # 拖拽区边框


# ── 间距 ──────────────────────────────────────────────────────────────────
SPACING = {
    "xs":  4,
    "sm":  8,
    "md":  12,
    "lg":  16,
    "xl":  24,
    "card_pad": (14, 12),
    "section_gap": 10,
}


# ── 字体 ──────────────────────────────────────────────────────────────────
FONTS = {
    "heading":      ("Microsoft YaHei UI", 16, "bold"),
    "subheading":   ("Microsoft YaHei UI", 10),
    "body":         ("Microsoft YaHei UI", 10),
    "small":        ("Microsoft YaHei UI", 8),
    "mono":         ("Consolas", 9),
    "title":        ("Microsoft YaHei UI", 14, "bold"),
    "card_title":   ("Microsoft YaHei UI", 9, "bold"),
}


# ── 主题应用 ──────────────────────────────────────────────
def apply_theme():
    """应用 MoeFace 全局主题（仅设暗色模式，颜色由各控件显式传入）"""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
