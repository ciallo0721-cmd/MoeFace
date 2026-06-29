"""
MoeFace AI 识别模块包
====================
提供三项独立可组合的 AI 识别功能：
1. emotion  — 表情与情绪识别
2. speech   — 角色级语音转文字（说话人分离 + ASR）
3. nsfw     — 多模态 NSFW 内容识别

所有模块均遵循统一的结果输出格式（JSON 兼容 dict）。
"""

from .emotion import EmotionRecognizer
from .speech import SpeechRecognizer
from .nsfw import NSFWDetector

__all__ = [
    "EmotionRecognizer",
    "SpeechRecognizer",
    "NSFWDetector",
]
