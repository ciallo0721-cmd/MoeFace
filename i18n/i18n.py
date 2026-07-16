"""
MoeFace i18n 多语言支持
"""
import os, json
from pathlib import Path

_I18N_DIR = Path(__file__).resolve().parent

class I18n:
    """多语言支持单例"""
    _instance = None
    _strings = {}
    _current_lang = "zh-CN"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load("zh-CN")
        return cls._instance

    def _load(self, lang: str):
        """加载语言包"""
        path = _I18N_DIR / f"{lang}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self._strings = json.load(f)
            self._current_lang = lang
        else:
            self._strings = {}
            self._current_lang = lang

    def set_language(self, lang: str):
        """切换语言"""
        self._load(lang)

    def get(self, key: str, default: str = None) -> str:
        """获取翻译文本"""
        if default is None:
            default = key
        return self._strings.get(key, default)

    @property
    def current_language(self) -> str:
        return self._current_lang

    @staticmethod
    def available_languages() -> list:
        """返回可用语言列表"""
        langs = []
        for f in _I18N_DIR.glob("*.json"):
            langs.append(f.stem)
        return sorted(langs)

    @staticmethod
    def language_display_name(lang: str) -> str:
        """返回语言的显示名称"""
        names = {
            "zh-CN": "简体中文",
            "en": "English",
            "ja": "日本語",
        }
        return names.get(lang, lang)


# 快捷函数
_translator = None

def _(key: str, default: str = None) -> str:
    """快捷翻译函数"""
    global _translator
    if _translator is None:
        _translator = I18n()
    return _translator.get(key, default)


def set_language(lang: str):
    """设置语言"""
    global _translator
    if _translator is None:
        _translator = I18n()
    _translator.set_language(lang)
