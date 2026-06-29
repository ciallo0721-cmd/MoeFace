"""
MoeFace 模块基类 — 统一所有 AI 识别模块的生命周期与输出格式
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime


class AIResult:
    """
    统一识别结果容器。

    每个识别事件（单帧/单个音频段/单张检测）均包装为此对象，
    便于序列化为 JSON 供后续展示或存储。
    """

    def __init__(
        self,
        module: str,
        event_type: str,
        timestamp: float,
        data: Dict[str, Any],
        confidence: Optional[float] = None,
    ):
        self.module = module          # 模块名: "emotion" / "speech" / "nsfw"
        self.event_type = event_type  # 事件类型: "face_emotion" / "utterance" / "nsfw_frame" / ...
        self.timestamp = timestamp    # 时间戳（秒）
        self.data = data              # 具体结果数据
        self.confidence = confidence  # 总体置信度（可选）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "data": self.data,
            "confidence": self.confidence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __repr__(self) -> str:
        return f"<AIResult {self.module}.{self.event_type} @ {self.timestamp:.2f}s>"


class AIResultCollection:
    """
    一次处理任务的全部识别结果集合。
    可序列化为统一 JSON，方便前端/外部系统消费。
    """

    def __init__(self, source: str, modules_enabled: Dict[str, bool]):
        self.source = source
        self.modules_enabled = modules_enabled
        self.created_at = datetime.now().isoformat()
        self.results: list = []

    def add(self, result: AIResult):
        self.results.append(result.to_dict())

    def extend(self, results: list):
        for r in results:
            if isinstance(r, AIResult):
                self.results.append(r.to_dict())
            else:
                self.results.append(r)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "modules_enabled": self.modules_enabled,
            "created_at": self.created_at,
            "results": self.results,
        }

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def __len__(self) -> int:
        return len(self.results)


class BaseAIModule(ABC):
    """
    所有 AI 识别模块的抽象基类。

    子类需要实现:
    - name: 模块名称（静态属性）
    - _initialize(): 加载模型等一次性初始化
    """

    def __init__(self):
        self._initialized = False

    @property
    @abstractmethod
    def name(self) -> str:
        """模块名称，如 'emotion'、'speech'、'nsfw'"""
        ...

    def ensure_initialized(self, log_fn=print):
        """线程安全地初始化（只做一次）"""
        if not self._initialized:
            self._initialize(log_fn)
            self._initialized = True

    @abstractmethod
    def _initialize(self, log_fn=print):
        """加载模型和依赖"""
        ...
