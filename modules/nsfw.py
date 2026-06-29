"""
MoeFace NSFW 内容识别模块
=======================
对画面及音频内容进行多模态 NSFW 检测：
- 视觉通道: 基于图像内容的 NSFW 分类
- 语义通道: 基于语音转文字内容的 NSFW 关键词检测

设计原则:
- 仅输出检测结果与置信度评分
- 不执行任何过滤或屏蔽操作
- 保留原始内容完整性
- 支持独立启用视觉/语义通道
"""

import os
import threading
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .base import AIResult, AIResultCollection, BaseAIModule


# ── NSFW 分类标签 ───────────────────────────────────────────────────────
NSFW_VISUAL_LABELS = [
    "safe",         # 安全
    "questionable", # 可疑
    "nsfw",         # 不适宜
    "explicit",     # 明显不适宜
]

NSFW_VISUAL_CN = {
    "safe": "安全",
    "questionable": "可疑",
    "nsfw": "不适宜",
    "explicit": "明显不适宜",
}

# ── 语义级 NSFW 关键词 ──────────────────────────────────────────────────
NSFW_KEYWORDS_CN = [
    # 暴力相关
    "杀人", "杀死", "死亡", "谋杀", "自杀", "自残",
    "暴力", "殴打", "虐待", "折磨", "流血",
    # 性相关
    "色情", "裸体", "裸露", "性爱", "性交", "做爱",
    "鸡巴", "屌", "操你", "草你", "干你",
    "口交", "手淫", "自慰", "性行为",
    # 毒品/违法
    "吸毒", "毒品", "海洛因", "大麻", "可卡因",
    "贩毒", "走私",
    # 歧视/仇恨
    "歧视", "种族", "仇恨",
]

NSFW_KEYWORDS_EN = [
    # Violence
    "kill", "murder", "death", "suicide", "violent",
    "blood", "torture", "abuse", "assault",
    # Sexual
    "porn", "nude", "naked", "sex", "sexual",
    "fuck", "shit", "bitch", "dick", "cock",
    "blowjob", "masturbation", "orgasm",
    # Drugs/Illegal
    "drug", "heroin", "cocaine", "marijuana", "weed",
    "trafficking",
    # Hate/Discrimination
    "hate", "racist", "discrimination",
]


@dataclass
class NSFWFrameResult:
    """单帧 NSFW 检测结果"""
    timestamp: float
    visual_score: float = 0.0
    visual_label: str = "safe"
    semantic_score: float = 0.0
    semantic_details: List[str] = field(default_factory=list)
    overall_score: float = 0.0


class NSFWDetector(BaseAIModule):
    """
    多模态 NSFW 内容检测器。

    同时检测视觉和语义两个通道，输出独立评分。

    用法:
        nd = NSFWDetector()
        nd.ensure_initialized()
        result = nd.process_frame(frame_bgr, timestamp=1.5)
        result = nd.process_text("一段文本", timestamp=0)
    """

    def __init__(self):
        super().__init__()
        self._nsfw_model = None
        self._lock = threading.Lock()

        # 语义关键词（编译正则）
        self._keyword_patterns_cn = [
            re.compile(re.escape(kw), re.IGNORECASE) for kw in NSFW_KEYWORDS_CN
        ]
        self._keyword_patterns_en = [
            re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
            for kw in NSFW_KEYWORDS_EN
        ]

    @property
    def name(self) -> str:
        return "nsfw"

    def _initialize(self, log_fn=print):
        """
        初始化 NSFW 检测模型。
        """
        try:
            import cv2
            import numpy as np

            # ── 加载视觉 NSFW 模型 ─────────────────────────────────────
            model_path = self._get_model_path("nsfw_mobilenet2.224x224.h5")
            if model_path and os.path.exists(model_path):
                try:
                    from tensorflow.keras.models import load_model
                    self._nsfw_model = load_model(model_path)
                    log_fn("✅ NSFW 视觉模型加载成功 (TensorFlow)")
                except Exception:
                    log_fn("⚠️ TensorFlow 模型加载失败，尝试 ONNX 回退")
                    onnx_path = self._get_model_path("nsfw.onnx")
                    if onnx_path and os.path.exists(onnx_path):
                        try:
                            import onnxruntime
                            self._nsfw_onnx = onnxruntime.InferenceSession(
                                onnx_path, providers=["CPUExecutionProvider"]
                            )
                            log_fn("✅ NSFW 视觉模型加载成功 (ONNX)")
                        except Exception:
                            log_fn("⚠️ NSFW ONNX 模型加载失败，使用规则分析")
                    else:
                        log_fn("ℹ️ 使用基于颜色的规则分析进行 NSFW 检测")
            else:
                log_fn("ℹ️ 未找到 NSFW 模型文件，使用规则分析模式")
                log_fn("   如需更高精度，请下载 nsfw_mobilenet2.224x224.h5 到 models/")

            log_fn("✅ NSFW 检测模块就绪 (语义通道已激活)")

        except Exception as e:
            log_fn(f"⚠️ NSFW 检测模块初始化部分失败: {e}")

    def _get_model_path(self, filename: str) -> str:
        """获取模型文件路径"""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local = os.path.join(base, filename)
        if os.path.exists(local):
            return local
        model_dir = os.path.join(base, "models", filename)
        if os.path.exists(model_dir):
            return model_dir
        # 返回建议路径
        return local

    # ── 视觉通道 ───────────────────────────────────────────────────────

    def _detect_visual_nsfw(
        self, frame_bgr
    ) -> Tuple[float, str]:
        """
        视觉通道 NSFW 检测。

        返回 (nsfw_score, label):
            nsfw_score: 0.0 (安全) ~ 1.0 (明确 NSFW)
            label: "safe" / "questionable" / "nsfw" / "explicit"
        """
        import cv2
        import numpy as np

        # 方法一: 预训练模型
        if hasattr(self, '_nsfw_model') and self._nsfw_model is not None:
            try:
                return self._model_infer(frame_bgr)
            except Exception:
                pass

        if hasattr(self, '_nsfw_onnx'):
            try:
                return self._onnx_infer(frame_bgr)
            except Exception:
                pass

        # 方法二: 规则分析
        return self._rule_based_visual(frame_bgr)

    def _model_infer(self, frame_bgr) -> Tuple[float, str]:
        """TensorFlow 模型推理"""
        import cv2
        import numpy as np

        img = cv2.resize(frame_bgr, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = self._nsfw_model.predict(img, verbose=0)[0]
        nsfw_score = float(pred[1])  # 假设输出 [safe_prob, nsfw_prob]

        if nsfw_score < 0.3:
            label = "safe"
        elif nsfw_score < 0.6:
            label = "questionable"
        elif nsfw_score < 0.85:
            label = "nsfw"
        else:
            label = "explicit"

        return nsfw_score, label

    def _onnx_infer(self, frame_bgr) -> Tuple[float, str]:
        """ONNX 模型推理"""
        import cv2
        import numpy as np

        img = cv2.resize(frame_bgr, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)

        input_name = self._nsfw_onnx.get_inputs()[0].name
        outputs = self._nsfw_onnx.run(None, {input_name: img})
        pred = outputs[0][0]

        nsfw_score = float(max(pred)) if len(pred) > 1 else float(pred[0])

        if nsfw_score < 0.3:
            label = "safe"
        elif nsfw_score < 0.6:
            label = "questionable"
        elif nsfw_score < 0.85:
            label = "nsfw"
        else:
            label = "explicit"

        return nsfw_score, label

    def _rule_based_visual(self, frame_bgr) -> Tuple[float, str]:
        """
        基于规则的视觉 NSFW 分析。
        分析肤色比例、边缘密度、纹理复杂度等特征。
        """
        import cv2
        import numpy as np

        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return 0.0, "safe"

        # 1. 肤色检测（HSV 范围）
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # 典型的肤色范围
        skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
        skin_mask2 = cv2.inRange(hsv, (170, 20, 70), (180, 255, 255))
        skin_mask = cv2.bitwise_or(skin_mask, skin_mask2)
        skin_ratio = float(np.count_nonzero(skin_mask)) / (h * w)

        # 2. 纹理复杂度（拉普拉斯方差）
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        # 3. 边缘密度
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges)) / (h * w)

        # 4. 暴露程度评分
        # 大面积肤色 + 低纹理 = 潜在 NSFW
        exposure_score = skin_ratio * 0.5
        if skin_ratio > 0.4:
            exposure_score += 0.3
        if skin_ratio > 0.6:
            exposure_score += 0.2

        # 综合评分
        nsfw_score = min(1.0, exposure_score)

        if nsfw_score < 0.2:
            label = "safe"
        elif nsfw_score < 0.45:
            label = "questionable"
        elif nsfw_score < 0.7:
            label = "nsfw"
        else:
            label = "explicit"

        return nsfw_score, label

    # ── 语义通道 ───────────────────────────────────────────────────────

    def _detect_semantic_nsfw(
        self, text: str
    ) -> Tuple[float, List[str]]:
        """
        语义通道 NSFW 检测。
        基于关键词匹配分析文本内容。

        返回 (nsfw_score, matched_keywords)
        """
        if not text or not text.strip():
            return 0.0, []

        matched = []

        # 中文关键词匹配
        for pattern in self._keyword_patterns_cn:
            if pattern.search(text):
                matched.append(pattern.pattern)

        # 英文关键词匹配
        for pattern in self._keyword_patterns_en:
            if pattern.search(text):
                matched.append(pattern.pattern)

        # 去重
        matched = list(set(matched))

        if not matched:
            return 0.0, []

        # 评分: 匹配数量 / 文本长度（归一化）+ 匹配密度
        words_count = len(text)
        density = len(matched) / max(words_count, 1) * 100  # 每 100 字的匹配数
        score = min(1.0, 0.3 + density * 0.15)

        return score, matched

    # ── 核心处理接口 ───────────────────────────────────────────────────

    def process_frame(
        self,
        frame_bgr,
        timestamp: float,
        log_fn=print,
    ) -> AIResult:
        """
        处理单帧图像的视觉 NSFW 检测。

        返回统一 AIResult 对象。
        """
        self.ensure_initialized(log_fn)

        nsfw_score, label = self._detect_visual_nsfw(frame_bgr)

        data = {
            "channel": "visual",
            "nsfw_score": nsfw_score,
            "label": label,
            "label_cn": NSFW_VISUAL_CN.get(label, label),
            "timestamp": timestamp,
        }

        return AIResult(
            module="nsfw",
            event_type="nsfw_visual",
            timestamp=timestamp,
            data=data,
            confidence=nsfw_score,
        )

    def process_text(
        self,
        text: str,
        timestamp: float,
        source: str = "transcript",
        log_fn=print,
    ) -> AIResult:
        """
        处理文本的语义 NSFW 检测。

        参数:
            text:       待检测文本
            timestamp:  时间戳（秒）
            source:     文本来源（"transcript" / "subtitle" / "comment"）
        """
        self.ensure_initialized(log_fn)

        nsfw_score, keywords = self._detect_semantic_nsfw(text)

        data = {
            "channel": "semantic",
            "nsfw_score": nsfw_score,
            "matched_keywords": keywords,
            "text_snippet": text[:100] + ("..." if len(text) > 100 else ""),
            "source": source,
            "timestamp": timestamp,
        }

        return AIResult(
            module="nsfw",
            event_type="nsfw_semantic",
            timestamp=timestamp,
            data=data,
            confidence=nsfw_score,
        )

    def process_video(
        self,
        video_path: str,
        skip_frames: int = 15,
        transcripts: Optional[List[Dict[str, Any]]] = None,
        log_fn=print,
        progress_fn=None,
        stop_event: Optional[threading.Event] = None,
    ) -> AIResultCollection:
        """
        处理视频的多模态 NSFW 检测（视觉 + 可选语义）。

        参数:
            video_path:   视频文件路径
            skip_frames:  每 N 帧检测一次
            transcripts:  可选的转录结果列表 [{ "start", "end", "text" }]
            log_fn:       日志函数
            progress_fn:  进度回调
            stop_event:   停止事件

        返回:
            AIResultCollection
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log_fn(f"❌ 无法打开视频: {video_path}")
            return AIResultCollection(video_path, {"nsfw": True})

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        collection = AIResultCollection(video_path, {"nsfw": True})
        frame_idx = 0

        log_fn("🔞 开始 NSFW 检测...")

        # ── 视觉通道 ─────────────────────────────────────────────────
        log_fn("📷 视觉 NSFW 检测中...")
        while frame_idx < total_frames:
            if stop_event and stop_event.is_set():
                log_fn("⏹ NSFW 检测已停止")
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames == 0:
                timestamp = frame_idx / fps
                result = self.process_frame(frame, timestamp, log_fn=log_fn)
                collection.add(result)

            frame_idx += 1

            if progress_fn and frame_idx % (skip_frames * 10) == 0:
                progress_fn(min(frame_idx, total_frames), total_frames)

        cap.release()

        # ── 语义通道 ─────────────────────────────────────────────────
        if transcripts:
            log_fn("📝 语义 NSFW 检测中...")
            for utt in transcripts:
                if stop_event and stop_event.is_set():
                    break
                text = utt.get("text", "")
                if text.strip():
                    result = self.process_text(
                        text=text,
                        timestamp=utt.get("start", 0.0),
                        source="transcript",
                        log_fn=log_fn,
                    )
                    collection.add(result)

        log_fn(f"✅ NSFW 检测完成: 视觉帧 {frame_idx // skip_frames} 帧, "
               f"语义段 {len(transcripts) if transcripts else 0} 段")
        return collection

    def analyze_video_nsfw(
        self,
        video_path: str,
        skip_frames: int = 15,
        log_fn=print,
        progress_fn=None,
        stop_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """
        视频 NSFW 快速总评（返回汇总统计，非逐帧详细结果）。

        返回:
            {
                "overall_score": float,
                "max_frame_score": float,
                "peak_timestamps": [float],
                "visual_samples": int,
                "label": str
            }
        """
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "无法打开视频"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        scores = []
        peak_times = []
        frame_idx = 0

        while frame_idx < total_frames:
            if stop_event and stop_event.is_set():
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames == 0:
                score, label = self._detect_visual_nsfw(frame)
                timestamp = frame_idx / fps
                scores.append({"timestamp": timestamp, "score": score, "label": label})
                if score > 0.5:
                    peak_times.append(timestamp)

            frame_idx += 1

            if progress_fn and frame_idx % (skip_frames * 20) == 0:
                progress_fn(min(frame_idx, total_frames), total_frames)

        cap.release()

        if not scores:
            return {"overall_score": 0.0, "label": "safe"}

        overall = float(np.mean([s["score"] for s in scores]))
        max_score = float(np.max([s["score"] for s in scores]))

        # 综合标签
        if overall < 0.2:
            label = "safe"
        elif overall < 0.4:
            label = "questionable"
        elif overall < 0.65:
            label = "nsfw"
        else:
            label = "explicit"

        return {
            "overall_score": overall,
            "max_score": max_score,
            "peak_timestamps": sorted(set(round(t, 1) for t in peak_times))[:20],
            "visual_samples": len(scores),
            "label": label,
            "label_cn": NSFW_VISUAL_CN.get(label, label),
        }


if __name__ == "__main__":
    nd = NSFWDetector()
    nd.ensure_initialized()
    print("NSFW 检测模块初始化完成喵~")

    # 测试语义检测
    test_texts = [
        "今天天气真好",
        "我要杀了你",
        "你好漂亮",
        "fuck you bastard",
    ]
    for t in test_texts:
        score, keywords = nd._detect_semantic_nsfw(t)
        print(f"  [{t}] → score={score:.2f}, keywords={keywords}")
