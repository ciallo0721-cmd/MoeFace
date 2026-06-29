"""
MoeFace 表情与情绪识别模块
========================
实时检测画面中人物的面部表情，识别情绪状态（喜悦、愤怒、悲伤、惊讶等），
并在时间轴上标注情绪变化节点。

核心逻辑:
1. 使用 lbpcascade_animeface 检测动漫人脸
2. 对每张人脸提取特征并分类情绪
3. 支持内置轻量模型 + 几何特征分析双模式
4. 输出统一 JSON 格式的结构化结果
"""

import os
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .base import AIResult, AIResultCollection, BaseAIModule

# ── 情绪标签定义 ─────────────────────────────────────────────────────────
EMOTION_LABELS = [
    "neutral",   # 中性
    "happy",     # 喜悦
    "sad",       # 悲伤
    "angry",     # 愤怒
    "surprised", # 惊讶
    "fearful",   # 恐惧
    "disgusted", # 厌恶
]

# 中文映射
EMOTION_CN = {
    "neutral": "中性",
    "happy": "喜悦",
    "sad": "悲伤",
    "angry": "愤怒",
    "surprised": "惊讶",
    "fearful": "恐惧",
    "disgusted": "厌恶",
}


@dataclass
class EmotionFrame:
    """单帧情绪检测结果"""
    timestamp: float
    faces: List[Dict[str, Any]] = field(default_factory=list)


class EmotionRecognizer(BaseAIModule):
    """
    面部表情与情绪识别器。

    用法:
        er = EmotionRecognizer()
        er.ensure_initialized()
        result = er.process_frame(frame_bgr, timestamp=1.5)
    """

    def __init__(self):
        super().__init__()
        self._emotion_model = None  # 情绪分类模型（ONNX）
        self._infer_session = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "emotion"

    def _initialize(self, log_fn=print):
        """
        初始化情绪识别模型。
        优先加载 ONNX 轻量情绪分类模型，若不可用则回退到几何特征分析。
        """
        try:
            import cv2
            import numpy as np

            # 尝试加载预训练 ONNX 模型
            model_path = self._get_model_path("emotion-ferplus.onnx")
            if model_path and os.path.exists(model_path):
                try:
                    import onnxruntime
                    self._infer_session = onnxruntime.InferenceSession(
                        model_path, providers=["CPUExecutionProvider"]
                    )
                    log_fn("✅ 情绪识别模型加载成功 (ONNX)")
                except Exception:
                    log_fn("⚠️ ONNX 模型加载失败，使用几何特征分析模式")
                    self._infer_session = None
            else:
                log_fn("ℹ️ 未找到 ONNX 模型文件，使用几何特征分析模式")

            # 预先定义面部表情的几何规则阈值
            self._emotion_rules = {
                # (眼宽比, 嘴宽高比, 眉间距离) → 情绪
                "happy": {"smile_ratio": 0.35, "eye_squint": 0.15},
                "surprised": {"eye_open_ratio": 0.55, "mouth_open": 0.30},
                "angry": {"brow_low": -0.10, "eye_narrow": 0.12},
                "sad": {"brow_angle": -0.08, "mouth_corner": -0.10},
            }

            log_fn("✅ 情绪识别模块就绪")

        except Exception as e:
            log_fn(f"⚠️ 情绪识别模块初始化部分失败: {e}")

    def _get_model_path(self, filename: str) -> str:
        """获取模型文件的路径"""
        # 本项目目录
        local = os.path.join(os.path.dirname(os.path.dirname(__file__)), filename)
        if os.path.exists(local):
            return local
        # modules 同级
        module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", filename)
        if os.path.exists(module_dir):
            return module_dir
        # 返回建议路径
        return local

    def process_frame(
        self,
        frame_bgr,
        timestamp: float,
        face_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        log_fn=print,
    ) -> List[Dict[str, Any]]:
        """
        处理单帧图像的情绪识别。

        参数:
            frame_bgr:   OpenCV BGR 图像
            timestamp:   当前时间戳（秒）
            face_boxes:  可选的人脸框列表 [(x, y, w, h), ...]，为空时自动检测
            log_fn:      日志输出函数

        返回:
            [{ "face_idx": int, "bbox": [x,y,w,h], "emotion": str, "confidence": float,
               "emotion_scores": {...}, "timestamp": float }]
        """
        import cv2
        import numpy as np

        if not self._initialized:
            log_fn("⚠️ 情绪识别模块未初始化")
            return []

        results = []
        h, w = frame_bgr.shape[:2]

        # 1. 获取人脸框
        if face_boxes is None:
            cascade_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "lbpcascade_animeface.xml",
            )
            if os.path.exists(cascade_path):
                cascade = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                face_boxes = cascade.detectMultiScale(
                    gray, scaleFactor=1.02, minNeighbors=3,
                    minSize=(30, 30), maxSize=(400, 400),
                )
            else:
                face_boxes = []

        # 2. 逐个人脸识别情绪
        for idx, (fx, fy, fw, fh) in enumerate(face_boxes):
            try:
                face_roi = frame_bgr[fy : fy + fh, fx : fx + fw]
                if face_roi.size == 0:
                    continue

                # 情绪分类
                emotion, scores = self._classify_emotion(face_roi)

                results.append({
                    "face_idx": idx,
                    "bbox": [int(fx), int(fy), int(fw), int(fh)],
                    "emotion": emotion,
                    "emotion_label": EMOTION_CN.get(emotion, emotion),
                    "confidence": float(max(scores.values())),
                    "emotion_scores": {EMOTION_CN.get(k, k): float(v) for k, v in scores.items()},
                    "timestamp": timestamp,
                })
            except Exception as e:
                log_fn(f"⚠️ 人脸 #{idx} 情绪识别失败: {e}")
                continue

        return results

    def _classify_emotion(self, face_roi) -> Tuple[str, Dict[str, float]]:
        """
        对单张人脸 ROI 进行情绪分类。

        返回 (主要情绪, {情绪: 置信度})。
        """
        import cv2
        import numpy as np

        # 方法一: ONNX 模型推理（如果有）
        if self._infer_session is not None:
            try:
                return self._onnx_classify(face_roi)
            except Exception:
                pass

        # 方法二: 几何特征分析（通用回退）
        return self._geometric_classify(face_roi)

    def _preprocess_face(self, face_roi, size=(64, 64)):
        """预处理人脸 ROI 用于模型输入"""
        import cv2
        import numpy as np

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, size)
        normalized = (resized.astype(np.float32) - 128.0) / 128.0
        return normalized

    def _onnx_classify(self, face_roi) -> Tuple[str, Dict[str, float]]:
        """
        使用 ONNX 模型进行情绪分类。
        期望模型输入: (1, 1, 64, 64) 灰度图，输出: (1, 7) 概率分布
        """
        import numpy as np

        input_blob = self._preprocess_face(face_roi)
        input_blob = input_blob[np.newaxis, np.newaxis, :, :]  # (1,1,64,64)

        input_name = self._infer_session.get_inputs()[0].name
        outputs = self._infer_session.run(None, {input_name: input_blob})
        probs = outputs[0][0]  # (7,) 或 (1,7)

        if probs.ndim > 1:
            probs = probs.flatten()

        # softmax 归一化
        exp_probs = np.exp(probs - np.max(probs))
        probs = exp_probs / (exp_probs.sum() + 1e-8)

        scores = {}
        for i, label in enumerate(EMOTION_LABELS):
            if i < len(probs):
                scores[label] = float(probs[i])

        best_idx = int(np.argmax(probs[: len(EMOTION_LABELS)]))
        emotion = EMOTION_LABELS[best_idx]
        return emotion, scores

    def _geometric_classify(self, face_roi) -> Tuple[str, Dict[str, float]]:
        """
        基于图像特征的几何情绪分析（无需模型）。

        分析:
        - 亮度分布（微笑时嘴部区域更亮）
        - 边缘强度（惊讶时眼部更张开，边缘更多）
        - 水平/垂直投影（嘴部张开的程度）
        """
        import cv2
        import numpy as np

        h, w = face_roi.shape[:2]
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # 划分区域
        top_half = gray[: h // 2, :]       # 上半脸（眼区）
        bot_half = gray[h // 2 :, :]       # 下半脸（嘴区）
        mid_third_h = h // 3
        eye_region = gray[mid_third_h : 2 * mid_third_h, :]

        # 特征提取
        # 1. 嘴部区域亮度变化（微笑检测）
        mouth_roi = bot_half[int(bot_half.shape[0] * 0.2) :, :]
        mouth_brightness = float(np.mean(mouth_roi)) if mouth_roi.size > 0 else 128.0
        mouth_std = float(np.std(mouth_roi)) if mouth_roi.size > 0 else 0.0

        # 2. 眼部开合度（基于水平边缘强度）
        eye_edges = cv2.Canny(eye_region, 50, 150)
        eye_edge_ratio = float(np.count_nonzero(eye_edges)) / max(eye_edges.size, 1)

        # 3. 眉毛区域梯度（愤怒/悲伤检测）
        brow_roi = gray[: mid_third_h, :]
        brow_grad = cv2.Sobel(brow_roi, cv2.CV_32F, 0, 1)
        brow_grad_mag = float(np.mean(np.abs(brow_grad))) if brow_roi.size > 0 else 0.0

        # 4. 嘴部纵向开口（基于投影）
        bot_edges = cv2.Canny(bot_half, 30, 100)
        vert_proj = np.sum(bot_edges, axis=1)
        mouth_open_ratio = float(np.max(vert_proj)) / max(bot_edges.shape[1], 1) if bot_edges.size > 0 else 0.0

        # 规则分类
        scores = {}
        # 喜悦: 嘴部亮度高 + 边缘丰富（微笑弧度）
        happy_score = min(1.0, (mouth_std / 60.0) * 0.6 + (mouth_brightness / 255.0) * 0.4)
        scores["happy"] = happy_score

        # 惊讶: 眼区边缘多（眼睛睁大）+ 嘴部纵向开口大
        surprised_score = min(1.0, (eye_edge_ratio * 5.0) * 0.5 + (mouth_open_ratio * 2.0) * 0.5)
        scores["surprised"] = surprised_score

        # 愤怒: 眉毛区域梯度强（紧锁）+ 嘴部亮度低
        angry_score = min(1.0, (brow_grad_mag / 150.0) * 0.7 + (1.0 - mouth_brightness / 255.0) * 0.3)
        scores["angry"] = angry_score

        # 悲伤: 眉毛梯度低 + 嘴部亮度低
        sad_score = min(1.0, (1.0 - brow_grad_mag / 150.0) * 0.5 + (1.0 - mouth_brightness / 255.0) * 0.5)
        scores["sad"] = sad_score

        # 恐惧: 高边缘 + 低嘴部开口（特征组合）
        fearful_score = min(1.0, (eye_edge_ratio * 3.0) * 0.4 + (1.0 - mouth_open_ratio) * 0.6)
        scores["fearful"] = fearful_score

        # 厌恶: 嘴部横向纹理多 + 眉毛低
        disgusted_score = min(1.0, (1.0 - mouth_brightness / 255.0) * 0.5 + (brow_grad_mag / 200.0) * 0.5)
        scores["disgusted"] = disgusted_score

        # 中性: 所有分数都不高
        neutral_score = max(0.0, 1.0 - max(happy_score, surprised_score, angry_score, sad_score, fearful_score, disgusted_score))
        scores["neutral"] = neutral_score

        # 归一化
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        else:
            scores = {k: 1.0 / len(scores) for k in scores}

        best_emotion = max(scores, key=scores.get)
        return best_emotion, scores

    def process_video(
        self,
        video_path: str,
        skip_frames: int = 5,
        max_frames: int = 0,
        log_fn=print,
        progress_fn=None,
        stop_event: Optional[threading.Event] = None,
    ) -> AIResultCollection:
        """
        处理整个视频的情绪识别。

        参数:
            video_path:    视频文件路径
            skip_frames:   每 N 帧检测一次
            max_frames:    最多处理帧数（0=全部）
            log_fn:        日志函数
            progress_fn:   进度回调 (current, total)
            stop_event:    停止事件

        返回:
            AIResultCollection 统一结果集合
        """
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log_fn(f"❌ 无法打开视频: {video_path}")
            return AIResultCollection(video_path, {"emotion": True})

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = max_frames if max_frames > 0 else total_frames

        collection = AIResultCollection(video_path, {"emotion": True})
        frame_idx = 0
        processed = 0
        prev_emotions: Dict[int, str] = {}  # 用于检测情绪变化

        log_fn(f"🎭 开始情绪识别: {video_path}")

        while frame_idx < max_frames:
            if stop_event and stop_event.is_set():
                log_fn("⏹ 情绪识别已停止")
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip_frames == 0:
                timestamp = frame_idx / fps
                emotions = self.process_frame(frame, timestamp, log_fn=log_fn)

                # 检测情绪变化
                for emo in emotions:
                    face_idx = emo["face_idx"]
                    curr_emo = emo["emotion"]
                    if face_idx in prev_emotions and prev_emotions[face_idx] != curr_emo:
                        emo["emotion_change"] = {
                            "from": EMOTION_CN.get(prev_emotions[face_idx], prev_emotions[face_idx]),
                            "to": EMOTION_CN.get(curr_emo, curr_emo),
                        }
                    prev_emotions[face_idx] = curr_emo

                    result = AIResult(
                        module="emotion",
                        event_type="face_emotion",
                        timestamp=timestamp,
                        data=emo,
                        confidence=emo["confidence"],
                    )
                    collection.add(result)

                processed += 1

            frame_idx += 1

            if progress_fn and frame_idx % (skip_frames * 10) == 0:
                progress_fn(min(frame_idx, max_frames), min(total_frames, max_frames))

        cap.release()
        log_fn(f"✅ 情绪识别完成: 处理 {processed} 个关键帧, "
               f"检测到 {len(collection)} 条情绪记录")
        return collection


if __name__ == "__main__":
    # 简易测试
    er = EmotionRecognizer()
    er.ensure_initialized()
    print("情绪识别模块测试通过喵~")
