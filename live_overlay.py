"""
MoeFace 直播集成模块 — OBS 虚拟摄像头输出 + 实时弹幕标注
"""
import threading
import queue
import time
from typing import Optional, Callable

import numpy as np
import cv2

try:
    import pyvirtualcam
    VIRTUAL_CAM_AVAILABLE = True
except ImportError:
    VIRTUAL_CAM_AVAILABLE = False


class LiveOverlay:
    """
    实时直播叠加层

    接收识别结果帧，叠加弹幕样式信息后通过虚拟摄像头输出
    """

    def __init__(self, fps: int = 30, width: int = 1280, height: int = 720):
        self.fps = fps
        self.width = width
        self.height = height
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cam: Optional[pyvirtualcam.Camera] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=8)

        # 弹幕样式
        self.danmaku_enabled = True
        self.danmaku_size = 1.0
        self.danmaku_color = (0, 255, 0)  # BGR
        self.danmaku_position = "top"  # top / bottom / auto
        self.danmaku_opacity = 0.8

        # 边框叠加
        self.show_border = True
        self.border_color = (110, 231, 183)
        self.border_thickness = 2

        # 当前帧叠加信息
        self._current_annotations: list = []  # [{bbox, name, score}]

    def update_annotations(self, annotations: list):
        """更新当前帧的识别标注信息"""
        self._current_annotations = annotations

    def push_frame(self, frame_bgr: np.ndarray):
        """推入一帧（带标注渲染）"""
        if not self._running:
            return
        # 渲染标注
        if self.show_border:
            for ann in self._current_annotations:
                bbox = ann.get("bbox", {})
                x, y, w, h = bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0)
                name = ann.get("name", "?")
                score = ann.get("score", 0)
                cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), self.border_color, self.border_thickness)
                label = f"{name} {score:.0%}"
                cv2.putText(frame_bgr, label, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.border_color, 1)

        # 渲染弹幕（如果有）
        if self.danmaku_enabled:
            overlay = frame_bgr.copy()
            alpha = self.danmaku_opacity
            # 在顶部显示当前识别角色
            names = [a.get("name", "") for a in self._current_annotations if a.get("name")]
            if names:
                text = " | ".join(set(names))
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                tw, th = text_size
                x_pos = (self.width - tw) // 2
                y_pos = 30
                cv2.rectangle(overlay, (x_pos-10, y_pos-th-10), (x_pos+tw+10, y_pos+10),
                              (30, 30, 30), -1)
                cv2.addWeighted(overlay, alpha, frame_bgr, 1-alpha, 0, frame_bgr)
                cv2.putText(frame_bgr, text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)

        # 缩放至输出尺寸
        if frame_bgr.shape[:2] != (self.height, self.width):
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

        # 送入队列（非阻塞）
        try:
            self._frame_queue.put_nowait(frame_bgr)
        except queue.Full:
            pass

    def _output_loop(self):
        """虚拟摄像头输出循环"""
        if not VIRTUAL_CAM_AVAILABLE:
            return

        try:
            with pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps,
                                     backend="obs", print_fps=False) as cam:
                self._cam = cam
                black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                while self._running:
                    try:
                        frame = self._frame_queue.get(timeout=1.0/self.fps)
                    except queue.Empty:
                        frame = black_frame
                    # 转换为 RGB（pyvirtualcam 需要 RGB）
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cam.send(frame_rgb)
                    cam.sleep_until_next_frame()
        except Exception as e:
            print(f"[LiveOverlay] 虚拟摄像头异常: {e}")

    def start(self):
        """启动虚拟摄像头输出"""
        if not VIRTUAL_CAM_AVAILABLE:
            return False
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._output_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """停止虚拟摄像头输出"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        self._cam = None
