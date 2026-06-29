"""
MoeFace 角色级语音转文字模块
===========================
对音频/视频进行说话人分离（Speaker Diarization）+ 语音转文字（ASR），
精准识别每段语音对应的角色身份，输出带时间戳的逐字稿。

架构:
1. 从视频中提取音频（FFmpeg / moviepy）
2. ASR 转录: faster-whisper
3. 说话人分离: 基于声纹嵌入 + 聚类
4. 角色映射: 基于已知角色声纹或手动标注
5. 输出: 结构化 JSON 逐字稿
"""

import os
import json
import threading
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .base import AIResult, AIResultCollection, BaseAIModule


@dataclass
class Utterance:
    """单句话语"""
    speaker_id: str       # 说话人 ID（如 "speaker_0", "speaker_1"）
    speaker_name: str     # 说话人姓名（如 "初音未来", "speaker_0"）
    start: float          # 开始时间（秒）
    end: float            # 结束时间（秒）
    text: str             # 转写文本
    confidence: float     # 置信度
    language: str = "zh"  # 语言


class SpeechRecognizer(BaseAIModule):
    """
    语音转文字 + 说话人分离识别器。

    用法:
        sr = SpeechRecognizer()
        sr.ensure_initialized()
        result = sr.process_audio("audio.wav")  # 或 process_video("video.mp4")
    """

    def __init__(self, model_size: str = "base"):
        """
        参数:
            model_size: whisper 模型大小 (tiny/base/small/medium/large-v3)
        """
        super().__init__()
        self._model_size = model_size
        self._whisper_model = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "speech"

    def _initialize(self, log_fn=print):
        """
        初始化 faster-whisper 模型和说话人分离组件。
        """
        try:
            import numpy as np

            # ── 加载 faster-whisper ─────────────────────────────────────
            log_fn(f"正在加载 Whisper 模型 ({self._model_size})...")
            from faster_whisper import WhisperModel

            # 尝试 GPU 加速，回退 CPU
            try:
                self._whisper_model = WhisperModel(
                    self._model_size,
                    device="cuda",
                    compute_type="float16",
                    download_root=self._get_model_dir(),
                )
                log_fn(f"✅ Whisper ({self._model_size}) 加载成功 (CUDA)")
            except Exception:
                try:
                    self._whisper_model = WhisperModel(
                        self._model_size,
                        device="cpu",
                        compute_type="int8",
                        download_root=self._get_model_dir(),
                    )
                    log_fn(f"✅ Whisper ({self._model_size}) 加载成功 (CPU/int8)")
                except Exception as e:
                    log_fn(f"⚠️ Whisper 模型加载失败: {e}")
                    self._whisper_model = None
                    return

            # ── 说话人分离配置 ──────────────────────────────────────────
            self._speaker_clustering = True
            self._num_speakers = 0  # 0 = 自动检测

            # 声纹嵌入维度 (ECAPA-TDNN)
            self._speaker_emb_dim = 192

            log_fn("✅ 语音识别模块就绪")

        except ImportError as e:
            log_fn(f"⚠️ 语音识别模块依赖缺失: {e}")
            log_fn("请安装: pip install faster-whisper")
            self._whisper_model = None
        except Exception as e:
            log_fn(f"⚠️ 语音识别模块初始化失败: {e}")

    def _get_model_dir(self) -> str:
        """获取模型缓存目录"""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base, "models", "whisper")
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def _ensure_ffmpeg(self, log_fn=print) -> bool:
        """检查 ffmpeg 是否可用"""
        import subprocess
        import shutil
        if shutil.which("ffmpeg"):
            return True
        log_fn("⚠️ 未找到 ffmpeg，请安装或添加到 PATH")
        return False

    def _extract_audio(self, video_path: str, log_fn=print) -> Optional[str]:
        """
        从视频中提取音频为 WAV 文件。
        返回临时 WAV 文件路径。
        """
        import subprocess

        if not self._ensure_ffmpeg(log_fn):
            return None

        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",                    # 不提取视频
                "-acodec", "pcm_s16le",   # PCM 16-bit
                "-ar", "16000",           # 16kHz（whisper 最佳采样率）
                "-ac", "1",               # 单声道
                out_path,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                log_fn(f"⚠️ 音频提取失败: {result.stderr[:200]}")
                return None
            return out_path
        except subprocess.TimeoutExpired:
            log_fn("⚠️ 音频提取超时")
            return None
        except Exception as e:
            log_fn(f"⚠️ 音频提取异常: {e}")
            return None

    def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        log_fn=print,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        纯转录（无说话人分离），返回段落和片段。

        返回:
            (segments, all_segments)
            segments: [{ "start", "end", "text", "confidence" }]
            all_segments: 更细粒度的片段列表
        """
        if not self._whisper_model:
            log_fn("❌ Whisper 模型未加载")
            return [], []

        try:
            log_fn("🎤 正在转录音频...")

            segments, info = self._whisper_model.transcribe(
                audio_path,
                language=None if language == "auto" else language,
                beam_size=5,
                best_of=5,
                vad_filter=True,          # 过滤非语音段
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    threshold=0.5,
                ),
            )

            segment_list = []
            all_segment_list = []

            for seg in segments:
                seg_dict = {
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": seg.text.strip(),
                    "confidence": round(seg.avg_logprob, 4) if hasattr(seg, 'avg_logprob') else 0.0,
                }
                all_segment_list.append(seg_dict)

                # 将相邻短片段合并为段落
                if segment_list and seg_dict["start"] - segment_list[-1]["end"] < 1.0:
                    segment_list[-1]["end"] = seg_dict["end"]
                    segment_list[-1]["text"] += seg_dict["text"]
                    segment_list[-1]["confidence"] = max(
                        segment_list[-1]["confidence"], seg_dict["confidence"]
                    )
                else:
                    segment_list.append(seg_dict.copy())

            detected_lang = getattr(info, "language", "unknown")
            log_fn(f"✅ 转录完成: {len(segment_list)} 段落, "
                   f"语言={detected_lang}")

            return segment_list, all_segment_list

        except Exception as e:
            log_fn(f"⚠️ 转录失败: {e}")
            return [], []

    def _extract_speaker_embeddings(
        self,
        audio_path: str,
        segments: List[Dict[str, Any]],
        log_fn=print,
    ) -> Dict[int, List[float]]:
        """
        从音频中提取说话人声纹嵌入。

        使用简单的音频能量 + 频谱特征进行说话人聚类，
        作为轻量级替代方案。

        返回 {segment_index: embedding_vector}
        """
        import numpy as np

        try:
            import librosa

            # 加载音频
            y, sr = librosa.load(audio_path, sr=16000)

            embeddings = {}
            for i, seg in enumerate(segments):
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["end"] * sr)

                if start_sample >= len(y) or end_sample > len(y):
                    continue

                segment_audio = y[start_sample:end_sample]
                if len(segment_audio) < sr * 0.3:  # 少于 0.3 秒跳过
                    continue

                try:
                    # 提取 MFCC 特征作为声纹表示
                    mfcc = librosa.feature.mfcc(
                        y=segment_audio, sr=sr, n_mfcc=13
                    )
                    # 取均值作为声纹向量
                    emb = np.mean(mfcc, axis=1).tolist()
                    embeddings[i] = emb
                except Exception:
                    continue

            log_fn(f"🔊 提取了 {len(embeddings)} 个声纹嵌入")
            return embeddings

        except ImportError:
            log_fn("⚠️ 未安装 librosa，使用基础音频特征")
            return self._basic_audio_features(audio_path, segments)
        except Exception as e:
            log_fn(f"⚠️ 声纹提取异常: {e}")
            return {}

    def _basic_audio_features(
        self,
        audio_path: str,
        segments: List[Dict[str, Any]],
        log_fn=print,
    ) -> Dict[int, List[float]]:
        """
        基础音频特征（无需 librosa）。
        使用频谱能量分布作为特征向量。
        """
        import numpy as np
        import wave

        try:
            with wave.open(audio_path, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            embeddings = {}
            for i, seg in enumerate(segments):
                start_sample = int(seg["start"] * sr)
                end_sample = int(seg["end"] * sr)

                if start_sample >= len(y) or end_sample > len(y):
                    continue

                segment_audio = y[start_sample:end_sample]
                if len(segment_audio) < sr * 0.3:
                    continue

                # 简单频谱特征
                try:
                    import numpy as np

                    # 短时能量 + 过零率 + 频带能量
                    energy = np.mean(segment_audio ** 2)
                    zcr = np.mean(np.abs(np.diff(np.sign(segment_audio))))
                    # 分频带能量
                    fft = np.fft.rfft(segment_audio)
                    freqs = np.fft.rfftfreq(len(segment_audio), 1.0 / sr)
                    bands = [0, 500, 1000, 2000, 4000, 8000]
                    band_energies = []
                    for j in range(len(bands) - 1):
                        mask = (freqs >= bands[j]) & (freqs < bands[j + 1])
                        band_energy = np.sum(np.abs(fft[mask]) ** 2) / max(1, np.sum(mask))
                        band_energies.append(float(band_energy))

                    emb = [float(energy), float(zcr)] + band_energies
                    embeddings[i] = emb
                except Exception:
                    continue

            log_fn(f"🔊 提取了 {len(embeddings)} 个基础声纹特征")
            return embeddings

        except Exception as e:
            log_fn(f"⚠️ 基础声纹提取异常: {e}")
            return {}

    def _cluster_speakers(
        self,
        embeddings: Dict[int, List[float]],
        num_speakers: int = 0,
    ) -> Dict[int, str]:
        """
        对声纹嵌入进行聚类，分配说话人 ID。

        参数:
            embeddings: {segment_index: embedding_vector}
            num_speakers: 说话人数（0=自动检测）

        返回:
            {segment_index: "speaker_0"}
        """
        if not embeddings:
            return {}

        import numpy as np

        try:
            from sklearn.cluster import AgglomerativeClustering

            indices = list(embeddings.keys())
            vectors = np.array([embeddings[i] for i in indices])

            # 标准化
            vectors = (vectors - vectors.mean(axis=0)) / (vectors.std(axis=0) + 1e-8)

            # 自动确定说话人数
            if num_speakers <= 0:
                n_samples = len(vectors)
                num_speakers = min(max(2, n_samples // 3), n_samples)
                num_speakers = min(num_speakers, 6)  # 最多 6 个说话人

            # 如果样本太少，全部分为一个人
            if len(vectors) < 2:
                return {i: "speaker_0" for i in indices}

            clustering = AgglomerativeClustering(
                n_clusters=min(num_speakers, len(vectors)),
                linkage="ward",
            )
            labels = clustering.fit_predict(vectors)

            return {indices[i]: f"speaker_{labels[i]}" for i in range(len(indices))}

        except ImportError:
            log_fn("⚠️ sklearn 未安装，使用简单聚类")
            return self._simple_cluster(embeddings)
        except Exception as e:
            log_fn(f"⚠️ 说话人聚类异常: {e}")
            return {i: "speaker_0" for i in embeddings.keys()}

    def _resolve_speaker_names(
        self,
        utterances: List[Dict[str, Any]],
        face_timeline: List[Dict],
        log_fn=print,
    ) -> List[Dict[str, Any]]:
        """
        将声纹聚类结果（speaker_0/speaker_1）与画面角色人脸识别结果进行统计匹配。

        核心逻辑：
        对每个 speaker_id，统计其说话时段内画面中各角色出现的频率，
        出现频率最高的角色即被推断为该说话人的身份。

        参数:
            utterances:    说话人分离后的段落列表
            face_timeline: 画面角色时间线 [{"timestamp": float, "names": [str]}]
            log_fn:        日志函数

        返回:
            添加了 speaker_name 之后的新段落列表
        """
        import numpy as np
        from collections import Counter

        if not face_timeline:
            log_fn("⚠️ 无画面角色数据，无法进行角色匹配")
            for utt in utterances:
                if "speaker_name" not in utt or utt["speaker_name"] == utt.get("speaker_id", ""):
                    utt["speaker_name"] = f"说话人{int(utt.get('speaker_id', 'speaker_0').split('_')[1]) + 1}"
            return utterances

        # 构建时间索引
        face_times = np.array([e["timestamp"] for e in face_timeline])

        # 按 speaker_id 分组统计
        speakers = set(u["speaker_id"] for u in utterances)
        speaker_name_map = {}

        for spk in sorted(speakers):
            spk_utterances = [u for u in utterances if u["speaker_id"] == spk]
            if not spk_utterances:
                continue

            # 收集该说话人所有说话时段内画面中的角色
            name_counter = Counter()
            for utt in spk_utterances:
                seg_start = utt["start"]
                seg_end = utt["end"]

                # 找出该时间段内的画面帧
                mask = (face_times >= seg_start) & (face_times <= seg_end)
                relevant_indices = np.where(mask)[0]

                for idx in relevant_indices:
                    for name in face_timeline[idx]["names"]:
                        name_counter[name] += 1

            if name_counter:
                # 选取出现频率最高的角色名
                top_name, top_count = name_counter.most_common(1)[0]
                total_count = sum(name_counter.values())
                confidence = top_count / total_count
                speaker_name_map[spk] = (top_name, confidence)
                log_fn(f"  🎯 {spk} → {top_name} "
                       f"(匹配置信度: {confidence:.0%}, "
                       f"共 {total_count} 次画面匹配)")
            else:
                # 未匹配到画面角色，保留原始编号
                speaker_name_map[spk] = (
                    f"说话人{int(spk.split('_')[1]) + 1}", 0.0
                )
                log_fn(f"  ⚠️ {spk} 说话时画面中无人脸，标记为画外音")

        # 应用到输出
        result = []
        for utt in utterances:
            spk = utt["speaker_id"]
            name, conf = speaker_name_map.get(spk, (f"说话人{int(spk.split('_')[1]) + 1}", 0.0))
            utt["speaker_name"] = name
            utt["face_match_confidence"] = round(conf, 4)
            result.append(utt)

        return result

    def _simple_cluster(
        self,
        embeddings: Dict[int, List[float]],
    ) -> Dict[int, str]:
        """
        简单阈值聚类（不依赖 sklearn）。
        """
        import numpy as np

        if not embeddings:
            return {}

        indices = list(embeddings.keys())
        vectors = np.array([embeddings[i] for i in indices])
        vectors = (vectors - vectors.mean(axis=0)) / (vectors.std(axis=0) + 1e-8)

        assignments = {}
        # 简单能量阈值划分：高能量/低能量 → 2 个说话人
        energies = vectors[:, 0] if vectors.shape[1] > 0 else np.zeros(len(indices))
        threshold = np.median(energies)

        for i, idx in enumerate(indices):
            assignments[idx] = "speaker_0" if energies[i] > threshold else "speaker_1"

        return assignments

    def diarize(
        self,
        segments: List[Dict[str, Any]],
        audio_path: str,
        num_speakers: int = 0,
        known_speakers: Optional[Dict[str, str]] = None,
        log_fn=print,
    ) -> List[Dict[str, Any]]:
        """
        对转录结果进行说话人分离。

        参数:
            segments:           转录段落列表
            audio_path:         音频文件路径
            num_speakers:       说话人数（0=自动）
            known_speakers:     已知说话人映射 {"speaker_0": "初音未来"}
            log_fn:             日志函数

        返回:
            [{ "speaker": str, "start": float, "end": float, "text": str, ... }]
        """
        if not segments:
            return []

        log_fn("🔊 正在进行说话人分离...")

        # 提取声纹嵌入
        embeddings = self._extract_speaker_embeddings(audio_path, segments, log_fn)
        if not embeddings:
            log_fn("⚠️ 无法提取声纹，标记为单一说话人")
            return [
                {**seg, "speaker_id": "speaker_0", "speaker_name": "说话人1"}
                for seg in segments
            ]

        # 聚类
        speaker_map = self._cluster_speakers(embeddings, num_speakers)

        # 生成最终输出
        result = []
        for i, seg in enumerate(segments):
            spk_id = speaker_map.get(i, "speaker_0")
            spk_name = known_speakers.get(spk_id, f"说话人{int(spk_id.split('_')[1]) + 1}") if known_speakers else f"说话人{int(spk_id.split('_')[1]) + 1}"

            result.append({
                **seg,
                "speaker_id": spk_id,
                "speaker_name": spk_name,
            })

        speaker_count = len(set(r["speaker_id"] for r in result))
        log_fn(f"✅ 说话人分离完成: 检测到 {speaker_count} 个说话人")
        return result

    def process_audio(
        self,
        audio_path: str,
        language: str = "auto",
        num_speakers: int = 0,
        known_speakers: Optional[Dict[str, str]] = None,
        enable_diarization: bool = True,
        log_fn=print,
    ) -> AIResultCollection:
        """
        处理音频文件：ASR + 可选说话人分离。

        返回:
            AIResultCollection
        """
        self.ensure_initialized(log_fn)

        collection = AIResultCollection(audio_path, {"speech": True})

        if not self._whisper_model:
            log_fn("❌ Whisper 模型未加载，语音识别不可用")
            return collection

        log_fn(f"🎤 开始语音识别: {audio_path}")

        # 1. 转录
        segments, _ = self.transcribe(audio_path, language, log_fn)
        if not segments:
            log_fn("⚠️ 未检测到语音内容")
            return collection

        # 2. 说话人分离
        if enable_diarization and len(segments) > 1:
            utterances = self.diarize(
                segments, audio_path, num_speakers, known_speakers, log_fn
            )
        else:
            utterances = [
                {**seg, "speaker_id": "speaker_0", "speaker_name": "说话人1"}
                for seg in segments
            ]

        # 3. 打包为统一结果
        for utt in utterances:
            result = AIResult(
                module="speech",
                event_type="utterance",
                timestamp=utt["start"],
                data={
                    "speaker_id": utt.get("speaker_id", "speaker_0"),
                    "speaker_name": utt.get("speaker_name", "未知"),
                    "start": utt["start"],
                    "end": utt["end"],
                    "text": utt["text"],
                    "confidence": utt.get("confidence", 0.0),
                },
                confidence=utt.get("confidence", 0.0),
            )
            collection.add(result)

        log_fn(f"✅ 语音识别完成: {len(utterances)} 句话")
        return collection

    def process_video(
        self,
        video_path: str,
        language: str = "auto",
        num_speakers: int = 0,
        known_speakers: Optional[Dict[str, str]] = None,
        enable_diarization: bool = True,
        face_timeline: Optional[List[Dict]] = None,
        log_fn=print,
        progress_fn=None,
        stop_event: Optional[threading.Event] = None,
    ) -> AIResultCollection:
        """
        处理视频文件：提取音频 → ASR → 说话人分离 → 画面角色关联。

        参数:
            video_path:      视频文件路径
            language:        语言 (auto/zh/en/ja)
            num_speakers:    说话人数（0=自动）
            known_speakers:  已知说话人映射
            enable_diarization: 是否启用说话人分离
            face_timeline:   画面角色时间线 [{timestamp, names:[角色]}]
            log_fn:          日志函数
            progress_fn:     进度回调
            stop_event:      停止事件

        返回:
            AIResultCollection
        """
        self.ensure_initialized(log_fn)

        collection = AIResultCollection(video_path, {"speech": True})

        if not self._whisper_model:
            log_fn("❌ Whisper 模型未加载，语音识别不可用")
            return collection

        log_fn(f"🎬 开始语音识别: {video_path}")

        # 1. 提取音频
        if stop_event and stop_event.is_set():
            return collection

        log_fn("🔊 正在从视频提取音频...")
        audio_path = self._extract_audio(video_path, log_fn)
        if not audio_path:
            log_fn("❌ 音频提取失败")
            return collection

        try:
            if progress_fn:
                progress_fn(1, 3)

            # 2. 转录
            if stop_event and stop_event.is_set():
                return collection

            log_fn("📝 转录音频中...")
            segments, _ = self.transcribe(audio_path, language, log_fn)

            if progress_fn:
                progress_fn(2, 3)

            if not segments:
                log_fn("ℹ️ 未检测到语音内容")
                return collection

            # 3. 说话人分离
            if stop_event and stop_event.is_set():
                return collection

            if enable_diarization and len(segments) > 1:
                utterances = self.diarize(
                    segments, audio_path, num_speakers, known_speakers, log_fn
                )
            else:
                utterances = [
                    {**seg, "speaker_id": "speaker_0", "speaker_name": "说话人1"}
                    for seg in segments
                ]

            if progress_fn:
                progress_fn(3, 3)

            # ── 角色融合 ──────────────────────────────────────────────
            # 将声纹聚类结果与画面角色时间线进行统计匹配
            num_speakers_found = len(set(
                u.get("speaker_id", "speaker_0") for u in utterances
            ))
            log_fn(f"  🎭 声纹聚类检测到 {num_speakers_found} 个说话人")

            if face_timeline and num_speakers_found > 1:
                log_fn("  🔗 正在进行画面-音频角色关联...")
                utterances = self._resolve_speaker_names(
                    utterances, face_timeline, log_fn
                )
            elif face_timeline and num_speakers_found <= 1:
                # 只有一个人说话，直接取画面中出现的角色
                log_fn("  🎯 仅检测到一个说话人，直接匹配画面角色")
                from collections import Counter
                name_counter = Counter()
                for ft in face_timeline:
                    for n in ft.get("names", []):
                        if n: name_counter[n] += 1
                if name_counter:
                    top_name = name_counter.most_common(1)[0][0]
                    for utt in utterances:
                        utt["speaker_name"] = top_name
                        utt["face_match_confidence"] = round(
                            name_counter[top_name] / sum(name_counter.values()), 4
                        )
                    log_fn(f"  🎯 画面中出现频率最高: {top_name}")

            # 4. 打包结果
            for utt in utterances:
                result = AIResult(
                    module="speech",
                    event_type="utterance",
                    timestamp=utt["start"],
                    data={
                        "speaker_id": utt.get("speaker_id", "speaker_0"),
                        "speaker_name": utt.get("speaker_name", "未知"),
                        "start": utt["start"],
                        "end": utt["end"],
                        "text": utt["text"],
                        "confidence": utt.get("confidence", 0.0),
                    },
                    confidence=utt.get("confidence", 0.0),
                )
                collection.add(result)

            log_fn(f"✅ 语音识别完成: {len(utterances)} 句话")
        finally:
            # 清理临时文件
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

        return collection


if __name__ == "__main__":
    sr = SpeechRecognizer(model_size="tiny")
    sr.ensure_initialized()
    print("语音识别模块初始化完成喵~")
