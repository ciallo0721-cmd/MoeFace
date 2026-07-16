"""
MoeFace 推理分析报告生成器
支持 PDF / HTML 格式报告
"""
import os, json, io, base64
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


class ReportGenerator:
    """推理分析报告生成器"""

    def __init__(self, title: str = "MoeFace 识别分析报告"):
        self.title = title
        self.created_at = datetime.now()
        self.source_file: Optional[str] = None
        self.frames: List[Dict] = []          # {timestamp, image_base64, roles: [{name, score}]}
        self.appearance_stats: Dict[str, List[float]] = {}  # role_name -> [timestamps]
        self.emotion_curve: List[Dict] = []    # {timestamp, emotion, confidence}
        self.speech_segments: List[Dict] = []  # {start, end, speaker, text}
        self.nsfw_results: List[Dict] = []     # {timestamp, label, score}
        self.total_processed = 0
        self.unique_roles: set = set()
        self.parameters: Dict = {}

    def add_frame(self, timestamp: float, image_cv, roles: list):
        """添加一帧的识别结果"""
        import cv2
        _, buffer = cv2.imencode(".jpg", image_cv, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buffer).decode("utf-8")
        self.frames.append({
            "timestamp": timestamp,
            "image_b64": b64,
            "roles": roles,
        })
        self.total_processed += 1

    def _generate_charts(self, output_dir: str) -> List[str]:
        """生成统计图表，返回图片路径列表"""
        charts = []
        if not MPL_AVAILABLE:
            return charts

        # 1. 角色出场统计（条形图）
        if self.appearance_stats:
            fig, ax = plt.subplots(figsize=(10, 5))
            names = list(self.appearance_stats.keys())
            counts = [len(v) for v in self.appearance_stats.values()]
            colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
            bars = ax.barh(range(len(names)), counts, color=colors)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=10)
            ax.set_xlabel("出现帧数")
            ax.set_title("角色出场统计")
            for bar, c in zip(bars, counts):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                        str(c), va="center", fontsize=9)
            plt.tight_layout()
            path = os.path.join(output_dir, "chart_appearance.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            charts.append(path)

        # 2. 情绪曲线（如果有）
        if self.emotion_curve:
            fig, ax = plt.subplots(figsize=(10, 4))
            timestamps = [e["timestamp"] for e in self.emotion_curve]
            confidences = [e["confidence"] for e in self.emotion_curve]
            emotions = [e["emotion"] for e in self.emotion_curve]
            colors_map = {"happy": "gold", "sad": "cornflowerblue", "angry": "tomato",
                          "surprise": "limegreen", "neutral": "gray", "fear": "purple",
                          "disgust": "brown"}
            scatter_colors = [colors_map.get(e, "gray") for e in emotions]
            ax.scatter(timestamps, confidences, c=scatter_colors, alpha=0.7, s=20)
            ax.set_xlabel("时间 (秒)")
            ax.set_ylabel("置信度")
            ax.set_title("情绪变化曲线")
            ax.set_ylim(0, 1.1)
            plt.tight_layout()
            path = os.path.join(output_dir, "chart_emotion.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            charts.append(path)

        # 3. NSFW 检测结果
        if self.nsfw_results:
            fig, ax = plt.subplots(figsize=(10, 3))
            timestamps = [r["timestamp"] for r in self.nsfw_results]
            scores = [r["score"] for r in self.nsfw_results]
            labels = [r["label"] for r in self.nsfw_results]
            colors_nsfw = ["red" if s > 0.5 else "green" for s in scores]
            ax.scatter(timestamps, scores, c=colors_nsfw, alpha=0.7, s=30)
            ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="警戒线")
            ax.set_xlabel("时间 (秒)")
            ax.set_ylabel("NSFW 分数")
            ax.set_title("NSFW 检测结果")
            ax.set_ylim(0, 1.1)
            ax.legend()
            plt.tight_layout()
            path = os.path.join(output_dir, "chart_nsfw.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            charts.append(path)

        return charts

    def generate_html(self, output_path: str) -> str:
        """生成 HTML 格式报告"""
        elapsed = self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        roles_html = "".join(
            f'<span class="role-tag">{r}</span>'
            for r in sorted(self.unique_roles)
        ) if self.unique_roles else "<em>无识别结果</em>"

        # 关键帧（最多 20 张）
        frames_html = ""
        for i, f in enumerate(self.frames[:20]):
            ts = f["timestamp"]
            roles_str = ", ".join(f"{r['name']}({r['score']:.2f})" for r in f["roles"]) if f["roles"] else "无"
            frames_html += f"""
            <div class="frame-card">
                <img src="data:image/jpeg;base64,{f['image_b64']}" alt="帧 {ts:.1f}s">
                <div class="frame-info">
                    <strong>⏱ {ts:.1f}s</strong> | {roles_str}
                </div>
            </div>"""

        # 角色统计表格
        stats_rows = ""
        if self.appearance_stats:
            total = max(sum(len(v) for v in self.appearance_stats.values()), 1)
            for name, timestamps in sorted(self.appearance_stats.items(),
                                            key=lambda x: len(x[1]), reverse=True):
                count = len(timestamps)
                pct = count / total * 100
                bar_w = int(pct * 2)
                stats_rows += f"""
                <tr>
                    <td>{name}</td>
                    <td>{count}</td>
                    <td>{pct:.1f}%</td>
                    <td><div class="bar" style="width:{bar_w}px"></div></td>
                </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{self.title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #f5f7fa; color: #1a1a2e; }}
.header {{ background: linear-gradient(135deg, #6c5ce7, #a29bfe); color: white; padding: 32px; text-align: center; }}
.header h1 {{ font-size: 24px; }}
.header p {{ opacity: 0.85; margin-top: 8px; font-size: 14px; }}
.container {{ max-width: 960px; margin: 0 auto; padding: 20px; }}
.section {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
.section h2 {{ font-size: 18px; color: #6c5ce7; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 2px solid #f0f0f5; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
.stat-card {{ text-align: center; padding: 16px; background: #f8f9fc; border-radius: 8px; }}
.stat-card .num {{ font-size: 28px; font-weight: bold; color: #6c5ce7; }}
.stat-card .label {{ font-size: 12px; color: #888; margin-top: 4px; }}
.role-tag {{ display: inline-block; background: #6c5ce7; color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; margin: 2px; }}
.frames-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }}
.frame-card {{ border-radius: 8px; overflow: hidden; border: 1px solid #eee; }}
.frame-card img {{ width: 100%; height: 140px; object-fit: cover; }}
.frame-info {{ padding: 8px; font-size: 12px; color: #555; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ background: #f0f0f5; padding: 8px 12px; text-align: left; font-weight: 600; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #f0f0f5; }}
.bar {{ height: 12px; background: linear-gradient(90deg, #6c5ce7, #a29bfe); border-radius: 6px; }}
.chart-img {{ max-width: 100%; height: auto; border-radius: 8px; margin: 8px 0; }}
@media print {{ .section {{ break-inside: avoid; }} }}
</style>
</head>
<body>
<div class="header">
    <h1>{self.title}</h1>
    <p>生成时间: {elapsed}</p>
    <p>源文件: {self.source_file or "未知"}</p>
</div>
<div class="container">

<div class="section">
    <h2>📊 概要统计</h2>
    <div class="stats-grid">
        <div class="stat-card"><div class="num">{self.total_processed}</div><div class="label">处理帧/张</div></div>
        <div class="stat-card"><div class="num">{len(self.unique_roles)}</div><div class="label">识别角色</div></div>
        <div class="stat-card"><div class="num">{sum(len(v) for v in self.appearance_stats.values())}</div><div class="label">识别次数</div></div>
        <div class="stat-card"><div class="num">{len(self.nsfw_results)}</div><div class="label">NSFW 检测</div></div>
    </div>
    <div style="margin-top:12px"><strong>识别角色：</strong>{roles_html}</div>
</div>

<div class="section">
    <h2>📈 角色出场统计</h2>
    <table><thead><tr><th>角色</th><th>出现次数</th><th>占比</th><th>分布</th></tr></thead><tbody>{stats_rows}</tbody></table>
</div>

<div class="section">
    <h2>🖼 关键帧预览</h2>
    <div class="frames-grid">{frames_html}</div>
</div>

</div>
</body>
</html>"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    def generate_pdf(self, output_path: str) -> Optional[str]:
        """生成 PDF 格式报告"""
        if not FPDF_AVAILABLE:
            return None

        pdf = FPDF()
        pdf.add_page()

        # 封面
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 20, self.title, align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"生成时间: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                 align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"源文件: {self.source_file or '未知'}",
                 align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)

        # 统计概要
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "概要统计", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"处理帧/张: {self.total_processed}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 7, f"识别角色数: {len(self.unique_roles)}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 7, f"识别角色: {', '.join(sorted(self.unique_roles))[:80]}",
                 new_x="LMARGIN", new_y="NEXT")

        pdf.output(output_path)
        return output_path
