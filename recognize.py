import os
import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from pathlib import Path
import argparse
import tempfile
import warnings
import shutil
import time
from PIL import Image, ImageDraw, ImageFont

# 尝试导入moviepy用于音频处理
try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    warnings.warn(f"moviepy导入失败: {e}。输出视频将不包含音频。如需修复，请检查环境或手动安装：pip install moviepy")

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 Anime Face Cascade 分类器
CASCADE_PATH = "D:/pythonshibie/lbpcascade_animeface.xml"  # 请确保文件存在
if not os.path.isfile(CASCADE_PATH):
    raise FileNotFoundError(f"未找到动漫人脸级联分类器: {CASCADE_PATH}")
anime_face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if anime_face_cascade.empty():
    raise IOError("无法加载 lbpcascade_animeface.xml，请检查文件路径")

# 加载 FaceNet 特征提取模型
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 中文字体路径（请修改为您系统实际存在的字体文件）
FONT_PATH = "D:/pythonshibie/simhei.ttf"  # 例如黑体，可从Windows系统复制或下载
if not os.path.isfile(FONT_PATH):
    # 如果找不到，回退到默认字体（但可能不支持中文）
    FONT_PATH = None
    warnings.warn("未找到中文字体文件，中文显示可能为方块。请下载字体（如simhei.ttf）并修改FONT_PATH。")

def draw_chinese_text(img, text, position, font_size=20, color=(0,255,0)):
    """
    在OpenCV图像上绘制中文文本
    img: OpenCV BGR图像
    text: 要绘制的文本（支持中文）
    position: (x, y) 文本左上角坐标
    font_size: 字体大小
    color: BGR颜色元组
    返回：绘制后的OpenCV图像
    """
    # 将OpenCV BGR图像转换为PIL RGB图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # 加载字体
    if FONT_PATH and os.path.isfile(FONT_PATH):
        font = ImageFont.truetype(FONT_PATH, font_size)
    else:
        # 使用默认字体（可能不支持中文）
        font = ImageFont.load_default()
    # 绘制文本（注意PIL颜色为RGB）
    draw.text(position, text, font=font, fill=color[::-1])  # BGR->RGB
    # 转换回OpenCV BGR图像
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def extract_features_from_image(image_path):
    """从单张图片中提取人脸特征（使用 Anime Face Cascade 检测）"""
    # 使用二进制流读取，避免中文路径问题
    try:
        with open(image_path, 'rb') as f:
            bytes_data = bytearray(f.read())
            numpyarray = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"无法读取图片 {image_path}: {e}")
        return None
    
    if img is None:
        print(f"图片解码失败: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # 提高对比度，有助于检测
    
    # 检测人脸
    faces = anime_face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
    )
    if len(faces) == 0:
        return None
    
    # 取第一张脸
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (160, 160))
    face_tensor = torch.tensor(face_resized).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        embedding = resnet(face_tensor).cpu().numpy().flatten()
    return embedding

def build_database(data_root):
    """遍历数据集文件夹，构建角色特征库（每个角色取平均特征）"""
    database = {}
    data_root = Path(data_root)
    for person_dir in data_root.iterdir():
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name  # 中文名直接保留
        embeddings = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        for img_file in person_dir.glob('*.*'):
            if img_file.suffix.lower() not in valid_extensions:
                continue
            emb = extract_features_from_image(str(img_file))
            if emb is not None:
                embeddings.append(emb)
        if embeddings:
            avg_emb = np.mean(embeddings, axis=0)
            database[person_name] = avg_emb
            print(f"已加载角色：{person_name}，使用 {len(embeddings)} 张图像")
    return database

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(face_img, database, threshold=0.6):
    """对输入的人脸图像（RGB数组）进行识别，返回（名称，相似度）或 (None, 0)"""
    face = cv2.resize(face_img, (160, 160))
    face_tensor = torch.tensor(face).permute(2,0,1).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        emb = resnet(face_tensor).cpu().numpy().flatten()
    
    best_name = None
    best_score = 0
    for name, ref_emb in database.items():
        sim = cosine_similarity(emb, ref_emb)
        if sim > best_score:
            best_score = sim
            best_name = name
    if best_score >= threshold:
        return best_name, best_score
    else:
        return None, best_score

def add_audio_to_video(input_video, output_video, temp_video):
    """将原视频的音频合并到临时无声视频中生成最终输出"""
    if not MOVIEPY_AVAILABLE:
        shutil.move(temp_video, output_video)
        print("moviepy未安装，已生成无声视频。如需音频，请手动使用ffmpeg合并：")
        print(f"ffmpeg -i {input_video} -i {temp_video} -c:v copy -c:a aac -map 0:a? -map 1:v {output_video}")
        return
    try:
        # 读取原视频的音频
        video_clip = VideoFileClip(input_video)
        audio = video_clip.audio
        # 读取无声视频
        silent_clip = VideoFileClip(temp_video)
        # 使用 with_audio 方法（moviepy 2.x 兼容）
        final_clip = silent_clip.with_audio(audio)
        # 写入输出视频
        final_clip.write_videofile(output_video, codec='libx264', audio_codec='aac')
        # 关闭所有剪辑
        video_clip.close()
        silent_clip.close()
        final_clip.close()
        print(f"已生成带音频的视频：{output_video}")
        # 删除临时文件
        os.unlink(temp_video)
    except Exception as e:
        print(f"音频合并失败：{e}，将保留无声视频")
        # 确保所有剪辑已关闭
        for clip in ['video_clip', 'silent_clip', 'final_clip']:
            if clip in locals():
                try:
                    locals()[clip].close()
                except:
                    pass
        # 等待一小段时间确保文件句柄释放
        time.sleep(1)
        shutil.move(temp_video, output_video)

def process_video(source, database, output_path=None, threshold=0.6, skip_frames=2, camera=False):
    """处理视频或摄像头流"""
    # 打开视频源
    if camera:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else 0)
        print(f"摄像头 {source} 已打开")
        if output_path:
            fps = 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"无法打开视频文件：{source}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if output_path:
            # 在输出文件同目录下创建临时文件夹，避免跨盘问题
            output_dir = os.path.dirname(os.path.abspath(output_path))
            temp_dir = os.path.join(output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', dir=temp_dir, delete=False).name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        else:
            out = None
        print(f"视频文件：{source}，总帧数：{total_frames}，分辨率：{width}x{height}，帧率：{fps}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 跳帧处理
        if frame_idx % skip_frames == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = anime_face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48)
            )
            
            for (x, y, w, h) in faces:
                x1, y1, x2, y2 = x, y, x+w, y+h
                # 确保边界在图像内
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                
                # 转为RGB用于特征提取
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                name, score = recognize_face(face_rgb, database, threshold)
                
                # 绘制绿色矩形
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 准备文本
                label_name = name if name else "Unknown"
                label_conf = f"{score:.2f}" if name else ""
                
                # 左上角绘制中文名称（使用自定义函数）
                frame = draw_chinese_text(frame, label_name, (x1, y1-25), font_size=16, color=(0,255,0))
                
                # 右上角绘制置信度（英文数字，可直接用 cv2.putText）
                if label_conf:
                    (tw, th), _ = cv2.getTextSize(label_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.putText(frame, label_conf, (x2 - tw - 5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        if out:
            out.write(frame)
        else:
            cv2.imshow('Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx += 1
        if not camera and frame_idx % 100 == 0:
            print(f"处理进度：{frame_idx}/{total_frames} 帧")

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # 如果是视频文件模式且有输出路径，则处理音频合并
    if not camera and output_path and os.path.exists(temp_output):
        add_audio_to_video(source, output_path, temp_output)
        # 删除临时文件夹（如果为空）
        try:
            os.unlink(temp_output)
            temp_dir = os.path.dirname(temp_output)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='视频/摄像头人物识别，保留原音频')
    parser.add_argument('--data', type=str, default='./data', help='数据集根目录（默认为./data）')
    parser.add_argument('--source', type=str, help='输入源：视频文件路径 或 摄像头ID（如0）')
    parser.add_argument('--video', type=str, help='[已废弃] 请使用 --source 代替')
    parser.add_argument('--camera', action='store_true', help='使用摄像头模式（忽略--video，使用--source指定设备ID，默认为0）')
    parser.add_argument('--output', type=str, default=None, help='输出视频路径（可选，摄像头模式下保存无声录像）')
    parser.add_argument('--threshold', type=float, default=0.6, help='识别阈值')
    parser.add_argument('--skip_frames', type=int, default=2, help='跳帧数（每skip_frames帧处理一帧）')
    args = parser.parse_args()

    if args.video and not args.source:
        args.source = args.video

    if not args.source:
        print("错误：必须指定 --source（视频文件路径或摄像头ID）")
        exit(1)

    camera_mode = args.camera
    if args.source.isdigit() and not camera_mode:
        print("提示：输入源为数字，如需使用摄像头请添加 --camera 参数，否则视为文件名")

    print("正在构建特征库...")
    database = build_database(args.data)
    print(f"特征库构建完成，共 {len(database)} 个角色")

    print("开始处理...")
    process_video(
        source=args.source,
        database=database,
        output_path=args.output,
        threshold=args.threshold,
        skip_frames=args.skip_frames,
        camera=camera_mode
    )
    print("处理完成")
