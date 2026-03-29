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
import json
from PIL import Image, ImageDraw, ImageFont

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    warnings.warn("moviepy未安装，输出视频无音频")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用相对路径
CASCADE_PATH = "./lbpcascade_animeface.xml"
FONT_PATH = "./simhei.ttf"

# ============ 关键词配置 ============
KEYWORD_MAPPING = [
    {"keyword": "塔菲", "db_name": "永雏塔菲"},
    {"keyword": "雏草姬", "db_name": "永雏塔菲"},
    {"keyword": "东雪莲", "db_name": "东雪莲"},
    {"keyword": "罕见", "db_name": "东雪莲"},
    {"keyword": "丛雨", "db_name": "丛雨"},
    {"keyword": "棍母", "db_name": "棍母"},
    {"keyword": "Ayachi", "db_name": "Ayachi_Nene"},
    {"keyword": "Nene", "db_name": "Ayachi_Nene"},
    {"keyword": "绫地宁宁", "db_name": "Ayachi_Nene"},
    {"keyword": "宁宁", "db_name": "Ayachi_Nene"},
    {"keyword": "Neuro", "db_name": "Neuro-sama"},
    {"keyword": "牛肉", "db_name": "Neuro-sama"},
    {"keyword": "Otto", "db_name": "otto"},
    {"keyword": "夏目", "db_name": "ShikiNatsume"},
    {"keyword": "四季夏目", "db_name": "ShikiNatsume"},
    {"keyword": "枣子姐", "db_name": "ShikiNatsume"},
    {"keyword": "Shiki", "db_name": "ShikiNatsume"},
]
DEFAULT_DB_NAME = "全部特征库"
# ===================================

# 加载分类器
if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = "lbpcascade_animeface.xml"
anime_face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if anime_face_cascade.empty():
    raise IOError(f"分类器加载失败，请确保 {CASCADE_PATH} 文件存在")

# 加载模型
print("正在加载FaceNet模型...")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("模型加载成功！")

if not os.path.isfile(FONT_PATH):
    FONT_PATH = None
    warnings.warn("无中文字体，将使用默认字体")

# ============ JSON特征库管理 ============
FEATURES_DIR = "./features"
os.makedirs(FEATURES_DIR, exist_ok=True)

def save_database_to_json(database, json_name):
    if not database:
        return None
    safe_name = "".join(c for c in json_name if c.isalnum() or c in "._- ")
    json_path = os.path.join(FEATURES_DIR, f"{safe_name}.json")
    serializable_db = {name: emb.tolist() for name, emb in database.items()}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_db, f, indent=2, ensure_ascii=False)
    print(f" 特征库已保存至 {json_path}")
    return json_path

def load_database_from_json(json_name):
    safe_name = "".join(c for c in json_name if c.isalnum() or c in "._- ")
    json_path = os.path.join(FEATURES_DIR, f"{safe_name}.json")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            serializable_db = json.load(f)
        database = {name: np.array(emb) for name, emb in serializable_db.items()}
        print(f" 从 {json_path} 加载了 {len(database)} 个角色")
        return database
    except Exception as e:
        print(f" 加载JSON失败: {e}")
        return None

def get_db_name_from_filename(filename):
    filename_lower = filename.lower()
    for mapping in KEYWORD_MAPPING:
        keyword = mapping["keyword"].lower()
        if keyword in filename_lower:
            db_name = mapping["db_name"]
            print(f" 检测到关键词 '{mapping['keyword']}'，加载特征库: {db_name}")
            return db_name
    print(f" 未检测到特定关键词，将加载全部特征库")
    return DEFAULT_DB_NAME

# ============ 图像处理函数 ============
def draw_chinese_text(img, text, position, font_size=20, color=(0,255,0)):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        if FONT_PATH and os.path.isfile(FONT_PATH):
            font = ImageFont.truetype(FONT_PATH, font_size)
        else:
            font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=color[::-1])
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

def extract_features_from_image(image_path):
    """
    从图像提取特征（支持中文路径，使用二进制读取避免编码问题）
    """
    try:
        # 直接二进制读取，绕过 cv2.imread 的路径编码问题
        with open(image_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
    except Exception:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 放宽检测参数，提高检出率
    faces = anime_face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.02,
        minNeighbors=1,       # 降低阈值，更容易检出
        minSize=(20, 20),
        maxSize=(800, 800)
    )
    if len(faces) == 0:
        return None

    # 优先选面积最大的脸
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    x, y, w, h = faces[0]

    face = img[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (160, 160))
    face_tensor = torch.tensor(face_resized).permute(2,0,1).float().unsqueeze(0).to(device)/255.0
    
    with torch.no_grad():
        embedding = resnet(face_tensor).cpu().numpy().flatten()
    return embedding

def build_database(data_root):
    """从图片构建特征库（自动识别单角色/多角色）"""
    database = {}
    data_root = Path(data_root)
    
    if not data_root.exists():
        print(f"❌ 路径不存在: {data_root}")
        return database

    subdirs = [d for d in data_root.iterdir() if d.is_dir()]
    
    # 情况1：有子文件夹 -> 多角色模式
    if subdirs:
        total_persons = 0
        total_images = 0
        print(f"正在从 {data_root} 构建特征库（多角色模式）...")
        
        for person_dir in subdirs:
            person_name = person_dir.name
            embeddings = []
            exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
            img_files = []
            for ext in exts:
                img_files.extend(person_dir.glob(f'*{ext}'))
                img_files.extend(person_dir.glob(f'*{ext.upper()}'))
            
            if not img_files:
                continue
                
            print(f"\n 处理角色: {person_name} ({len(img_files)}张图片)")
            for i, img_file in enumerate(img_files):
                print(f"\r 处理中 [{i+1}/{len(img_files)}]", end="", flush=True)
                emb = extract_features_from_image(str(img_file))
                if emb is not None:
                    embeddings.append(emb)
                    total_images += 1
            
            if embeddings:
                avg_emb = np.mean(embeddings, axis=0)
                database[person_name] = avg_emb
                total_persons += 1
                print(f"  成功提取 {len(embeddings)} 张")
            else:
                print(f" ❌ 无人脸")
        
        print(f"\n📊 总计: {total_persons}个角色, {total_images}张图片")
    
    # 情况2：无子文件夹 -> 单角色模式
    else:
        exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
        img_files = []
        for ext in exts:
            img_files.extend(data_root.glob(f'*{ext}'))
            img_files.extend(data_root.glob(f'*{ext.upper()}'))
        
        if not img_files:
            print(f"⚠️ {data_root} 下没有图片文件")
            return database
        
        person_name = data_root.name
        embeddings = []
        print(f" 单角色模式: {person_name} ({len(img_files)}张图片)")
        
        for i, img_file in enumerate(img_files):
            print(f"\r 处理中 [{i+1}/{len(img_files)}]", end="", flush=True)
            emb = extract_features_from_image(str(img_file))
            if emb is not None:
                embeddings.append(emb)
        
        if embeddings:
            avg_emb = np.mean(embeddings, axis=0)
            database[person_name] = avg_emb
            print(f"\n 成功提取 {len(embeddings)} 张图片")
        else:
            print(f"\n ❌ 没有检测到人脸")
    
    return database

def build_all_database():
    data_path = Path("./data")
    if not data_path.exists():
        print("❌ ./data 文件夹不存在")
        return {}
    return build_database(data_path)

def get_or_build_database(db_name, force_rebuild=False):
    if not force_rebuild:
        db = load_database_from_json(db_name)
        if db is not None:
            return db
    
    if db_name == DEFAULT_DB_NAME:
        print(f" 构建全部特征库...")
        db = build_all_database()
    else:
        db_path = os.path.join("./data", db_name)
        if not os.path.exists(db_path):
            print(f"❌ 特征库文件夹不存在: {db_path}")
            return {}
        print(f" 构建特征库: {db_name}")
        db = build_database(db_path)
    
    if db:
        save_database_to_json(db, db_name)
    return db

# ============ 识别函数 ============
def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-6)

def recognize_face(face_img, database, threshold=0.45):
    face = cv2.resize(face_img, (160,160))
    face_tensor = torch.tensor(face).permute(2,0,1).float().unsqueeze(0).to(device)/255.0
    
    with torch.no_grad():
        emb = resnet(face_tensor).cpu().numpy().flatten()
    
    best_name, best_score = None, 0
    for name, ref_emb in database.items():
        sim = cosine_similarity(emb, ref_emb)
        if sim > best_score:
            best_score = sim
            best_name = name
    
    return (best_name, best_score) if best_score >= threshold else (None, best_score)

# ============ 视频处理 ============
def add_audio_to_video(input_video, output_video, temp_video):
    if not MOVIEPY_AVAILABLE:
        shutil.move(temp_video, output_video)
        return
    try:
        video_clip = VideoFileClip(input_video)
        audio = video_clip.audio
        silent_clip = VideoFileClip(temp_video)
        final_clip = silent_clip.with_audio(audio)
        final_clip.write_videofile(output_video, codec='libx264', audio_codec='aac')
        video_clip.close()
        silent_clip.close()
        final_clip.close()
        if os.path.exists(temp_video):
            os.unlink(temp_video)
    except Exception as e:
        print(f"添加音频失败: {e}")
        shutil.move(temp_video, output_video)

def process_video(source, database, output_path=None, threshold=0.45, skip_frames=1, camera=False):
    if camera:
        cap = cv2.VideoCapture(0)
        if output_path:
            fps = 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        else:
            out = None
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {source}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频信息: {w}x{h}, {fps:.2f}fps, 共{total}帧")
        
        if output_path:
            temp_dir = os.path.join(os.path.dirname(output_path) or ".", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', dir=temp_dir, delete=False).name
            out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        else:
            out = None

    frame_idx = 0
    recognized_count = 0
    recognized_names = set()
    
    print("️ 开始处理... (按 'q' 停止)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if not camera and frame_idx % 30 == 0 and total > 0:
            progress = (frame_idx / total) * 100
            print(f"\r 处理进度: {progress:.1f}%", end="", flush=True)
        
        if frame_idx % skip_frames == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = anime_face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.02,
                minNeighbors=1,      # 视频中也可以适当降低阈值
                minSize=(20, 20),
                maxSize=(800, 800)
            )
            
            for (x, y, w, h) in faces:
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
                
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                    
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                name, score = recognize_face(face_rgb, database, threshold)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if name:
                    txt = f"{name} ({score:.2f})"
                    recognized_count += 1
                    recognized_names.add(name)
                    print(f"\n 识别到: {name} ({score:.2f})")
                else:
                    txt = "Unknown"
                
                frame = draw_chinese_text(frame, txt, (x1, y1-25), 16, (0, 255, 0))
        
        if out:
            out.write(frame)
        else:
            cv2.imshow('Anime Face Recognition', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n️ 用户停止")
                break
        
        frame_idx += 1
    
    print(f"\n处理完成: 共{frame_idx}帧, 识别到{recognized_count}张脸")
    if recognized_names:
        print(f" 识别到的角色: {', '.join(recognized_names)}")
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    if not camera and output_path and 'temp_output' in locals() and os.path.exists(temp_output):
        add_audio_to_video(source, output_path, temp_output)
        print(f" 视频已保存: {output_path}")

# ============ 列出可用的特征库 ============
def list_available_databases():
    data_path = Path("./data")
    if not data_path.exists():
        print("❌ ./data 文件夹不存在")
        return []
    folders = [d.name for d in data_path.iterdir() if d.is_dir()]
    return folders

# ============ 主程序 ============
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='动漫人脸识别系统')
    parser.add_argument('--data', default='./data', help='特征库图片文件夹')
    parser.add_argument('--source', help='视频文件路径或摄像头ID（如果不指定则只执行list）')
    parser.add_argument('--camera', action='store_true', help='使用摄像头模式')
    parser.add_argument('--output', help='输出视频路径')
    parser.add_argument('--threshold', type=float, default=0.45, help='识别阈值 (默认:0.45)')
    parser.add_argument('--skip_frames', type=int, default=1, help='跳帧数 (默认:1)')
    parser.add_argument('--rebuild', action='store_true', help='强制重新构建特征库')
    parser.add_argument('--db_name', help='直接指定特征库名称')
    parser.add_argument('--list', action='store_true', help='列出所有可用的特征库')
    args = parser.parse_args()

    if args.list:
        available_dbs = list_available_databases()
        print("\n 可用的特征库:")
        for db in available_dbs:
            print(f"  - {db}")
        print(f"\n 关键词映射:")
        for mapping in KEYWORD_MAPPING:
            print(f"  - '{mapping['keyword']}' → {mapping['db_name']}")
        exit(0)

    if not args.source and not args.list:
        print("❌ 错误: 必须指定 --source 或使用 --list")
        parser.print_help()
        exit(1)

    if args.db_name:
        db_name = args.db_name
        print(f" 使用指定的特征库: {db_name}")
    elif args.camera:
        db_name = DEFAULT_DB_NAME
        print(" 摄像头模式，使用全部特征库")
    else:
        filename = os.path.basename(args.source)
        db_name = get_db_name_from_filename(filename)
    
    print(f" 目标特征库: {db_name}")
    database = get_or_build_database(db_name, force_rebuild=args.rebuild)
    
    if not database:
        print("❌ 特征库为空，无法进行识别")
        print(" 提示: 请确保 ./data 文件夹下有图片")
        exit(1)
    
    print(f"当前特征库包含 {len(database)} 个角色:")
    for i, name in enumerate(database.keys(), 1):
        print(f"   {i}. {name}")
    
    process_video(
        args.source, 
        database, 
        args.output, 
        args.threshold, 
        args.skip_frames, 
        args.camera
    )
    
    print(" 处理完成！")
