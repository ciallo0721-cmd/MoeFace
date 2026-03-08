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

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    warnings.warn("moviepy未安装，输出视频无音频")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路径保持你原来的
CASCADE_PATH = "./lbpcascade_animeface.xml"
FONT_PATH = "./simhei.ttf"

anime_face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if anime_face_cascade.empty():
    raise IOError("分类器加载失败")

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

if not os.path.isfile(FONT_PATH):
    FONT_PATH = None
    warnings.warn("无中文字体")

def draw_chinese_text(img, text, position, font_size=20, color=(0,255,0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    if FONT_PATH and os.path.isfile(FONT_PATH):
        font = ImageFont.truetype(FONT_PATH, font_size)
    else:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def extract_features_from_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            bytes_data = bytearray(f.read())
            numpyarray = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)
    except:
        return None
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 🔴 修复1：去掉直方图均衡化，避免大脸变暗/丢失
    # gray = cv2.equalizeHist(gray)

    # 🔴 修复2：放宽检测参数，能检测大脸
    faces = anime_face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,    # 更慢更准
        minNeighbors=3,      # 从5降到3
        minSize=(30, 30),    # 从48降到30
        maxSize=(500, 500)   # 新增：允许超大脸
    )
    if len(faces) == 0:
        return None

    # 🔴 修复3：优先选面积最大的脸（中间大塔菲）
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
    database = {}
    data_root = Path(data_root)
    for person_dir in data_root.iterdir():
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        embeddings = []
        exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
        for img_file in person_dir.glob('*.*'):
            if img_file.suffix.lower() not in exts:
                continue
            emb = extract_features_from_image(str(img_file))
            if emb is not None:
                embeddings.append(emb)
        if embeddings:
            avg_emb = np.mean(embeddings, axis=0)
            database[person_name] = avg_emb
            print(f"加载：{person_name}，{len(embeddings)}张")
    return database

def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-6)

def recognize_face(face_img, database, threshold=0.5):  # 🔴 修复4：阈值降到0.5
    face = cv2.resize(face_img, (160,160))
    face_tensor = torch.tensor(face).permute(2,0,1).float().unsqueeze(0).to(device)/255.0
    with torch.no_grad():
        emb = resnet(face_tensor).cpu().numpy().flatten()
    best_name, best_score = None, 0
    for name, ref_emb in database.items():
        sim = cosine_similarity(emb, ref_emb)
        if sim>best_score:
            best_score=sim
            best_name=name
    return (best_name,best_score) if best_score>=threshold else (None,best_score)

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
        os.unlink(temp_video)
    except:
        shutil.move(temp_video, output_video)

def process_video(source, database, output_path=None, threshold=0.5, skip_frames=1, camera=False):
    # 🔴 修复5：跳帧设为1，不跳帧，保证每帧都检测
    if camera:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else 0)
        if output_path:
            fps=30
            w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        else: out=None
    else:
        cap=cv2.VideoCapture(source)
        if not cap.isOpened():return
        fps=cap.get(cv2.CAP_PROP_FPS)
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if output_path:
            temp_dir=os.path.join(os.path.dirname(output_path),"temp")
            os.makedirs(temp_dir,exist_ok=True)
            temp_output=tempfile.NamedTemporaryFile(suffix='.mp4',dir=temp_dir,delete=False).name
            out=cv2.VideoWriter(temp_output,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        else:out=None

    frame_idx=0
    while True:
        ret,frame=cap.read()
        if not ret:break
        if frame_idx%skip_frames==0:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # 🔴 视频也去掉均衡化
            # gray=cv2.equalizeHist(gray)
            faces=anime_face_cascade.detectMultiScale(
                gray,1.05,3,minSize=(30,30),maxSize=(500,500)
            )
            for (x,y,w,h) in faces:
                x1,y1,x2,y2=x,y,x+w,y+h
                x1,y1=max(0,x1),max(0,y1)
                x2,y2=min(frame.shape[1],x2),min(frame.shape[0],y2)
                face=frame[y1:y2,x1:x2]
                if face.size==0:continue
                face_rgb=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
                name,score=recognize_face(face_rgb,database,threshold)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                txt=name if name else "Unknown"
                frame=draw_chinese_text(frame,txt,(x1,y1-25),16,(0,255,0))
                if name:
                    cv2.putText(frame,f"{score:.2f}",(x2-40,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        if out:out.write(frame)
        else:
            cv2.imshow('Recognition',frame)
            if cv2.waitKey(1)&0xFF==ord('q'):break
        frame_idx+=1
    cap.release()
    if out:out.release()
    cv2.destroyAllWindows()
    if not camera and output_path and 'temp_output' in locals() and os.path.exists(temp_output):
        add_audio_to_video(source,output_path,temp_output)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='./data')
    parser.add_argument('--source',required=True)
    parser.add_argument('--camera',action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--threshold',type=float,default=0.5)
    parser.add_argument('--skip_frames',type=int,default=1)
    args=parser.parse_args()
    print("构建特征库...")
    db=build_database(args.data)
    process_video(args.source,db,args.output,args.threshold,args.skip_frames,args.camera)
    print("完成")
