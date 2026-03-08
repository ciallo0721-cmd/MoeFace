import os
import requests
import time
import random
import urllib.parse
from threading import Lock

# 数据集根目录
ROOT_DIR = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(ROOT_DIR):
    print(f"错误：找不到目录 {ROOT_DIR}")
    exit(1)

# 获取所有子文件夹（角色名）
roles = [d for d in os.listdir(ROOT_DIR) 
         if os.path.isdir(os.path.join(ROOT_DIR, d))]

# 每个角色下载的图片数量
MAX_IMAGES_PER_ROLE = 10

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://image.so.com/'
}

def get_360_image_urls(keyword, num=10, retry_keywords=None):
    """
    从360图片获取图片URL，会尝试多个关键词
    """
    keywords_to_try = [keyword]
    if retry_keywords:
        keywords_to_try.extend([f"{keyword} {suffix}" for suffix in retry_keywords])
    
    all_urls = []
    for kw in keywords_to_try:
        if len(all_urls) >= num:
            break
        print(f"  尝试关键词: {kw}")
        sn = 0
        kw_urls = []
        while len(kw_urls) < num * 2 and len(all_urls) < num:
            # 360图片搜索接口
            url = f"https://image.so.com/j?q={urllib.parse.quote(kw)}&src=360pic_strong&sn={sn}"
            try:
                resp = requests.get(url, headers=HEADERS, timeout=10)
                if resp.status_code != 200:
                    print(f"    请求失败，状态码：{resp.status_code}")
                    break
                data = resp.json()
                items = data.get('list', [])
                if not items:
                    break
                for item in items:
                    # 取高清图
                    img_url = item.get('qhimg_url') or item.get('img')
                    if img_url and img_url not in all_urls:
                        kw_urls.append(img_url)
                        all_urls.append(img_url)
                        if len(all_urls) >= num:
                            break
                sn += len(items)  # 实际返回数量可能小于30
                time.sleep(random.uniform(1, 2))
            except Exception as e:
                print(f"    解析出错: {e}")
                break
        print(f"  关键词 '{kw}' 获取到 {len(kw_urls)} 个有效URL")
    return all_urls[:num]

def download_image(url, save_path, retries=3):
    """下载单张图片"""
    for i in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                # 检查内容类型是否是图片
                content_type = resp.headers.get('content-type', '')
                if 'image' in content_type:
                    with open(save_path, 'wb') as f:
                        f.write(resp.content)
                    return True
                else:
                    print(f"    内容类型不是图片: {content_type}")
            else:
                print(f"    状态码 {resp.status_code}")
        except Exception as e:
            print(f"    下载异常: {e}, 重试 {i+1}/{retries}")
            time.sleep(2)
    return False

# 主循环
for role in roles:
    print(f"\n{'='*40}")
    print(f"处理角色: {role}")
    target_dir = os.path.join(ROOT_DIR, role)
    os.makedirs(target_dir, exist_ok=True)

    # 获取已存在图片数量，确定起始序号
    existing_files = [f for f in os.listdir(target_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))]
    base_index = len(existing_files)
    print(f"  已存在 {base_index} 张图片")

    # 针对不同角色自定义备选关键词（可自行增删）
    if role == "丛雨":
        suffixes = ["立绘", "壁纸", "千恋万花"]
    elif role == "Neuro-sama":
        suffixes = ["vtuber", "art", "fanart", "character"]
    elif role == "永雏塔菲":
        suffixes = ["虚拟主播", "立绘", "壁纸", "Taffy"]
    elif role == "东雪莲":
        suffixes = ["虚拟主播", "立绘", "壁纸"]
    elif role == "四季夏目":
        suffixes = ["立绘", "壁纸", "Shiki Natsume"]
    elif role == "棍母":
        suffixes = ["棍子", "棍娘"]
    elif role == "电棍":
        suffixes = ["otto", "吉吉国王"]
    elif role == "绫地宁宁":
        suffixes = ["立绘", "壁纸", "Ayachi Nene", "宁宁"]
    else:
        suffixes = ["立绘", "壁纸", "art", "vtuber"]

    # 获取图片URL
    urls = get_360_image_urls(role, MAX_IMAGES_PER_ROLE, suffixes)
    print(f"总共获取到 {len(urls)} 个图片URL")

    if not urls:
        print("未找到图片，跳过")
        continue

    # 下载图片
    lock = Lock()
    success_count = 0
    for idx, img_url in enumerate(urls):
        # 提取扩展名
        ext = os.path.splitext(img_url.split('?')[0])[1].lower()
        if ext not in ('.jpg', '.jpeg', '.png', '.gif', '.webp'):
            ext = '.jpg'  # 默认

        # 生成不重复的文件名
        with lock:
            while True:
                new_name = f"{role}_{base_index + success_count}{ext}"
                save_path = os.path.join(target_dir, new_name)
                if not os.path.exists(save_path):
                    break
                success_count += 1  # 如果已存在，跳过该序号

        print(f"下载 {idx+1}/{len(urls)}: {img_url[:60]}...")
        if download_image(img_url, save_path):
            success_count += 1
            print(f"  ✅ 成功: {new_name}")
        else:
            print(f"  ❌ 失败")

        time.sleep(random.uniform(1, 2))  # 下载间隔

    print(f"角色 {role} 完成，成功下载 {success_count} 张新图片")
