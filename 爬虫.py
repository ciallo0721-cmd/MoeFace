import os
import requests
import time
import random
import urllib.parse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

# 配置
ROOT_DIR = os.path.join(os.path.dirname(__file__), 'data')
MAX_IMAGES_PER_ROLE = 10
DOWNLOAD_THREADS = 5  # 并发线程数
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://cn.bing.com/'
}

# 确保根目录存在
if not os.path.exists(ROOT_DIR):
    print(f"错误：找不到目录 {ROOT_DIR}")
    exit(1)

# 获取所有角色文件夹
roles = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]

def get_bing_image_urls(keyword, num=10, retry_keywords=None):
    """快速从 Bing 获取图片原图 URL（减少等待）"""
    keywords_to_try = [keyword] + [f"{keyword} {suffix}" for suffix in (retry_keywords or [])]
    all_urls = []
    for kw in keywords_to_try:
        if len(all_urls) >= num:
            break
        print(f"  搜索: {kw}")
        start = 1
        while len(all_urls) < num:
            url = f"https://cn.bing.com/images/search?q={urllib.parse.quote(kw)}&first={start}"
            try:
                resp = requests.get(url, headers=HEADERS, timeout=8)
                soup = BeautifulSoup(resp.text, 'html.parser')
                links = soup.find_all('a', class_='iusc')
                if not links:
                    break
                for link in links:
                    m = link.get('m')
                    if m:
                        try:
                            img_url = json.loads(m).get('murl')
                            if img_url and img_url not in all_urls:
                                all_urls.append(img_url)
                                if len(all_urls) >= num:
                                    break
                        except:
                            continue
                start += len(links)
                time.sleep(0.5)  # 短间隔，提高速度
            except Exception as e:
                print(f"    解析出错: {e}")
                break
    return all_urls[:num]

def download_image(url, dir_path, base_name):
    """直接使用 requests 下载图片，快速失败"""
    try:
        # 发起请求，流式下载
        resp = requests.get(url, headers=HEADERS, timeout=15, stream=True)
        if resp.status_code != 200:
            return False
        
        # 确定扩展名
        content_type = resp.headers.get('content-type', '')
        ext = None
        if 'jpeg' in content_type or 'jpg' in content_type:
            ext = '.jpg'
        elif 'png' in content_type:
            ext = '.png'
        elif 'gif' in content_type:
            ext = '.gif'
        elif 'webp' in content_type:
            ext = '.webp'
        else:
            # 从 URL 中猜扩展名
            path = urllib.parse.urlparse(url).path
            guess = os.path.splitext(path)[1].lower()
            ext = guess if guess in ['.jpg','.jpeg','.png','.gif','.webp'] else '.jpg'
        
        # 避免文件名冲突
        file_path = os.path.join(dir_path, base_name + ext)
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(dir_path, f"{base_name}_{counter}{ext}")
            counter += 1
        
        # 写入文件
        with open(file_path, 'wb') as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
        
        # 验证文件大小
        if os.path.getsize(file_path) > 1024:  # 至少 1KB
            return True
        else:
            os.remove(file_path)
            return False
    except Exception:
        return False

# 主循环
for role in roles:
    print(f"\n{'='*40}\n处理角色: {role}")
    target_dir = os.path.join(ROOT_DIR, role)
    os.makedirs(target_dir, exist_ok=True)

    # 已存在图片数
    existing = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.gif','.webp'))]
    base_index = len(existing)
    print(f"  已有 {base_index} 张")

    # 角色特定关键词
    suffixes = {
        "丛雨": ["立绘","壁纸","千恋万花","Murasame","ムラサメ","丛雨丸"],
        "Neuro-sama": ["vtuber","art","fanart","Neuro sama","Neurosama","AI vtuber"],
        "永雏塔菲": ["虚拟主播","立绘","Taffy","Ace Taffy","唐人塔菲"],
        "东雪莲": ["虚拟主播","立绘","東雪蓮","Yukiren","罕见"],
        "ShikiNatsume": ["立绘","壁纸","棗シキ","Shiki Natsume","枣子姐"],
        "棍母": ["棍娘","Gun Mu","电棍母亲","电棍妈妈"],
        "otto": ["吉吉国王","电棍","otto lol","帅,otto"],
        "Ayachi_Nene": ["立绘","壁纸","綾地寧々","Nene Ayachi","桌角战士"]
    }.get(role, ["立绘","壁纸","art","fanart","illustration"])

    # 获取 URL
    urls = get_bing_image_urls(role, MAX_IMAGES_PER_ROLE, suffixes)
    print(f"获取到 {len(urls)} 个 URL")

    if not urls:
        continue

    # 多线程下载
    success = 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as executor:
        future_to_idx = {}
        for idx, img_url in enumerate(urls):
            base_name = f"{role}_{base_index + idx}"
            future = executor.submit(download_image, img_url, target_dir, base_name)
            future_to_idx[future] = (idx, img_url, base_name)

        for future in as_completed(future_to_idx):
            idx, img_url, base_name = future_to_idx[future]
            try:
                if future.result():
                    success += 1
                    print(f"  ✅ {idx+1}: {base_name}.*")
                else:
                    print(f"  ❌ {idx+1}: 下载失败")
            except Exception as e:
                print(f"  ❌ {idx+1}: 异常 - {e}")

    print(f"角色 {role} 完成，成功 {success} 张")
