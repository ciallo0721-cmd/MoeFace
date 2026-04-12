import os
import requests
import time
import random
import urllib.parse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

# ======================== 配置区域 ========================
ROOT_DIR = os.path.join(os.path.dirname(__file__), 'data')
MAX_IMAGES_PER_ROLE = 10          # 每个角色总共要下载多少张
DOWNLOAD_THREADS = 5              # 下载并发数
REQUEST_DELAY = 0.5               # 同一图源内每次请求间隔（秒）

# 各图源的开关及配置
ENABLE_BING = True
ENABLE_GOOGLE = True              # Google 反爬严重，默认关闭，可自行开启测试
ENABLE_BAIDU = True
ENABLE_PIXIV = False               # 如需开启，必须在下方配置 PIXIV_COOKIE
ENABLE_SAFEBOORU = True            # 稳定图库，无 NSFW 过滤

# Pixiv 配置（从浏览器复制 Cookie 字符串）
PIXIV_COOKIE = ""                  # 格式: "PHPSESSID=xxxxx; device_token=xxxxx"
# 如果未填写 Cookie 且 ENABLE_PIXIV=True，会自动跳过 Pixiv

# 通用请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

# 确保根目录存在
if not os.path.exists(ROOT_DIR):
    print(f"错误：找不到目录 {ROOT_DIR}")
    exit(1)

# 获取所有角色文件夹
roles = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]

# 角色特定的搜索后缀（用于扩充关键词）
ROLE_SUFFIXES = {
    "丛雨": ["立绘","壁纸","千恋万花","Murasame","ムラサメ","丛雨丸"],
    "Neuro-sama": ["vtuber","art","fanart","Neuro sama","Neurosama","AI vtuber"],
    "永雏塔菲": ["虚拟主播","立绘","Taffy","Ace Taffy","唐人塔菲"],
    "东雪莲": ["虚拟主播","立绘","東雪蓮","Yukiren","罕见"],
    "ShikiNatsume": ["立绘","壁纸","棗シキ","Shiki Natsume","枣子姐"],
    "棍母": ["棍娘","Gun Mu","电棍母亲","电棍妈妈"],
    "otto": ["吉吉国王","电棍","otto lol","帅,otto"],
    "Ayachi_Nene": ["立绘","壁纸","綾地寧々","Nene Ayachi","桌角战士"]
}

# ======================== 各图源获取函数 ========================

def get_bing_images(keyword, num):
    """Bing 图片搜索"""
    urls = []
    start = 1
    while len(urls) < num:
        search_url = f"https://cn.bing.com/images/search?q={urllib.parse.quote(keyword)}&first={start}"
        try:
            resp = requests.get(search_url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            links = soup.find_all('a', class_='iusc')
            if not links:
                break
            for link in links:
                m = link.get('m')
                if m:
                    try:
                        img_url = json.loads(m).get('murl')
                        if img_url and img_url not in urls:
                            urls.append(img_url)
                            if len(urls) >= num:
                                break
                    except:
                        continue
            start += len(links)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"    [Bing] 解析出错: {e}")
            break
    return urls[:num]

def get_baidu_images(keyword, num):
    """百度图片搜索"""
    urls = []
    pn = 0
    while len(urls) < num:
        search_url = f"https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word={urllib.parse.quote(keyword)}&pn={pn}"
        try:
            resp = requests.get(search_url, headers=HEADERS, timeout=10)
            # 百度图片的图片URL藏在页面里的 objURL 中
            # 简单正则提取
            pattern = r'"objURL":"(https?://[^"]+)"'
            matches = re.findall(pattern, resp.text)
            if not matches:
                break
            for url in matches:
                # 处理反斜杠转义
                url = url.replace('\\/', '/')
                if url and url not in urls:
                    urls.append(url)
                    if len(urls) >= num:
                        break
            pn += 60   # 每页大约60张
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"    [百度] 解析出错: {e}")
            break
    return urls[:num]

def get_google_images(keyword, num):
    """Google 图片搜索（反爬严重，成功率低，仅供参考）"""
    urls = []
    start = 0
    while len(urls) < num:
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(keyword)}&tbm=isch&start={start}"
        try:
            resp = requests.get(search_url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Google 图片的URL通常在 data-src 或 img 标签中
            img_tags = soup.find_all('img')
            for img in img_tags:
                src = img.get('data-src') or img.get('src')
                if src and src.startswith('http') and src not in urls:
                    urls.append(src)
                    if len(urls) >= num:
                        break
            if not img_tags:
                break
            start += 20
            time.sleep(REQUEST_DELAY + random.uniform(0.5, 1.5))
        except Exception as e:
            print(f"    [Google] 解析出错: {e}")
            break
    return urls[:num]

def get_pixiv_images(keyword, num):
    """Pixiv 搜索（需要登录 Cookie）"""
    if not PIXIV_COOKIE:
        return []
    urls = []
    headers = HEADERS.copy()
    headers['Cookie'] = PIXIV_COOKIE
    headers['Referer'] = 'https://www.pixiv.net/'
    # 使用 Pixiv 的 API (非官方)
    # 注意：Pixiv 需要带 user-agent 和 cookie，否则返回403
    page = 1
    per_page = 30
    while len(urls) < num:
        api_url = f"https://www.pixiv.net/ajax/search/artworks/{urllib.parse.quote(keyword)}?word={urllib.parse.quote(keyword)}&order=date_d&mode=all&p={page}"
        try:
            resp = requests.get(api_url, headers=headers, timeout=10)
            data = resp.json()
            if data.get('error'):
                break
            works = data.get('body', {}).get('illustManga', {}).get('data', [])
            if not works:
                break
            for work in works:
                # 获取原图URL
                illust_id = work.get('id')
                if illust_id:
                    # 获取作品详情页获取原图链接
                    detail_url = f"https://www.pixiv.net/ajax/illust/{illust_id}"
                    detail_resp = requests.get(detail_url, headers=headers, timeout=10)
                    detail_data = detail_resp.json()
                    urls_big = detail_data.get('body', {}).get('urls', {}).get('original')
                    if urls_big and urls_big not in urls:
                        urls.append(urls_big)
                        if len(urls) >= num:
                            break
            page += 1
            time.sleep(REQUEST_DELAY * 2)
        except Exception as e:
            print(f"    [Pixiv] 出错: {e}")
            break
    return urls[:num]

def get_safebooru_images(keyword, num):
    """Safebooru 稳定图库（无需登录，只返回安全图片）"""
    urls = []
    # Safebooru 的 API: https://safebooru.org/index.php?page=dapi&s=post&q=index&tags=xxx
    # 返回 XML，需要解析
    page = 0
    limit = 40
    while len(urls) < num:
        api_url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&tags={urllib.parse.quote(keyword)}&pid={page}&limit={limit}"
        try:
            resp = requests.get(api_url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, 'xml')
            posts = soup.find_all('post')
            if not posts:
                break
            for post in posts:
                file_url = post.get('file_url')
                if file_url and file_url.startswith('https://') and file_url not in urls:
                    urls.append(file_url)
                    if len(urls) >= num:
                        break
            page += 1
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            print(f"    [Safebooru] 出错: {e}")
            break
    return urls[:num]

# 所有启用的图源列表（顺序影响合并优先级，靠前的优先使用）
SOURCE_GETTERS = []
if ENABLE_BING:
    SOURCE_GETTERS.append(('Bing', get_bing_images))
if ENABLE_BAIDU:
    SOURCE_GETTERS.append(('百度', get_baidu_images))
if ENABLE_GOOGLE:
    SOURCE_GETTERS.append(('Google', get_google_images))
if ENABLE_PIXIV:
    SOURCE_GETTERS.append(('Pixiv', get_pixiv_images))
if ENABLE_SAFEBOORU:
    SOURCE_GETTERS.append(('Safebooru', get_safebooru_images))

def collect_urls_from_sources(keyword, target_num, suffixes):
    """从多个图源收集图片URL，直到达到 target_num 或所有源尝试完毕"""
    all_urls = []
    # 构造关键词列表：原始关键词 + 关键词+后缀
    keywords_to_try = [keyword] + [f"{keyword} {suffix}" for suffix in suffixes]
    # 记录每个源已经尝试过的关键词索引，避免重复搜索相同关键词（简单起见，按顺序尝试）
    for kw in keywords_to_try:
        if len(all_urls) >= target_num:
            break
        print(f"  尝试关键词: {kw}")
        for src_name, getter in SOURCE_GETTERS:
            if len(all_urls) >= target_num:
                break
            # 每个关键词从该源获取剩余所需数量
            need = target_num - len(all_urls)
            # 为防止单个源返回太多重复或无用数据，最多请求 need * 2 个
            urls = getter(kw, need * 2) if need > 0 else []
            new_urls = [u for u in urls if u not in all_urls]
            all_urls.extend(new_urls)
            if new_urls:
                print(f"    [{src_name}] 获得 {len(new_urls)} 个新URL")
            # 短暂休息，礼貌爬取
            time.sleep(0.2)
    return all_urls[:target_num]

# ======================== 下载函数（与原脚本一致） ========================
def download_image(url, dir_path, base_name):
    """下载单张图片，返回是否成功"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, stream=True)
        if resp.status_code != 200:
            return False
        # 判断扩展名
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
            path = urllib.parse.urlparse(url).path
            guess = os.path.splitext(path)[1].lower()
            ext = guess if guess in ['.jpg','.jpeg','.png','.gif','.webp'] else '.jpg'
        
        file_path = os.path.join(dir_path, base_name + ext)
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(dir_path, f"{base_name}_{counter}{ext}")
            counter += 1
        
        with open(file_path, 'wb') as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
        
        if os.path.getsize(file_path) > 1024:
            return True
        else:
            os.remove(file_path)
            return False
    except Exception:
        return False

# ======================== 主程序 ========================
for role in roles:
    print(f"\n{'='*40}\n处理角色: {role}")
    target_dir = os.path.join(ROOT_DIR, role)
    os.makedirs(target_dir, exist_ok=True)

    # 统计现有图片
    existing = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.gif','.webp'))]
    base_index = len(existing)
    print(f"  已有 {base_index} 张")

    # 获取该角色的后缀列表
    suffixes = ROLE_SUFFIXES.get(role, ["立绘","壁纸","art","fanart","illustration"])

    # 从多个图源收集URL
    urls = collect_urls_from_sources(role, MAX_IMAGES_PER_ROLE, suffixes)
    print(f"共获取到 {len(urls)} 个有效URL")

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
