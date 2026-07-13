"""
爬虫.py — MoeFace 动漫人脸识别项目配套：动漫/VTuber 图片采集工具
基于 Bing 图片搜索的二次元角色图片爬虫
用于构建动漫人脸识别、VTuber 识别训练数据集和角色特征库
"""

import os
import sys
import requests
import time
import random
import urllib.parse
import json
import re
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

# ======================== 配置区域 ========================
DOWNLOAD_THREADS = 5
REQUEST_DELAY = 0.5
REQUEST_TIMEOUT = (5, 4000)          # (连接超时, 读取超时) 单位秒

ENABLE_BING = True
ENABLE_GOOGLE = True
ENABLE_BAIDU = True
ENABLE_PIXIV = False
ENABLE_SAFEBOORU = True

PIXIV_COOKIE = ""

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

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

# ======================== 日志函数 ========================
def log(msg):
    """带时间戳的日志"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ======================== 加载负面词列表 ========================
NEGATIVE_WORDS_FILE = "./cname/pachong.json"
negative_words = []

def load_negative_words():
    """从JSON文件加载负面词列表，返回字符串列表"""
    global negative_words
    if not os.path.exists(NEGATIVE_WORDS_FILE):
        log(f"警告: 负面词文件 {NEGATIVE_WORDS_FILE} 不存在，将不进行内容过滤")
        return []
    try:
        with open(NEGATIVE_WORDS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            words = data
        elif isinstance(data, dict) and 'words' in data:
            words = data['words']
        else:
            words = []
        # 统一转为小写，便于匹配
        negative_words = [str(w).lower() for w in words if w]
        log(f"已加载 {len(negative_words)} 个负面词")
        return negative_words
    except json.JSONDecodeError as e:
        log(f"负面词文件 JSON 格式错误: {e}")
        log(f"请检查文件 {NEGATIVE_WORDS_FILE} 的格式，应为 ['词1', '词2'] 或 {{'words': ['词1', '词2']}}")
        return []
    except Exception as e:
        log(f"加载负面词文件失败: {e}")
        return []

def contains_negative_word(text):
    """检查文本中是否包含任意负面词（不区分大小写）"""
    if not negative_words:
        return False
    lower_text = text.lower()
    return any(word in lower_text for word in negative_words)

# 程序启动时加载负面词
load_negative_words()

# ======================== 图片内容审核系统 ========================
# 在保存图片前进行内容安全检测，防止爬取到CSAM、虐待、血腥暴力等内容

# 审核阈值配置
REVIEW_ENABLED = True                # 是否启用审核
NSFW_THRESHOLD = 0.75                 # NSFW 阈值（0~1），超过则拦截（ML模型已训练，不易误判）
GORE_THRESHOLD = 0.75                 # 血腥/暴力阈值（0~1），超过则拦截（提高阈值避免误伤红色系角色）
DARK_THRESHOLD = 0.80                 # 异常黑暗/恐怖阈值（0~1），超过则拦截

# 审核统计
review_stats = {"checked": 0, "passed": 0, "blocked_nsfw": 0, "blocked_gore": 0,
                "blocked_dark": 0, "blocked_other": 0}

_nsfw_detector = None                 # 全局 NSFW 检测器实例


def _get_nsfw_detector():
    """懒加载返回全局 NSFW 检测器"""
    global _nsfw_detector
    if _nsfw_detector is None:
        try:
            # 从 modules/nsfw.py 加载检测器
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from modules.nsfw import NSFWDetector
            _nsfw_detector = NSFWDetector()
            _nsfw_detector.ensure_initialized(log_fn=log)
            log("✅ NSFW 检测器就绪")
        except Exception as e:
            log(f"⚠️ NSFW 检测器加载失败（降级为纯 CV 分析）: {e}")
            _nsfw_detector = False  # False = 加载失败，降级
    return _nsfw_detector


def review_image(file_path: str) -> dict:
    """
    审核单张图片的内容安全性。

    返回:
        {"pass": True/False, "reason": str, "scores": {...}}
    """
    import cv2
    import numpy as np

    global review_stats
    review_stats["checked"] += 1

    result = {"pass": True, "reason": "", "scores": {}}

    try:
        with open(file_path, "rb") as f:
            img_data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            result["pass"] = False
            result["reason"] = "图片解码失败"
            return result

        h, w = img.shape[:2]
        if h == 0 or w == 0:
            result["pass"] = False
            result["reason"] = "图片尺寸无效"
            return result

        # ── 1. NSFW 模型检测（性/裸露内容） ────────────────────────────
        nsfw_score = 0.0
        detector = _get_nsfw_detector()
        if detector and detector is not False:
            try:
                ns = detector._detect_visual_nsfw(img)
                nsfw_score = ns[0]  # (score, label)
            except Exception:
                pass

        result["scores"]["nsfw"] = round(nsfw_score, 4)
        if nsfw_score > NSFW_THRESHOLD:
            review_stats["blocked_nsfw"] += 1
            result["pass"] = False
            result["reason"] = f"NSFW 内容 (score={nsfw_score:.2f})"
            return result

        # ── 2. 血腥/暴力检测（强红色区域分析） ────────────────────────
        gore_score = _detect_gore(img)
        result["scores"]["gore"] = round(gore_score, 4)
        if gore_score > GORE_THRESHOLD:
            review_stats["blocked_gore"] += 1
            result["pass"] = False
            result["reason"] = f"血腥/暴力内容 (score={gore_score:.2f})"
            return result

        # ── 3. 异常黑暗/惊悚检测 ──────────────────────────────────────
        dark_score = _detect_dark_disturbing(img)
        result["scores"]["dark"] = round(dark_score, 4)
        if dark_score > DARK_THRESHOLD:
            review_stats["blocked_dark"] += 1
            result["pass"] = False
            result["reason"] = f"异常图像 (score={dark_score:.2f})"
            return result

        review_stats["passed"] += 1
        return result

    except Exception as e:
        review_stats["blocked_other"] += 1
        return {"pass": False, "reason": f"审核异常: {e}", "scores": {}}


def _detect_gore(img_bgr):
    """
    血腥/暴力内容检测。
    分析红色通道饱和度 + 暗红色块覆盖比例 + 边缘尖锐度。
    注意：避免误伤红色系动漫角色（红衣/红发/红背景）。
    通过检查红色区域的纹理均匀度来区分血迹 vs 衣物。
    返回 0~1 的 gore 分数。
    """
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return 0.0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 计算红色通道占 RGB 的比例
    r, g, b = img_rgb[:, :, 0].astype(float), img_rgb[:, :, 1].astype(float), img_rgb[:, :, 2].astype(float)
    total = r + g + b + 1e-6
    r_ratio = r / total

    # 暗红色区域：红色占比 > 0.5 且亮度较低（深色血渍区域）
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    dark_red_mask = (r_ratio > 0.50) & (gray < 100)
    dark_red_area = float(np.sum(dark_red_mask)) / (h * w)

    # 高饱和度红色区域（新鲜血迹）
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # HSV 中红色分布在 H=0附近 和 H=170附近
    red_mask1 = cv2.inRange(hsv, (0, 40, 40), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 40, 40), (180, 255, 255))
    bright_red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    bright_red_area = float(np.count_nonzero(bright_red_mask)) / (h * w)

    # 边缘尖锐度分析（暴力场景常有锯齿状边缘）
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edge_ratio = float(np.count_nonzero(edges)) / (h * w)

    # ── 抗误伤：检查红色区域的纹理均匀度 ────────────────────────────
    # 衣物/角色皮肤的红色区域纹理均匀，血迹则纹理杂乱
    texture_penalty = 0.0
    if bright_red_area > 0.15:
        # 取红色区域的局部方差
        bright_red_mask_8u = bright_red_mask.astype(np.uint8)
        mean_val = cv2.mean(gray.astype(np.uint8), mask=bright_red_mask_8u)[0]
        std_val = cv2.meanStdDev(gray.astype(np.uint8), mask=bright_red_mask_8u)[1][0][0]
        # 血迹通常亮度差异大（std 高），衣物较均匀（std 低）
        if std_val < 30:
            texture_penalty = 0.15  # 纹理均匀 → 大概率是衣物/背景，降低分数

    # 综合评分
    gore_score = 0.0
    gore_score += dark_red_area * 0.5          # 暗红色区域权重（降低以免误伤）
    gore_score += bright_red_area * 0.25        # 鲜红色区域权重
    gore_score += min(0.25, edge_ratio * 0.6)   # 边缘尖锐度（上限 0.25）
    gore_score = max(0, gore_score - texture_penalty)  # 减去纹理均匀的折减

    return min(1.0, gore_score)


def _detect_dark_disturbing(img_bgr):
    """
    异常黑暗/惊悚内容检测。
    分析整体亮度分布 + 对比度异常 + 偏色。
    返回 0~1 的 dark/disturbing 分数。
    """
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return 0.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray.flatten()

    # 1. 极暗像素比例（像素值 < 30）
    dark_ratio = float(np.sum(pixels < 30)) / len(pixels)

    # 2. 平均亮度
    mean_brightness = float(np.mean(pixels)) / 255.0

    # 3. 亮度标准差（高对比度 = 可能异常）
    std_brightness = float(np.std(pixels)) / 255.0

    # 4. 偏色检测：纯黑白或严重偏色的图像可能异常
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # 饱和度极低的像素（饱和度 < 20）
    saturation = hsv[:, :, 1].astype(float)
    low_sat_ratio = float(np.sum(saturation < 20)) / (h * w)

    # 综合评分
    dark_score = 0.0
    dark_score += dark_ratio * 0.5               # 极暗区域权重
    if mean_brightness < 0.3:
        dark_score += (0.3 - mean_brightness) * 0.8  # 整体偏暗
    # 异常：非常暗但又有极高对比度
    if mean_brightness < 0.35 and std_brightness > 0.2:
        dark_score += 0.25
    # 纯黑白/低饱和度 + 极暗 = 异常
    if low_sat_ratio > 0.7 and dark_ratio > 0.4:
        dark_score += 0.15

    return min(1.0, dark_score)


def _detect_exposure(img_bgr):
    """
    大面积裸露/暴露度检测。
    基于肤色区域占比 + 大面积相似肤色的连续区域分析。
    返回 0~1 的 exposure 分数。
    """
    import cv2
    import numpy as np

    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return 0.0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 多种肤色范围
    skin_mask1 = cv2.inRange(hsv, (0, 15, 50), (20, 170, 255))
    skin_mask2 = cv2.inRange(hsv, (165, 15, 50), (180, 170, 255))
    # 较浅肤色
    skin_mask3 = cv2.inRange(hsv, (0, 0, 100), (30, 80, 255))
    skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
    skin_mask = cv2.bitwise_or(skin_mask, skin_mask3)

    skin_ratio = float(np.count_nonzero(skin_mask)) / (h * w)

    # 检查裸露区域的连续性（大面积连续肤色 = 更可能为裸露）
    # 使用形态学操作分析
    kernel = np.ones((50, 50), np.uint8)
    large_regions = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    large_skin_ratio = float(np.count_nonzero(large_regions)) / (h * w)

    # 肤色比例越高 + 连续区域越大 → 暴露度越高
    score = skin_ratio * 0.4 + large_skin_ratio * 0.6
    return min(1.0, score)


def print_review_stats():
    """打印审核统计"""
    total = review_stats["checked"]
    if total == 0:
        log("📊 审核统计: 无图片被审核")
        return
    blocked = total - review_stats["passed"]
    log("─" * 40)
    log(f"📊 审核统计:")
    log(f"   检查: {total} 张")
    log(f"   通过: {review_stats['passed']} 张")
    log(f"   拦截: {blocked} 张")
    if review_stats["blocked_nsfw"]:
        log(f"     ├─ NSFW/色情: {review_stats['blocked_nsfw']}")
    if review_stats["blocked_gore"]:
        log(f"     ├─ 血腥/暴力: {review_stats['blocked_gore']}")
    if review_stats["blocked_dark"]:
        log(f"     ├─ 异常图像: {review_stats['blocked_dark']}")
    if review_stats["blocked_other"]:
        log(f"     └─ 其他异常: {review_stats['blocked_other']}")
    log("─" * 40)

# ======================== 图源函数 ========================
def get_bing_images(keyword, num):
    log(f"Bing 开始搜索: {keyword}")
    urls = []
    start = 1
    max_pages = 3  # 最多翻3页
    page = 0
    while len(urls) < num and page < max_pages:
        search_url = f"https://cn.bing.com/images/search?q={urllib.parse.quote(keyword)}&first={start}"
        try:
            log(f"  Bing 请求第{page+1}页: {search_url[:80]}...")
            resp = requests.get(search_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            soup = BeautifulSoup(resp.text, 'html.parser')
            links = soup.find_all('a', class_='iusc')
            if not links:
                log(f"  Bing 第{page+1}页没有找到链接")
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
            page += 1
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            log(f"  Bing 出错: {e}")
            break
    log(f"Bing 结束，获得 {len(urls)} 个URL")
    return urls[:num]

def get_baidu_images(keyword, num):
    log(f"百度 开始搜索: {keyword}")
    urls = []
    pn = 0
    max_pages = 3
    page = 0
    while len(urls) < num and page < max_pages:
        search_url = f"https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word={urllib.parse.quote(keyword)}&pn={pn}"
        try:
            log(f"  百度请求第{page+1}页: {search_url[:80]}...")
            resp = requests.get(search_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            pattern = r'"objURL":"(https?://[^"]+)"'
            matches = re.findall(pattern, resp.text)
            if not matches:
                log(f"  百度第{page+1}页没有找到图片链接")
                break
            for url in matches:
                url = url.replace('\\/', '/')
                if url and url not in urls:
                    urls.append(url)
                    if len(urls) >= num:
                        break
            pn += 60
            page += 1
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            log(f"  百度出错: {e}")
            break
    log(f"百度结束，获得 {len(urls)} 个URL")
    return urls[:num]

def get_google_images(keyword, num):
    log(f"Google 开始搜索: {keyword}")
    urls = []
    start = 0
    max_pages = 2
    page = 0
    while len(urls) < num and page < max_pages:
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(keyword)}&tbm=isch&start={start}"
        try:
            log(f"  Google请求第{page+1}页")
            resp = requests.get(search_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            soup = BeautifulSoup(resp.text, 'html.parser')
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
            page += 1
            time.sleep(REQUEST_DELAY + random.uniform(0.5, 1.5))
        except Exception as e:
            log(f"  Google出错: {e}")
            break
    log(f"Google结束，获得 {len(urls)} 个URL")
    return urls[:num]

def get_pixiv_images(keyword, num):
    if not PIXIV_COOKIE:
        log("Pixiv 未配置Cookie，跳过")
        return []
    log(f"Pixiv 开始搜索: {keyword}")
    urls = []
    headers = HEADERS.copy()
    headers['Cookie'] = PIXIV_COOKIE
    headers['Referer'] = 'https://www.pixiv.net/'
    page = 1
    max_pages = 2
    while len(urls) < num and page <= max_pages:
        api_url = f"https://www.pixiv.net/ajax/search/artworks/{urllib.parse.quote(keyword)}?word={urllib.parse.quote(keyword)}&order=date_d&mode=all&p={page}"
        try:
            log(f"  Pixiv请求第{page}页")
            resp = requests.get(api_url, headers=headers, timeout=REQUEST_TIMEOUT)
            data = resp.json()
            if data.get('error'):
                break
            works = data.get('body', {}).get('illustManga', {}).get('data', [])
            if not works:
                break
            for work in works:
                illust_id = work.get('id')
                if illust_id:
                    detail_url = f"https://www.pixiv.net/ajax/illust/{illust_id}"
                    detail_resp = requests.get(detail_url, headers=headers, timeout=REQUEST_TIMEOUT)
                    detail_data = detail_resp.json()
                    urls_big = detail_data.get('body', {}).get('urls', {}).get('original')
                    if urls_big and urls_big not in urls:
                        urls.append(urls_big)
                        if len(urls) >= num:
                            break
            page += 1
            time.sleep(REQUEST_DELAY * 2)
        except Exception as e:
            log(f"  Pixiv出错: {e}")
            break
    log(f"Pixiv结束，获得 {len(urls)} 个URL")
    return urls[:num]

def get_safebooru_images(keyword, num):
    log(f"Safebooru 开始搜索: {keyword}")
    urls = []
    page = 0
    max_pages = 3
    while len(urls) < num and page < max_pages:
        #https://safebooru.org/index.php?page=post&s=list&tags=ace+taffy
        api_url = f"https://safebooru.org/index.php?page={page}&s=list&tags={urllib.parse.quote(keyword)}"
        try:
            log(f"  Safebooru请求第{page+1}页")
            resp = requests.get(api_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            soup = BeautifulSoup(resp.text, 'xml')
            posts = soup.find_all('post')
            if not posts:
                log(f"  Safebooru第{page+1}页没有post")
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
            log(f"  Safebooru出错: {e}")
            break
    log(f"Safebooru结束，获得 {len(urls)} 个URL")
    return urls[:num]

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
    all_urls = []
    keywords_to_try = [keyword] + [f"{keyword} {suffix}" for suffix in suffixes]
    for kw in keywords_to_try:
        if len(all_urls) >= target_num:
            break
        log(f"尝试关键词: {kw}")
        for src_name, getter in SOURCE_GETTERS:
            if len(all_urls) >= target_num:
                break
            need = target_num - len(all_urls)
            log(f"  使用图源 {src_name} 获取最多 {need*2} 个URL...")
            urls = getter(kw, need * 2) if need > 0 else []
            # 过滤包含负面词的URL
            filtered_urls = []
            for u in urls:
                if contains_negative_word(u):
                    log(f"    负面词过滤跳过URL: {u[:80]}...")
                elif u not in all_urls:
                    filtered_urls.append(u)
            new_urls = filtered_urls
            all_urls.extend(new_urls)
            log(f"    {src_name} 新增 {len(new_urls)} 个URL（共获得{len(urls)}，过滤掉{len(urls)-len(new_urls)}个），累计 {len(all_urls)}")
            time.sleep(0.2)
    return all_urls[:target_num]

def download_image(url, dir_path, base_name):
    # 下载前检查负面词（双重保险）
    if contains_negative_word(url):
        log(f"下载跳过（负面词）: {url[:80]}...")
        return False
    try:
        resp = requests.get(url, headers=HEADERS, timeout=(5, 15), stream=True)
        if resp.status_code != 200:
            return False
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
        
        if os.path.getsize(file_path) <= 1024:
            os.remove(file_path)
            return False

        # ── 图片内容审核 ──────────────────────────────────────────────
        if REVIEW_ENABLED:
            review = review_image(file_path)
            if not review["pass"]:
                reason = review.get("reason", "内容违规")
                log(f"  🚫 审核拦截 {base_name}: {reason}")
                os.remove(file_path)
                return False
            sr = review.get("scores", {})
            if sr.get("nsfw", 0) > 0.3 or sr.get("gore", 0) > 0.3:
                log(f"  👀 审核观察 {base_name}: nsfw={sr.get('nsfw',0):.2f} "
                    f"gore={sr.get('gore',0):.2f}")

        return True
    except Exception:
        # 清理不完整文件
        if 'file_path' in locals() and os.path.exists(file_path):
            try: os.remove(file_path)
            except: pass
        return False

# ======================== 修改后的 get_roles_from_dir 函数 ========================
def get_roles_from_dir(root_dir):
    """
    从目录获取角色列表
    支持空目录：如果目录为空或没有图片，自动将根目录名作为角色名创建文件夹
    """
    if not os.path.isdir(root_dir):
        return []
    
    items = os.listdir(root_dir)
    subdirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
    
    # 情况1：有子文件夹，每个子文件夹作为一个角色
    if subdirs:
        roles = []
        for sub in subdirs:
            role_path = os.path.join(root_dir, sub)
            roles.append((sub, role_path))
        return roles
    
    # 情况2：当前目录有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
    has_images = any(f.lower().endswith(image_extensions) for f in items if os.path.isfile(os.path.join(root_dir, f)))
    if has_images:
        role_name = os.path.basename(root_dir)
        return [(role_name, root_dir)]
    
    # 情况3：目录为空或没有图片 - 直接使用当前目录，不再创建嵌套子文件夹
    role_name = os.path.basename(root_dir)
    # 如果根目录名无效或为空，使用默认名称
    if not role_name or role_name == '':
        role_name = "默认角色"
    print(f"目录为空，直接使用当前目录: {root_dir}")
    return [(role_name, root_dir)]

def crawl_all_roles(root_dir, max_images_per_role):
    if not os.path.exists(root_dir):
        print(f"错误：目录不存在 {root_dir}")
        return
    if not os.path.isdir(root_dir):
        print(f"错误：路径不是目录 {root_dir}")
        return
    
    roles = get_roles_from_dir(root_dir)
    if not roles:
        print(f"警告：在 {root_dir} 下没有找到角色文件夹或图片，且无法自动创建角色")
        return
    
    print(f"共找到 {len(roles)} 个角色：{', '.join([name for name, _ in roles])}")
    
    for role_name, role_path in roles:
        print(f"\n{'='*40}\n处理角色: {role_name}")
        target_dir = role_path
        os.makedirs(target_dir, exist_ok=True)
        
        existing = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.gif','.webp'))]
        base_index = len(existing)
        print(f"  已有 {base_index} 张")
        
        suffixes = ROLE_SUFFIXES.get(role_name, ["立绘","壁纸","art","fanart","illustration"])
        print(f"  搜索关键词: {role_name}, 后缀: {suffixes}")
        urls = collect_urls_from_sources(role_name, max_images_per_role, suffixes)
        print(f"共获取到 {len(urls)} 个有效URL（已过滤负面词）")
        
        if not urls:
            print("  没有获取到任何URL，跳过下载")
            continue
        
        success = 0
        with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as executor:
            future_to_idx = {}
            for idx, img_url in enumerate(urls):
                base_name = f"{role_name}_{base_index + idx}"
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
        
        print(f"角色 {role_name} 完成，成功 {success} 张")
    print("\n所有角色处理完毕！")
    print_review_stats()


def batch_review_directory(review_dir: str, delete_bad=False):
    """
    对已有图片目录进行批量内容审核。
    扫描所有图片，标记/删除违规内容。
    返回 (total, bad_count)
    
    参数:
        review_dir: 要扫描的目录
        delete_bad: 是否自动删除违规图片（默认 False，仅报告）
    """
    import cv2
    import numpy as np
    
    if not os.path.isdir(review_dir):
        log(f"❌ 目录不存在: {review_dir}")
        return 0, 0
    
    image_exts = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
    images = []
    for root, dirs, files in os.walk(review_dir):
        for f in files:
            if f.lower().endswith(image_exts):
                images.append(os.path.join(root, f))
    
    if not images:
        log(f"📂 目录中没有图片: {review_dir}")
        return 0, 0
    
    log(f"📂 开始扫描 {len(images)} 张图片...")
    log(f"{'='*50}")
    
    global review_stats
    review_stats = {"checked": 0, "passed": 0, "blocked_nsfw": 0, "blocked_gore": 0,
                    "blocked_dark": 0, "blocked_other": 0}
    
    bad_files = []
    for i, img_path in enumerate(images):
        rel = os.path.relpath(img_path, review_dir)
        
        # 直接使用 review_image 审核
        result = review_image(img_path)
        
        if not result["pass"]:
            reason = result.get("reason", "违规")
            scores = result.get("scores", {})
            score_str = " ".join(f"{k}={v:.2f}" for k, v in scores.items() if v > 0)
            log(f"  🚫 [{i+1}/{len(images)}] {rel} → {reason} ({score_str})")
            bad_files.append((img_path, result))
            if delete_bad:
                try:
                    os.remove(img_path)
                    log(f"     🗑️ 已删除")
                except Exception as e:
                    log(f"     ⚠️ 删除失败: {e}")
        else:
            if (i+1) % 50 == 0:
                log(f"  ✅ [{i+1}/{len(images)}] 已扫描 {i+1} 张...")
    
    print_review_stats()
    
    if bad_files:
        log(f"\n📋 违规文件列表 ({len(bad_files)} 张):")
        for img_path, result in bad_files:
            rel = os.path.relpath(img_path, review_dir)
            log(f"  {rel}  ({result.get('reason', '')})")
        log(f"\n💡 提示: 使用 delete_bad=True 参数可自动删除违规图片")
    
    return len(images), len(bad_files)


# ======================== GUI 界面 ========================
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    def write(self, msg):
        self.text_widget.after(0, self._insert, msg)
    def _insert(self, msg):
        self.text_widget.insert(tk.END, msg)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()
    def flush(self):
        pass

class CrawlerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("简陋图片爬虫")
        self.root.geometry("700x500")
        self.running = False
        self.original_stdout = sys.stdout
        
        frame_path = tk.Frame(self.root)
        frame_path.pack(pady=10, padx=10, fill=tk.X)
        tk.Label(frame_path, text="目标文件夹:").pack(side=tk.LEFT, padx=5)
        self.path_var = tk.StringVar()
        self.entry_path = tk.Entry(frame_path, textvariable=self.path_var, width=50)
        self.entry_path.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.btn_browse = tk.Button(frame_path, text="浏览", command=self.browse_folder)
        self.btn_browse.pack(side=tk.LEFT, padx=5)
        
        frame_num = tk.Frame(self.root)
        frame_num.pack(pady=5, padx=10, fill=tk.X)
        tk.Label(frame_num, text="每个角色图片数:").pack(side=tk.LEFT, padx=5)
        self.num_var = tk.StringVar(value="10")
        self.entry_num = tk.Entry(frame_num, textvariable=self.num_var, width=10)
        self.entry_num.pack(side=tk.LEFT, padx=5)
        
        # ── 审核开关 ──────────────────────────────────────────────────
        frame_review = tk.Frame(self.root)
        frame_review.pack(pady=2, padx=10, fill=tk.X)
        self.review_var = tk.BooleanVar(value=REVIEW_ENABLED)
        self.chk_review = tk.Checkbutton(frame_review, text="启用内容安全审核（拦截NSFW/血腥/恐怖图）",
                                         variable=self.review_var,
                                         command=self._toggle_review)
        self.chk_review.pack(side=tk.LEFT, padx=5)
        # 审核状态灯
        self.review_led = tk.Canvas(frame_review, width=16, height=16, highlightthickness=0)
        self.review_led.pack(side=tk.LEFT, padx=5)
        self._review_led_green = True
        self._update_review_led()
        
        self.btn_start = tk.Button(self.root, text="开始爬取", command=self.start_crawling, bg="lightgreen")
        self.btn_start.pack(pady=10)
        
        self.log_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=25)
        self.log_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        self.log_text.insert(tk.END, "请选择存放角色图片的根目录。\n")
        self.log_text.insert(tk.END, "支持两种结构：\n")
        self.log_text.insert(tk.END, "  1. 根目录下直接包含多个角色文件夹（每个文件夹一个角色）\n")
        self.log_text.insert(tk.END, "  2. 根目录本身就是某个角色的目录（目录内已有图片）\n")
        self.log_text.insert(tk.END, "  3. 根目录为空 - 自动创建以根目录名命名的角色文件夹并开始爬取\n")
        self.log_text.insert(tk.END, "然后点击「开始爬取」\n")
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _toggle_review(self):
        """审核开关切换"""
        global REVIEW_ENABLED
        REVIEW_ENABLED = self.review_var.get()
        self._update_review_led()
        status = "开启" if REVIEW_ENABLED else "关闭"
        self.log_text.insert(tk.END, f"图片内容审核已{status}\n")

    def _update_review_led(self):
        """更新审核状态指示灯"""
        if not hasattr(self, 'review_led'):
            return
        self.review_led.delete("all")
        if REVIEW_ENABLED:
            self.review_led.create_oval(2, 2, 14, 14, fill="#22c55e", outline="")
        else:
            self.review_led.create_oval(2, 2, 14, 14, fill="#ef4444", outline="")

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.path_var.set(folder)
    
    def start_crawling(self):
        if self.running:
            self.log_text.insert(tk.END, "爬取任务已在运行中，请稍后...\n")
            return
        root_dir = self.path_var.get().strip()
        if not root_dir:
            self.log_text.insert(tk.END, "请先选择目标文件夹！\n")
            return
        try:
            max_num = int(self.num_var.get())
            if max_num <= 0:
                raise ValueError
        except ValueError:
            self.log_text.insert(tk.END, "图片数量必须是正整数！\n")
            return
        
        self.running = True
        self.btn_start.config(state=tk.DISABLED, text="爬取中...")
        self.log_text.insert(tk.END, f"开始爬取，根目录：{root_dir}，每个角色最多 {max_num} 张\n")
        self.log_text.see(tk.END)
        
        thread = threading.Thread(target=self._run_crawler, args=(root_dir, max_num))
        thread.daemon = True
        thread.start()
    
    def _run_crawler(self, root_dir, max_num):
        sys.stdout = StdoutRedirector(self.log_text)
        try:
            crawl_all_roles(root_dir, max_num)
        except Exception as e:
            print(f"爬取过程中发生异常: {e}")
        finally:
            sys.stdout = self.original_stdout
            self.root.after(0, self._crawl_finished)
    
    def _crawl_finished(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL, text="开始爬取")
        self.log_text.insert(tk.END, "\n爬取任务结束。\n")
        self.log_text.see(tk.END)
    
    def on_close(self):
        if self.running:
            self.log_text.insert(tk.END, "正在爬取中，请等待完成后再关闭...\n")
            return
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = CrawlerGUI()
    app.run()
