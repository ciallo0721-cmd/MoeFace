"""
爬虫.py
MoeFace 动漫/VTuber 图片数据采集工具
网络增强版

修改:
- requests代理支持
- 浏览器级headers
- 自动重试
- 反403
- 降低封禁概率
"""

import os
import sys
import json
import re
import time
import random
import urllib.parse
import threading

import requests

from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

import tkinter as tk
from tkinter import filedialog, scrolledtext


# ==========================================================
# 网络配置
# ==========================================================


# 下载线程
# 图片网站不要开太高
DOWNLOAD_THREADS = 2


# 请求间隔
REQUEST_DELAY = 1.0


# 超时
REQUEST_TIMEOUT = (10, 30)



# ==========================================================
# 代理设置
# ==========================================================

ENABLE_PROXY = False


# Clash:
# 7890
#
# v2rayN:
# 10809

PROXIES = {

    "http":
    "http://127.0.0.1:7890",

    "https":
    "http://127.0.0.1:7890"

}



# ==========================================================
# 搜索源开关
# ==========================================================

ENABLE_BING = True

ENABLE_GOOGLE = False
# Google图片反爬严重
# 建议关闭


ENABLE_BAIDU = True


ENABLE_PIXIV = False


ENABLE_SAFEBOORU = True



PIXIV_COOKIE = ""



# ==========================================================
# 浏览器请求头池
# ==========================================================


HEADERS_POOL = [

{
"User-Agent":
"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
"AppleWebKit/537.36 "
"Chrome/120.0.0.0 Safari/537.36",

"Accept":
"image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",

"Accept-Language":
"zh-CN,zh;q=0.9,en;q=0.8",

"Connection":
"keep-alive"

},



{
"User-Agent":
"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
"AppleWebKit/537.36 "
"Chrome/121.0 Safari/537.36",

"Accept":
"*/*",

"Accept-Language":
"zh-CN,zh;q=0.9"

},



{
"User-Agent":
"Mozilla/5.0 (X11; Linux x86_64) "
"AppleWebKit/537.36 "
"Chrome/119 Safari/537.36",

"Accept":
"*/*"

}

]





def get_headers():

    """
    获取随机浏览器Headers
    """

    return random.choice(
        HEADERS_POOL
    ).copy()





def get_request_kwargs():

    """
    requests额外参数
    """

    if ENABLE_PROXY:

        return {
            "proxies":
            PROXIES
        }


    return {}

# ==========================================================
# 通用请求
# ==========================================================


def request_get(url, **kwargs):

    """
    统一requests入口

    自动:
    - headers随机
    - proxy
    - retry
    """

    retry = kwargs.pop(
        "retry",
        3
    )


    for i in range(retry):

        try:

            headers = kwargs.pop(
                "headers",
                None
            )


            if headers is None:
                headers = get_headers()


            resp = requests.get(

                url,

                headers=headers,

                **kwargs,

                **get_request_kwargs()

            )


            return resp



        except Exception as e:


            log(
                f"请求失败 {i+1}/{retry}: {e}"
            )


            time.sleep(
                random.uniform(1,3)
            )


    return None










# ==========================================================
# 图片来源Referer
# ==========================================================


SOURCE_REFERER = {


"Bing":
"https://cn.bing.com/",


"百度":
"https://image.baidu.com/",


"Google":
"https://www.google.com/",


"Pixiv":
"https://www.pixiv.net/",


"Safebooru":
"https://safebooru.org/"

}




# ==========================================================
# 角色关键词
# ==========================================================


ROLE_SUFFIXES = {


"丛雨":
[
"立绘",
"壁纸",
"千恋万花",
"Murasame",
"ムラサメ"
],


"Neuro-sama":
[
"vtuber",
"art",
"fanart",
"AI vtuber"
],


"永雏塔菲":
[
"虚拟主播",
"立绘",
"Taffy",
"Ace Taffy",
"唐人塔菲"
],


"东雪莲":
[
"虚拟主播",
"立绘",
"東雪蓮",
"Yukiren"
],


"ShikiNatsume":
[
"立绘",
"壁纸",
"棗シキ"
],


"Ayachi_Nene":
[
"立绘",
"壁纸",
"綾地寧々"
],


"米塔":
[
"MiSide",
"游戏",
"Mita",
"cosplay"
]


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
REVIEW_ENABLED = False                # 是否启用审核
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

        # ── 1. NSFW 检测（三次审核防误判） ────────────────────────────
        nsfw_score = 0.0
        detector = _get_nsfw_detector()
        if detector and detector is not False:
            try:
                votes = 0
                scores = []

                # 第1轮: 全流程检测（模型 → ONNX → 颜色+形状分析）
                s1 = detector._detect_visual_nsfw(img)[0]
                scores.append(s1)
                if s1 > NSFW_THRESHOLD:
                    votes += 1

                # 第2轮: 纯颜色规则分析（肉色/白色/粉色/黑色比例）
                if hasattr(detector, '_rule_based_visual'):
                    s2, _ = detector._rule_based_visual(img)
                else:
                    s2 = s1
                scores.append(s2)
                if s2 > NSFW_THRESHOLD:
                    votes += 1

                # 第3轮: 纯形状分析（(▪)(▪) / m/w / - - -）
                if hasattr(detector, '_shape_analysis'):
                    s3, _ = detector._shape_analysis(img)
                else:
                    s3 = 0.0
                scores.append(s3)
                if s3 > 0.70:  # 形状分析用独立阈值
                    votes += 1

                nsfw_score = max(scores)

                # 至少 2/3 票才拦截（防误判）
                if votes >= 2:
                    review_stats["blocked_nsfw"] += 1
                    result["pass"] = False
                    result["reason"] = f"NSFW 内容 (三次审核{votes}/3, score={nsfw_score:.2f})"
                    return result

            except Exception:
                pass

        result["scores"]["nsfw"] = round(nsfw_score, 4)

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


    log(
        f"Bing搜索: {keyword}"
    )


    urls=[]


    search_url = (

        "https://cn.bing.com/images/search?"

        f"q={urllib.parse.quote(keyword)}"

    )


    try:

        resp=request_get(

            search_url,

            timeout=REQUEST_TIMEOUT

        )


        if not resp:

            return []


        soup=BeautifulSoup(

            resp.text,

            "html.parser"

        )


        for item in soup.find_all(
            "a",
            class_="iusc"
        ):


            m=item.get("m")


            if m:

                try:

                    data=json.loads(m)

                    url=data.get(
                        "murl"
                    )


                    if url and url not in urls:

                        urls.append(url)


                except:

                    pass



            if len(urls)>=num:

                break



    except Exception as e:

        log(
            f"Bing错误:{e}"
        )


    log(
        f"Bing获得 {len(urls)} 个URL"
    )


    return urls[:num]

def get_baidu_images(keyword,num):


    log(
        f"百度搜索:{keyword}"
    )


    urls=[]


    url=(

    "https://image.baidu.com/search/flip?"

    "tn=baiduimage&ie=utf-8&word="

    +urllib.parse.quote(keyword)

    )



    try:


        resp=request_get(

            url,

            timeout=REQUEST_TIMEOUT

        )


        if not resp:

            return []



        result=re.findall(

            r'"objURL":"(.*?)"',

            resp.text

        )



        for u in result:


            u=u.replace(
                "\\/",
                "/"
            )


            if u not in urls:

                urls.append(u)



            if len(urls)>=num:

                break



    except Exception as e:

        log(
            f"百度错误:{e}"
        )


    log(
        f"百度获得 {len(urls)} 个URL"
    )


    return urls[:num]

def get_google_images(keyword,num):


    log(
        "Google搜索关闭"
    )


    return []

def get_pixiv_images(keyword, num):
    if not PIXIV_COOKIE:
        log("Pixiv 未配置Cookie，跳过")
        return []
    log(f"Pixiv 开始搜索: {keyword}")
    urls = []
    headers = get_headers()
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
    pid = 0
    max_pages = 3
    while len(urls) < num and pid < max_pages * 20:
        # Safebooru XML API 端点
        api_url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&tags={urllib.parse.quote(keyword)}&pid={pid}"
        try:
            log(f"  Safebooru请求第{pid//20 + 1}页 (pid={pid})")
            resp = requests.get(api_url, headers=get_headers(), timeout=REQUEST_TIMEOUT)
            soup = BeautifulSoup(resp.text, 'xml')
            posts = soup.find_all('post')
            if not posts:
                log(f"  Safebooru第{pid//20 + 1}页没有post")
                break
            for post in posts:
                file_url = post.get('file_url')
                if file_url and file_url.startswith('https://') and file_url not in urls:
                    urls.append(file_url)
                    if len(urls) >= num:
                        break
            pid += 20
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            log(f"  Safebooru出错: {e}")
            break
    log(f"Safebooru结束，获得 {len(urls)} 个URL")
    return urls[:num]

SOURCE_GETTERS=[]

if ENABLE_BING:

    SOURCE_GETTERS.append(
        (
            "Bing",
            get_bing_images
        )
    )


if ENABLE_BAIDU:

    SOURCE_GETTERS.append(
        (
            "百度",
            get_baidu_images
        )
    )


if ENABLE_GOOGLE:

    SOURCE_GETTERS.append(
        (
            "Google",
            get_google_images
        )
    )


if ENABLE_PIXIV:

    SOURCE_GETTERS.append(
        (
            "Pixiv",
            get_pixiv_images
        )
    )


if ENABLE_SAFEBOORU:

    SOURCE_GETTERS.append(
        (
            "Safebooru",
            get_safebooru_images
        )
    )




def collect_urls_from_sources(
        keyword,
        target_num,
        suffixes
):


    result=[]

    exists=set()



    keywords=[

        keyword

    ]



    for s in suffixes:

        keywords.append(

            f"{keyword} {s}"

        )




    for kw in keywords:


        if len(result)>=target_num:

            break



        log(
            f"搜索关键词:{kw}"
        )



        for name,getter in SOURCE_GETTERS:



            if len(result)>=target_num:

                break



            need=target_num-len(result)



            count=min(
                10,
                need
            )



            try:


                urls=getter(

                    kw,

                    count

                )



            except Exception as e:


                log(
                    f"{name}失败:{e}"
                )


                continue




            for u in urls:


                if not u:

                    continue



                if u in exists:

                    continue



                if contains_negative_word(u):

                    continue



                exists.add(u)



                result.append(

                    (
                        u,
                        name
                    )

                )



                if len(result)>=target_num:

                    break



            time.sleep(
                REQUEST_DELAY
            )



    log(
        f"最终获得 {len(result)} URL"
    )



    return result



def download_image(
        url,
        dir_path,
        base_name,
        referer=None
):


    if contains_negative_word(url):

        log(
            "负面词跳过"
        )

        return False



    for retry in range(3):


        try:


            headers=get_headers()



            if referer:

                headers["Referer"]=referer



            resp=requests.get(

                url,

                headers=headers,

                timeout=(10,25),

                stream=True,

                **get_request_kwargs()

            )



            # ==============================
            # 403处理
            # ==============================


            if resp.status_code==403:


                log(
                    f"403 重试 {retry+1}/3"
                )


                parsed=urllib.parse.urlparse(
                    url
                )


                headers["Referer"]=(
                    f"{parsed.scheme}://{parsed.netloc}/"
                )


                time.sleep(
                    random.uniform(1,3)
                )


                continue




            if resp.status_code!=200:


                log(
                    f"HTTP {resp.status_code}"
                )

                return False




            # 文件类型


            content_type=resp.headers.get(
                "content-type",
                ""
            )


            ext=".jpg"



            if "png" in content_type:

                ext=".png"


            elif "webp" in content_type:

                ext=".webp"



            elif "gif" in content_type:

                ext=".gif"




            file_path=os.path.join(

                dir_path,

                base_name+ext

            )




            with open(
                file_path,
                "wb"
            ) as f:


                for chunk in resp.iter_content(
                    8192
                ):


                    if chunk:

                        f.write(chunk)





            # 太小删除


            if os.path.getsize(file_path)<1024:


                os.remove(
                    file_path
                )

                return False




            # 内容审核

            if REVIEW_ENABLED:


                review=review_image(
                    file_path
                )


                if not review["pass"]:


                    log(
                        "审核删除:"
                        +review["reason"]
                    )


                    os.remove(
                        file_path
                    )


                    return False




            return True




        except requests.exceptions.Timeout:


            log(
                "下载超时"
            )



        except Exception as e:


            log(
                f"下载异常:{e}"
            )



        time.sleep(
            random.uniform(1,3)
        )



    return False



def get_roles_from_dir(root_dir):


    if not os.path.isdir(root_dir):

        return []



    items=os.listdir(
        root_dir
    )



    subdirs=[

        x for x in items

        if os.path.isdir(
            os.path.join(
                root_dir,
                x
            )
        )

    ]



    # 有角色文件夹

    if subdirs:


        return [

            (
                x,
                os.path.join(
                    root_dir,
                    x
                )

            )

            for x in subdirs

        ]




    # 当前目录有图片


    images=[

        x for x in items

        if x.lower().endswith(

            (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
            )

        )

    ]



    if images:


        return [

            (
            os.path.basename(root_dir),
            root_dir
            )

        ]




    # 空目录


    return [

        (
        os.path.basename(root_dir),
        root_dir
        )

    ]




def crawl_all_roles(
        root_dir,
        max_images_per_role
):


    roles=get_roles_from_dir(
        root_dir
    )



    if not roles:


        print(
            "没有角色"
        )

        return




    print(

        "角色:"
        +
        ",".join(
            [
                x[0]
                for x in roles
            ]
        )

    )




    for role_name,role_path in roles:



        print(
            "\n======================"
        )


        print(
            "处理:",
            role_name
        )



        os.makedirs(
            role_path,
            exist_ok=True
        )



        existing=[

            f for f in os.listdir(role_path)

            if f.lower().endswith(

                (
                ".jpg",
                ".png",
                ".webp"
                )

            )

        ]



        index=len(existing)



        suffixes=ROLE_SUFFIXES.get(

            role_name,

            [
            "立绘",
            "wallpaper",
            "fanart"
            ]

        )



        urls=collect_urls_from_sources(

            role_name,

            max_images_per_role,

            suffixes

        )




        success=0



        with ThreadPoolExecutor(

            max_workers=DOWNLOAD_THREADS

        ) as executor:



            tasks=[]



            for i,(url,src) in enumerate(urls):


                filename=(

                    f"{role_name}_"
                    f"{index+i}"

                )



                task=executor.submit(

                    download_image,

                    url,

                    role_path,

                    filename,

                    SOURCE_REFERER.get(src)

                )


                tasks.append(task)




            for i,t in enumerate(tasks):


                try:


                    if t.result():

                        success+=1


                        print(

                            "✅",

                            i+1

                        )


                    else:

                        print(

                            "❌",

                            i+1

                        )



                except Exception as e:


                    print(
                        "异常:",
                        e
                    )





        print(

            f"{role_name}完成:"
            f"{success}张"

        )




    print(
        "\n全部完成"
    )



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


# ==========================================================
# GUI
# ==========================================================


class StdoutRedirector:


    def __init__(self,text_widget):

        self.text_widget=text_widget



    def write(self,msg):

        self.text_widget.after(

            0,

            self._insert,

            msg

        )



    def _insert(self,msg):

        self.text_widget.insert(

            tk.END,

            msg

        )

        self.text_widget.see(
            tk.END
        )



    def flush(self):

        pass






class CrawlerGUI:


    def __init__(self):


        self.root=tk.Tk()


        self.root.title(

            "MoeFace 图片采集工具"

        )


        self.root.geometry(

            "700x500"

        )



        self.running=False


        self.old_stdout=sys.stdout




        # 路径


        frame=tk.Frame(
            self.root
        )

        frame.pack(

            pady=10,

            padx=10,

            fill=tk.X

        )



        tk.Label(

            frame,

            text="目标目录"

        ).pack(
            side=tk.LEFT
        )



        self.path=tk.StringVar()



        tk.Entry(

            frame,

            textvariable=self.path,

            width=50

        ).pack(

            side=tk.LEFT,

            expand=True,

            fill=tk.X

        )




        tk.Button(

            frame,

            text="浏览",

            command=self.choose

        ).pack(
            side=tk.LEFT
        )




        # 数量


        frame2=tk.Frame(
            self.root
        )

        frame2.pack()



        tk.Label(

            frame2,

            text="每角色数量"

        ).pack(
            side=tk.LEFT
        )



        self.number=tk.StringVar(

            value="100"

        )



        tk.Entry(

            frame2,

            textvariable=self.number,

            width=10

        ).pack(
            side=tk.LEFT
        )







        self.button=tk.Button(

            self.root,

            text="开始爬取",

            command=self.start

        )

        self.button.pack(
            pady=10
        )







        self.logbox=scrolledtext.ScrolledText(

            self.root,

            height=25

        )

        self.logbox.pack(

            padx=10,

            pady=5,

            fill=tk.BOTH,

            expand=True

        )



        self.logbox.insert(

            tk.END,

            "MoeFace 图片采集工具\n"

            "请选择角色目录后开始\n"

        )



        self.root.protocol(

            "WM_DELETE_WINDOW",

            self.close

        )







    def choose(self):


        folder=filedialog.askdirectory()


        if folder:

            self.path.set(
                folder
            )





    def start(self):


        if self.running:

            return



        folder=self.path.get()



        if not folder:

            return



        try:

            num=int(
                self.number.get()
            )

        except:


            num=100




        self.running=True


        self.button.config(

            state=tk.DISABLED,

            text="运行中..."

        )




        t=threading.Thread(

            target=self.worker,

            args=(folder,num)

        )


        t.daemon=True


        t.start()





    def worker(self,folder,num):


        sys.stdout=StdoutRedirector(

            self.logbox

        )



        try:


            crawl_all_roles(

                folder,

                num

            )



        except Exception as e:


            print(

                "错误:",

                e

            )



        finally:


            sys.stdout=self.old_stdout


            self.root.after(

                0,

                self.finish

            )





    def finish(self):


        self.running=False


        self.button.config(

            state=tk.NORMAL,

            text="开始爬取"

        )



        self.logbox.insert(

            tk.END,

            "\n任务结束\n"

        )





    def close(self):


        if self.running:


            print(
                "任务运行中"
            )

            return



        self.root.destroy()




    def run(self):


        self.root.mainloop()





# ==========================================================
# 启动
# ==========================================================


if __name__=="__main__":


    app=CrawlerGUI()

    app.run()
