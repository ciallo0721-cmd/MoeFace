#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import smtplib
import requests
import json
import time
import hashlib
import urllib.parse
import traceback
from bs4 import BeautifulSoup
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin

# ==================== 配置 ====================
# ==================== 配置 ====================
# 获取当前时间用于日期过滤
CURRENT_DATE = datetime.now()
DATE_THRESHOLD = CURRENT_DATE - timedelta(days=3)  # 只抓取3天内的新闻
TODAY_STR = CURRENT_DATE.strftime('%Y-%m-%d')
THRESHOLD_STR = DATE_THRESHOLD.strftime('%Y-%m-%d')

NEWS_SOURCES = [
    {
        'name': 'Hololive Production',
        'url': 'https://hololivepro.com/news/',
        'type': 'html',
        'parser': 'hololive'
    },
    {
        'name': 'Nijisanji',
        'url': 'https://www.nijisanji.jp/news',
        'type': 'html',
        'parser': 'nijisanji'
    }
]

BILIBILI_UIDS = [
    '33064694',
    '1265680561',
]

# ==================== 微博配置 ====================
WEIBO_CONFIG = {
    'enabled': True,
    'monitor_uids': ['7618923072'],  # 永雏塔菲的UID
    'monitor_names': ['永雏塔菲'],
    'cookie': os.getenv('WEIBO_COOKIE', ''),
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://weibo.com/',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Connection': 'keep-alive',
    }
}
if WEIBO_CONFIG['cookie']:
    WEIBO_CONFIG['headers']['Cookie'] = WEIBO_CONFIG['cookie']
else:
    print("[配置] ⚠️ 微博 Cookie 未配置，将尝试无登录抓取（可能失败）")

# SEMI-RESTFUL API endpoints
WEIBO_API_CONTAINERID = "100505"  # 用户微博类型前缀
WEIBO_API_BASE = "https://weibo.com/ajax/statuses/mymblog"

SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.qq.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '465'))
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PASS = os.getenv('SMTP_PASS', '')
TO_EMAIL_RAW = os.getenv('TO_EMAIL', '')
RECIPIENTS = [email.strip() for email in TO_EMAIL_RAW.split(',') if email.strip()] if TO_EMAIL_RAW else []

BILI_COOKIE = os.getenv('BILI_COOKIE', '')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.bilibili.com/',
    'Origin': 'https://www.bilibili.com',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Connection': 'keep-alive',
}

if BILI_COOKIE:
    HEADERS['Cookie'] = BILI_COOKIE
    print("[配置] ✅ B站 Cookie 已加载")
else:
    print("[配置] ⚠️ B站 Cookie 未配置，可能无法获取动态")

# ==================== WBI 签名 ====================
MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52
]

def get_wbi_keys() -> Tuple[str, str]:
    try:
        url = 'https://api.bilibili.com/x/web-interface/nav'
        response = requests.get(url, headers=HEADERS, timeout=10)
        data = response.json()
        if data.get('code') != 0:
            print(f"  [WBI] 获取 keys 失败: {data.get('message')}")
            return '', ''
        wbi_img_url = data['data']['wbi_img']['img_url']
        wbi_sub_url = data['data']['wbi_img']['sub_url']
        img_key = wbi_img_url.rsplit('/', 1)[-1].split('.')[0]
        sub_key = wbi_sub_url.rsplit('/', 1)[-1].split('.')[0]
        return img_key, sub_key
    except Exception as e:
        print(f"  [WBI] 获取 keys 异常: {e}")
        return '', ''

def encrypt_wbi(params: dict, img_key: str, sub_key: str) -> dict:
    key_str = img_key + sub_key
    if len(key_str) < 64:
        print(f"  [WBI警告] key 长度不足64 ({len(key_str)}), 跳过签名")
        return params

    mixin_key = ''
    for i in range(64):
        mixin_key += key_str[MIXIN_KEY_ENC_TAB[i]]
    mixin_key = mixin_key[:32]

    params['wts'] = int(time.time())
    sorted_params = sorted(params.items())
    query = urllib.parse.urlencode(sorted_params)
    sign_str = query + mixin_key
    params['w_rid'] = hashlib.md5(sign_str.encode()).hexdigest()
    return params

def sign_request(url: str, params: dict) -> Tuple[str, dict]:
    img_key, sub_key = get_wbi_keys()
    if not img_key or not sub_key:
        print("  [WBI] 获取 img_key/sub_key 失败，本次请求不签名")
        return url, params
    signed_params = encrypt_wbi(params.copy(), img_key, sub_key)
    return url, signed_params

# ==================== 工具函数 ====================

def fetch_html(url: str, timeout: int = 15) -> Optional[str]:
    try:
        response = requests.get(url, timeout=timeout, headers=HEADERS)
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.text
    except requests.RequestException as e:
        print(f"  [错误] 抓取失败 {url}: {e}")
        return None

def fetch_json(url: str, params: dict = None, timeout: int = 15) -> Optional[dict]:
    try:
        if params is None:
            params = {}
        signed_url, signed_params = sign_request(url, params)
        response = requests.get(signed_url, params=signed_params, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"  [错误] 抓取失败 {url}: {e}")
        return None

# ==================== Hololive 解析 ====================

def parse_hololive_news(html: str) -> List[Dict]:
    news_list = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        articles = soup.find_all('div', class_=re.compile(r'news|post|article|entry', re.I))
        if not articles or len(articles) < 2:
            articles = soup.find_all(['li', 'div'], class_=True)
        for article in articles[:15]:
            date_elem = article.find('time')
            if not date_elem:
                date_elem = article.find(class_=re.compile(r'date|time', re.I))
            if not date_elem:
                text = article.get_text()
                date_match = re.search(r'(\d{4}[.-]\d{1,2}[.-]\d{1,2})', text)
                if date_match:
                    pub_date_raw = date_match.group(1)
                else:
                    continue
            else:
                pub_date_raw = date_elem.get_text(strip=True)
                pub_date_raw = re.sub(r'[年月]', '.', pub_date_raw).replace('日', '').strip('.')
            # 标准化日期格式并过滤
            pub_date = normalize_date(pub_date_raw)
            if not pub_date:
                continue
            if not is_recent(pub_date):
                # print(f"  [过滤] Hololive 旧闻: {pub_date_raw} -> {pub_date}")
                continue
            link_elem = article.find('a', href=True)
            if not link_elem:
                title_elem = article.find(['h2', 'h3', 'h4', 'p', 'span'])
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    if title and len(title) > 5 and title != pub_date:
                        news_list.append({
                            'title': title[:150],
                            'link': '#',
                            'pub_date': pub_date,
                            'source': 'Hololive Production'
                        })
                continue
            link = link_elem.get('href', '')
            if link and not link.startswith('http'):
                link = urljoin('https://hololivepro.com', link)
            title = link_elem.get_text(strip=True)
            if not title or len(title) < 3:
                title_elem = article.find(['h2', 'h3', 'h4', 'p', 'span'])
                title = title_elem.get_text(strip=True)[:150] if title_elem else ""
            if title and len(title) > 5 and title != pub_date:
                title = re.sub(r'^\d{4}[.-]\d{1,2}[.-]\d{1,2}\s*', '', title)
                news_list.append({
                    'title': title[:150],
                    'link': link if link != '#' else '',
                    'pub_date': pub_date,
                    'source': 'Hololive Production'
                })
        print(f"  [OK] Hololive Production: 解析到 {len(news_list)} 条最近新闻")
    except Exception as e:
        print(f"  [错误] Hololive 解析失败: {e}")
    return news_list

# ==================== Nijisanji 解析 ====================

def parse_nijisanji_news(html: str) -> List[Dict]:
    news_list = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        items = soup.find_all('li', class_=re.compile(r'news|post|item', re.I))
        if not items:
            items = soup.find_all(['li', 'div'], class_=True)
        if not items:
            paragraphs = soup.find_all(['p', 'div'])
            items = [p for p in paragraphs if re.search(r'\d{4}[./]\d{1,2}[./]\d{1,2}', p.get_text())]
        for item in items[:20]:
            text = item.get_text()
            date_match = re.search(r'(\d{4}[./-]\d{1,2}[./-]\d{1,2})', text)
            if not date_match:
                continue
            pub_date_raw = date_match.group(1).replace('/', '.')
            # 标准化日期格式并过滤
            pub_date = normalize_date(pub_date_raw)
            if not pub_date or not is_recent(pub_date):
                continue
            link_elem = item.find('a', href=True)
            if not link_elem:
                continue
            link = link_elem.get('href', '')
            if link and not link.startswith('http'):
                link = urljoin('https://www.nijisanji.jp', link)
            title = link_elem.get_text(strip=True)
            if not title or len(title) < 3:
                title = re.sub(r'^\d{4}[./-]\d{1,2}[./-]\d{1,2}\s*', '', text).strip()
            if title and len(title) > 3:
                news_list.append({
                    'title': title[:150],
                    'link': link,
                    'pub_date': pub_date,
                    'source': 'Nijisanji'
                })
        print(f"  [OK] Nijisanji: 解析到 {len(news_list)} 条最近新闻")
    except Exception as e:
        print(f"  [错误] Nijisanji 解析失败: {e}")
    return news_list

# ==================== 微博微博抓取（支持登录/无登录）====================
# ==================== 微博微博抓取（支持登录/无登录）====================

def fetch_weibo_posts(uid, limit=10, name='微博用户'):
    """抓取微博用户的帖子，返回解析后的新闻列表"""
    weibo_news = []
    print(f"  [微博] 开始抓取用户: {name} (UID: {uid}) 的最近微博")
    
    try:
        # 使用微博的 semi-restful API
        params = {
            "uid": uid,
            "page": 1,
            "feature": 0,
            "since_id": None,
            "max_id": None,
            "count": limit
        }
        
        # 尝试使用API接口获取数据
        api_url = f"{WEIBO_API_BASE}?uid={uid}&page=1&feature=2&count={limit}"
        
        # 使用配置的headers或带Cookie的headers
        headers = WEIBO_CONFIG['headers'].copy()
        headers.update({
            'X-Requested-With': 'XMLHttpRequest',
            'Accept': 'application/json, text/plain, */*',
        })
        
        # 发送请求
        response = requests.get(api_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('ok') == 1 or 'data' in data:
                posts = data.get('data', {}).get('list', [])
                if not posts:
                    print(f"  [微博] {name} 没有获取到帖子数据")
                    return []
                
                for post in posts[:limit]:
                    try:
                        # 提取帖子内容
                        text_raw = post.get('text_raw', '') or post.get('text', '')
                        # 移除HTML标签
                        title = re.sub('<[^<]+?>', '', text_raw).strip()
                        if not title:
                            continue
                            
                        # 处理转发微博
                        if post.get('retweeted_status'):
                            retweet = post['retweeted_status']
                            retweet_text = re.sub('<[^<]+?>', '', retweet.get('text', ''))
                            title = f"[转发] {title[:80]} // {retweet_text[:80]}"
                        
                        # 限制标题长度
                        if len(title) > 150:
                            title = title[:147] + "..."
                            
                        # 构建微博链接
                        post_id = post.get('id', '')
                        link = f"https://weibo.com/{uid}/{post_id}" if post_id else f"https://weibo.com/u/{uid}"
                        
                        # 处理时间
                        created_at = post.get('created_at', '')
                        if created_at:
                            try:
                                # 解析微博时间格式: "Wed May 01 15:30:00 +0800 2026"
                                pub_date = parse_weibo_datetime(created_at)
                                if not is_recent(pub_date):
                                    # print(f"  [过滤] {name} 的微博时间: {pub_date} 超过3天")
                                    continue
                            except Exception as e:
                                print(f"    [警告] 时间解析失败: {created_at}, 错误: {e}")
                                pub_date = datetime.now().strftime('%Y-%m-%d %H:%M')
                        else:
                            pub_date = datetime.now().strftime('%Y-%m-%d %H:%M')
                        
                        weibo_news.append({
                            'title': title[:150],
                            'link': link,
                            'pub_date': pub_date,
                            'source': f'微博: {name}'
                        })
                    except Exception as post_error:
                        print(f"    [警告] 处理微博帖子失败: {post_error}")
                        continue
                
                print(f"  [OK] 微博 {name}: 获取到 {len(weibo_news)} 条最近动态")
                return weibo_news
            else:
                print(f"  [警告] 微博 {name}: API返回异常: {data.get('msg', '未知错误')}")
                return []
        else:
            print(f"  [警告] 微博 {name}: HTTP请求失败, 状态码: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"  [错误] 微博用户 {name} 抓取失败: {e}")
        import traceback
        print(traceback.format_exc())
        return []

def fetch_weibo_news() -> List[Dict]:
    """抓取所有配置的微博用户的动态"""
    all_weibo_news = []
    if not WEIBO_CONFIG['enabled']:
        print("[INFO] 微博抓取未启用")
        return []
    
    if not WEIBO_CONFIG['monitor_uids']:
        print("[INFO] 未配置微博UID列表")
        return []
    
    print(f"[INFO] 开始抓取 {len(WEIBO_CONFIG['monitor_uids'])} 个微博用户的动态")
    
    for idx, uid in enumerate(WEIBO_CONFIG['monitor_uids']):
        name = WEIBO_CONFIG['monitor_names'][idx] if idx < len(WEIBO_CONFIG['monitor_names']) else f'用户{uid}'
        posts = fetch_weibo_posts(uid, 15, name)
        all_weibo_news.extend(posts)
        time.sleep(1)  # 避免请求过快
    
    print(f"  [OK] 微博动态: 总共抓取到 {len(all_weibo_news)} 条")
    return all_weibo_news

def parse_weibo_datetime(dt_str):
    """解析微博时间格式"""
    # 示例："Wed May 01 15:30:00 +0800 2026"
    try:
        # 移除时区部分进行解析
        cleaned = re.sub(r'\+\d{4}', '', dt_str)
        from datetime import datetime
        parsed_time = datetime.strptime(cleaned.strip(), '%a %b %d %H:%M:%S %Y')
        return parsed_time.strftime('%Y-%m-%d %H:%M')
    except Exception as e:
        # 尝试简单提取日期
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', dt_str)
        if date_match:
            return date_match.group(1)
        print(f"    [警告] 无法解析时间: {dt_str}")
        return datetime.now().strftime('%Y-%m-%d %H:%M')

# ==================== 日期处理辅助函数 ====================

def normalize_date(date_str: str) -> Optional[str]:
    """标准化日期格式为 YYYY-MM-DD"""
    if not date_str:
        return None
    # 处理各种常见格式
    date_str = date_str.strip()
    # 格式1: 2026.04.27
    match = re.match(r'(\d{4})[./-](\d{1,2})[./-](\d{1,2})', date_str)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    # 格式2: 2026年04月27日
    match = re.match(r'(\d{4})年(\d{1,2})月(\d{1,2})日', date_str)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return None

def is_recent(date_str: str, days: int = 3) -> bool:
    """判断日期是否在指定天数内"""
    if not date_str:
        return False
    try:
        # 尝试解析日期（可能包含时间）
        if ' ' in date_str:
            dt = datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d')
        else:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
        threshold = datetime.now() - timedelta(days=days)
        return dt >= threshold
    except Exception as e:
        print(f"  [日期警告] 无法解析日期 '{date_str}': {e}")
        return False

def fetch_bilibili_dynamics(uid: str, limit: int = 10) -> List[Dict]:
    dynamics = []
    try:
        limit = int(limit)
        url = "https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space"
        params = {
            "host_mid": uid,
            "offset": "",
            "page_size": limit
        }
        print(f"  [B站] 正在获取 UID {uid} 的动态...")
        data = fetch_json(url, params)

        if not data:
            print(f"  [警告] B站 API 返回空数据，UID: {uid}")
            return []

        if data.get('code') != 0:
            print(f"  [警告] B站 API 返回错误码 {data.get('code')}: {data.get('message', '未知错误')}, UID: {uid}")
            if data.get('code') == -352:
                print(f"  [提示] 错误码 -352 表示风控校验失败，请检查 Cookie 是否过期或需要 WBI 签名")
            return []

        items = data.get('data', {}).get('items', [])
        if not items:
            print(f"  [B站] UID {uid} 暂无动态")
            return []

        for item in items[:limit]:
            try:
                # ... existing code for parsing dynamic ...
                modules = item.get('modules', {}) or {}
                author = modules.get('module_author', {}) or {}
                author_name = author.get('name', f'UID {uid}') if isinstance(author, dict) else f'UID {uid}'

                # 处理动态内容
                dynamic_module = modules.get('module_dynamic') or {}
                if isinstance(dynamic_module, dict):
                    desc = dynamic_module.get('desc') or {}
                else:
                    desc = {}
                if isinstance(desc, dict):
                    title = desc.get('text', '')
                else:
                    title = ''

                # 处理转发动态
                if not title and 'orig' in item:
                    orig = item.get('orig', {}) or {}
                    orig_modules = orig.get('modules', {}) or {}
                    orig_dynamic = orig_modules.get('module_dynamic') or {}
                    if isinstance(orig_dynamic, dict):
                        orig_desc = orig_dynamic.get('desc') or {}
                        if isinstance(orig_desc, dict):
                            title = orig_desc.get('text', '')
                        if not title:
                            orig_author = orig_modules.get('module_author') or {}
                            if isinstance(orig_author, dict):
                                author_name = orig_author.get('name', author_name)

                if not title:
                    title = author_name + ' 发布了新动态'

                if len(title) > 100:
                    title = title[:100] + '...'

                # 处理时间戳并过滤
                timestamp = author.get('pub_ts', 0) if isinstance(author, dict) else 0
                if timestamp:
                    try:
                        ts_int = int(timestamp)
                        pub_date = datetime.fromtimestamp(ts_int).strftime('%Y-%m-%d %H:%M')
                        # 只保留3天内的动态
                        dt = datetime.fromtimestamp(ts_int)
                        if dt < DATE_THRESHOLD:
                            # print(f"  [过滤] B站动态时间: {pub_date} 超过3天")
                            continue
                    except (ValueError, TypeError):
                        pub_date = ''
                else:
                    pub_date = ''

                id_str = item.get('id_str', '')
                link = f"https://t.bilibili.com/{id_str}" if id_str else ''

                if title:
                    dynamics.append({
                        'title': title[:150],
                        'link': link,
                        'pub_date': pub_date,
                        'source': f'B站: {author_name}'
                    })
            except Exception as inner_e:
                print(f"    [警告] 处理单条动态时出错: {inner_e}, 跳过")
                continue

        if dynamics:
            print(f"  [OK] UID {uid}: 获取到 {len(dynamics)} 条最近动态")
        else:
            print(f"  [B站] UID {uid} 没有符合条件的动态")

        time.sleep(1)

    except Exception as e:
        print(f"  [错误] B站动态获取失败 UID {uid}: {e}")
        print(traceback.format_exc())

    return dynamics

def fetch_bilibili_news() -> List[Dict]:
    all_dynamics = []
    if not BILIBILI_UIDS:
        print("  [提示] 未配置 B站 UID 列表，跳过国产 VTuber 动态")
        return []
    print(f"[INFO] 开始抓取 {len(BILIBILI_UIDS)} 个 B站 UP 主动态")
    if not BILI_COOKIE:
        print("  [警告] 未配置 BILI_COOKIE，低版本的 API 可能无法获取数据")
    for uid in BILIBILI_UIDS:
        dynamics = fetch_bilibili_dynamics(uid)
        all_dynamics.extend(dynamics)
    print(f"  [OK] B站动态: 总共抓取到 {len(all_dynamics)} 条")
    return all_dynamics

# ==================== HTML 生成 ====================

def generate_html(all_news: Dict[str, List[Dict]]) -> str:
    total_news = sum(len(items) for items in all_news.values())
    # 预留新闻条数，避免内容过多
    display_limit = 50
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>VTuber新闻汇总</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #ff6b6b; border-bottom: 3px solid #ff6b6b; padding-bottom: 10px; }}
            h2 {{ color: #4ecdc4; margin-top: 25px; border-left: 4px solid #4ecdc4; padding-left: 12px; }}
            .news-item {{ margin: 15px 0; padding: 12px; background: #f9f9f9; border-radius: 8px; }}
            .news-title {{ font-size: 16px; font-weight: bold; margin-bottom: 5px; }}
            .news-title a {{ color: #0066cc; text-decoration: none; }}
            .news-title a:hover {{ text-decoration: underline; }}
            .news-meta {{ font-size: 12px; color: #888; }}
            .footer {{ margin-top: 30px; padding-top: 15px; border-top: 1px solid #ddd; font-size: 12px; color: #888; text-align: center; }}
        </style>
    </head>
    <body>
        <h1>🎮 VTuber新闻汇总</h1>
        <p>抓取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>新鲜新闻总数 (近3天内): {total_news} 条</p>
    """
    displayed_news = 0
    for source_name, news_list in all_news.items():
        if news_list:
            html += f'<h2>📌 {source_name}</h2>'
            # 限制显示新闻数，避免HTML过大
            for i, news in enumerate(news_list):
                if i >= display_limit:
                    html += f'<p><em>... 以上仅显示最近{display_limit}条新闻，更多请访问官网</em></p>'
                    break
                displayed_news += 1
                link = news.get('link', '')
                title = news.get('title', '无标题')
                pub_date = news.get('pub_date', '')
                html += f'''
                <div class="news-item">
                    <div class="news-title">
                        <a href="{link}" target="_blank">{escape_html(title)}</a>
                    </div>
                    <div class="news-meta">
                        日期: {escape_html(pub_date) if pub_date else '未知'} | 来源: {escape_html(source_name)}
                    </div>
                </div>
                '''
    if displayed_news == 0:
        html += '<p>近3天内暂无新新闻，请访问官网查看最新动态。</p>'
    html += f"""
        <div class="footer">
            <p>本邮件由 GitHub Actions 自动生成 | VTuber新闻订阅 | 只显示最近3天内的新闻</p>
            <p>📧 如有问题请联系管理员</p>
        </div>
    </body>
    </html>
    """
    return html

def escape_html(text):
    """简单的HTML转义"""
    if not text:
        return ''
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

# ==================== 邮件发送（支持多收件人）====================

def send_email(html_body: str, recipients: List[str]):
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS]) or not recipients:
        print("[错误] 邮件配置不完整或无收件人")
        return
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = f"MoeFace News <{SMTP_USER}>"
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"VTuber新闻汇总 - {datetime.now().strftime('%Y-%m-%d')}"
        msg['Date'] = formatdate(localtime=True)

        html_part = MIMEText(html_body, 'html', 'utf-8')
        msg.attach(html_part)

        print(f"[邮件] 正在连接 {SMTP_HOST}:{SMTP_PORT}")
        if SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
            server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        print(f"[邮件] 登录成功，准备发送给 {len(recipients)} 个收件人")
        server.send_message(msg)
        server.quit()
        print(f"[邮件] ✅ 发送成功！")
    except smtplib.SMTPAuthenticationError as e:
        print(f"[邮件] ❌ 认证失败，请检查SMTP_USER和SMTP_PASS")
        print(f"  错误详情: {e}")
    except smtplib.SMTPDataError as e:
        print(f"[邮件] ❌ 数据错误: {e}")
    except smtplib.SMTPException as e:
        print(f"[邮件] ❌ SMTP错误: {e}")
    except Exception as e:
        print(f"[邮件] ❌ 未知错误: {e}")
        import traceback
        print(traceback.format_exc())

# ==================== 主函数 ====================

def main():
    print("=" * 50)
    print("VTuber News Fetcher 启动 (仅限3天内新闻)")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"新闻过滤器: 只收录 {DATE_THRESHOLD.strftime('%Y-%m-%d')} 之后的新闻")
    print("=" * 50)

    if not SMTP_USER:
        print("[警告] SMTP_USER 未设置，将只抓取新闻不发送邮件")
    if not RECIPIENTS:
        print("[警告] TO_EMAIL 未设置或为空，将只抓取新闻不发送邮件")
    else:
        print(f"[配置] 收件人数: {len(RECIPIENTS)}")

    all_news = {}

    # 抓取 Hololive 和 Nijisanji 新闻
    for source in NEWS_SOURCES:
        html = fetch_html(source['url'])
        if not html:
            print(f"[警告] {source['name']} 抓取失败，已跳过")
            all_news[source['name']] = []
            continue
        if source['parser'] == 'hololive':
            news_list = parse_hololive_news(html)
        elif source['parser'] == 'nijisanji':
            news_list = parse_nijisanji_news(html)
        else:
            news_list = []
        all_news[source['name']] = news_list

    # 抓取 B 站动态
    bilibili_news = fetch_bilibili_news()
    if bilibili_news:
        all_news['国产 VTuber 动态'] = bilibili_news

    # 抓取微博动态
    weibo_news = fetch_weibo_news()
    if weibo_news:
        all_news['微博 VTuber 动态'] = weibo_news

    total_news = sum(len(items) for items in all_news.values())
    print(f"\n总共抓取到 {total_news} 条近3天内新闻")

    # 生成 HTML 报告
    html_body = generate_html(all_news)

    if SMTP_USER and RECIPIENTS:
        send_email(html_body, RECIPIENTS)
    else:
        print(f"成功抓取 {total_news} 条新闻，但邮件配置不完整，未发送")
        # 保存到文件作为备份
        with open('vtuber_news_report.md', 'w', encoding='utf-8') as f:
            f.write(f"# VTuber新闻汇总\n\n")
            f.write(f"抓取时间: {datetime.now()}\n")
            f.write(f"新闻数量(近3天): {total_news}\n\n")
            for source_name, news_list in all_news.items():
                if news_list:
                    f.write(f"## {source_name}\n\n")
                    for news in news_list:
                        f.write(f"- [{news['title']}]({news['link']})\n")
                    f.write("\n")
        print("新闻已保存到 vtuber_news_report.md")

if __name__ == "__main__":
    main()