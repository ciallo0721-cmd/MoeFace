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
CURRENT_DATE = datetime.now()
DATE_THRESHOLD = CURRENT_DATE - timedelta(days=3)

# 永雏塔菲判定阈值（可修改）
TAFFY_THRESHOLD = 3               # 至少出现 3 条
TAFFY_RATIO_THRESHOLD = 0.3       # 且占总新闻数的 30% 以上

# 音频文件硬编码 Raw URL (仓库: ciallo0721-cmd/moeface, 分支: main, 目录: scripts)
AUDIO_URL_TAFFY = "https://raw.githubusercontent.com/ciallo0721-cmd/moeface/main/scripts/yctf.mp3"
AUDIO_URL_DEFAULT = "https://raw.githubusercontent.com/ciallo0721-cmd/moeface/main/scripts/vtb.mp3"

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

BILIBILI_UIDS = ['33064694', '1265680561']

# 微博配置
WEIBO_CONFIG = {
    'enabled': True,
    'monitor_uids': ['7618923072'],
    'monitor_names': ['永雏塔菲'],
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://m.weibo.cn/',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Connection': 'keep-alive',
        'X-Requested-With': 'XMLHttpRequest',
    }
}

# 读取环境变量 WEIBO_COOKIE
weibo_cookie_env = os.getenv('WEIBO_COOKIE', '')
print(f"[DEBUG] WEIBO_COOKIE 环境变量存在: {bool(weibo_cookie_env)}")
if weibo_cookie_env:
    WEIBO_CONFIG['headers']['Cookie'] = weibo_cookie_env
    print("[配置] ✅ 微博 Cookie 已从环境变量加载")
else:
    print("[配置] ℹ️ 微博 Cookie 未配置，将尝试无登录抓取")

SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.qq.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '465'))
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PASS = os.getenv('SMTP_PASS', '')
TO_EMAIL_RAW = os.getenv('TO_EMAIL', '')
RECIPIENTS = [email.strip() for email in TO_EMAIL_RAW.split(',') if email.strip()] if TO_EMAIL_RAW else []

# 加载 Issue #1 订阅者列表
SUBSCRIBERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'subscribers.txt')
if os.path.exists(SUBSCRIBERS_FILE):
    try:
        with open(SUBSCRIBERS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                email = line.strip()
                if email and '@' in email and not email.startswith('#'):
                    if email not in RECIPIENTS:
                        RECIPIENTS.append(email)
        if RECIPIENTS:
            print(f"[配置] ✅ 已加载订阅者列表: 共 {len(RECIPIENTS)} 个收件人")
    except Exception as e:
        print(f"[警告] 读取 subscribers.txt 失败: {e}")
elif RECIPIENTS:
    print(f"[配置] 收件人数: {len(RECIPIENTS)} (仅环境变量 TO_EMAIL)")


BILI_COOKIE = os.getenv('BILI_COOKIE', '')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.bilibili.com/',
    'Origin': 'https://www.bilibili.com',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

if BILI_COOKIE:
    HEADERS['Cookie'] = BILI_COOKIE
    print("[配置] ✅ B站 Cookie 已加载")
else:
    print("[配置] ⚠️ B站 Cookie 未配置")

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

# ==================== 日期处理辅助函数 ====================

def normalize_date(date_str: str) -> Optional[str]:
    if not date_str:
        return None
    date_str = date_str.strip()
    match = re.match(r'(\d{4})[./-](\d{1,2})[./-](\d{1,2})', date_str)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    match = re.match(r'(\d{4})年(\d{1,2})月(\d{1,2})日', date_str)
    if match:
        year, month, day = match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return None

def is_recent(date_str: str, days: int = 3) -> bool:
    if not date_str:
        return False
    try:
        if ' ' in date_str:
            dt = datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d')
        else:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
        threshold = datetime.now() - timedelta(days=days)
        return dt >= threshold
    except Exception:
        return False

def parse_weibo_datetime(dt_str: str) -> str:
    try:
        cleaned = re.sub(r'\+\d{4}', '', dt_str)
        dt = datetime.strptime(cleaned.strip(), '%a %b %d %H:%M:%S %Y')
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', dt_str)
        if date_match:
            return date_match.group(1)
        return datetime.now().strftime('%Y-%m-%d %H:%M')

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
            pub_date = normalize_date(pub_date_raw)
            if not pub_date or not is_recent(pub_date):
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
        print(f"  [OK] Hololive Production: {len(news_list)} 条")
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
        print(f"  [OK] Nijisanji: {len(news_list)} 条")
    except Exception as e:
        print(f"  [错误] Nijisanji 解析失败: {e}")
    return news_list

# ==================== 微博抓取 ====================

# 微博反爬状态码说明（SHANHAI-SERVER）
WEIBO_BLOCK_CODES = {
    418: "被识别为自动化请求（反爬拦截）",
    432: "山海关防火墙拦截（需要有效登录Cookie）",
    403: "IP被临时封禁或访问频率过高",
    302: "被重定向到登录页（Cookie失效）",
}


def fetch_weibo_posts(uid: str, limit: int = 10, name: str = '微博用户') -> List[Dict]:
    weibo_news = []
    print(f"  [微博] 抓取: {name} (UID: {uid})")
    print(f"  [微博] Cookie状态: {'已配置' if WEIBO_CONFIG['headers'].get('Cookie') else '未配置(无Cookie)'}")

    try:
        containerid = f"107603{uid}"
        url = f"https://m.weibo.cn/api/container/getIndex?type=uid&value={uid}&containerid={containerid}"
        headers = WEIBO_CONFIG['headers'].copy()

        resp = requests.get(url, headers=headers, timeout=15)

        # ====== 阶段1：HTTP 状态码检查（增强版）======
        if resp.status_code != 200:
            block_reason = WEIBO_BLOCK_CODES.get(resp.status_code, f'HTTP错误')
            print(f"  [微博] ❌ {block_reason} (HTTP {resp.status_code})")
            # 针对已知反爬码给出具体修复建议
            if resp.status_code in (418, 432):
                print(f"  [微博] 💡 修复建议：更新 GitHub Secrets 中的 WEIBO_COOKIE（当前Cookie可能已过期/无效）")
                print(f"  [微博]    获取方式：浏览器登录 m.weibo.cn → F12 → 复制完整 Cookie 值")
            elif resp.status_code == 403:
                print(f"  [微博] 💡 建议：检查 IP 是否被限制，或等待几分钟后重试")
            elif resp.status_code == 302:
                print(f"  [微博] 💡 Cookie 已失效，需要重新获取并更新 WEIBO_COOKIE")
            return []

        # ====== 阶段2：JSON 解析安全处理 ======
        if not resp.text.strip():
            print(f"  [微博] ❌ 响应体为空（服务器返回了空内容）")
            return []

        try:
            data = resp.json()
        except Exception as json_err:
            print(f"  [微博] ❌ JSON解析失败: {json_err}")
            print(f"  [微博]    响应内容前100字符: {repr(resp.text[:100])}")
            return []

        # ====== 阶段3：业务状态码 + 详细诊断 ======
        ok_val = data.get('ok')
        msg_val = data.get('msg', '')

        if ok_val != 1:
            # 打印完整的响应结构用于调试
            print(f"  [微博] ❌ API 业务异常: ok={ok_val}, msg={repr(msg_val)}")
            # 打印顶层 key 帮助分析异常结构
            top_keys = list(data.keys())[:10]
            print(f"  [微博]    响应结构 keys: {top_keys}")
            # 尝试提取更多有用信息
            if isinstance(data.get('data'), dict):
                data_keys = list(data['data'].keys())[:8]
                print(f"  [微博]    data.keys: {data_keys}")
            print(f"  [微博] 💡 可能原因：Cookie权限不足 / 账号异常 / 接口变更")
            return []

        cards = data.get('data', {}).get('cards', [])
        if not cards:
            print(f"  [微博] ⚠️ {name} API 返回正常但无卡片数据（账号可能设置了隐私/无微博）")
            return []

        total_cards = len(cards)
        filtered_by_date = 0
        filtered_by_empty_text = 0
        processed = 0
        for card in cards:
            if processed >= limit:
                break
            mblog = card.get('mblog') or card.get('status')
            if not mblog:
                continue

            try:
                raw_text = mblog.get('text', '')
                title = re.sub('<[^<]+?>', '', raw_text).strip()
                if not title:
                    if mblog.get('retweeted_status'):
                        title = "转发微博"
                    else:
                        filtered_by_empty_text += 1
                        continue

                if mblog.get('retweeted_status'):
                    retweet = mblog['retweeted_status']
                    retweet_text = re.sub('<[^<]+?>', '', retweet.get('text', ''))
                    if title == "转发微博":
                        title = f"[转发] {retweet_text}"
                    else:
                        title = f"{title} // {retweet_text}"

                if len(title) > 200:
                    title = title[:197] + "..."

                post_id = mblog.get('id', '')
                link = f"https://m.weibo.cn/detail/{post_id}" if post_id else f"https://m.weibo.cn/u/{uid}"

                created_at = mblog.get('created_at', '')
                if created_at:
                    pub_date = parse_weibo_datetime(created_at)
                    if not is_recent(pub_date):
                        filtered_by_date += 1
                        continue
                else:
                    pub_date = datetime.now().strftime('%Y-%m-%d %H:%M')

                weibo_news.append({
                    'title': title,
                    'link': link,
                    'pub_date': pub_date,
                    'source': f'微博: {name}'
                })
                processed += 1

            except Exception as post_error:
                print(f"    [警告] 处理单条微博失败: {post_error}")
                continue

        unique = []
        seen = set()
        for item in weibo_news:
            key = (item['title'][:50], item['pub_date'])
            if key not in seen:
                seen.add(key)
                unique.append(item)

        # 详细日志：帮助定位"获取成功但不显示"的问题
        print(f"  [微博] {name}: 总卡片 {total_cards}, 日期过滤 {filtered_by_date}, 空文本过滤 {filtered_by_empty_text}, 最终 {len(unique)} 条")
        if filtered_by_date > 0:
            print(f"  [微博] ⚠️ {name} 有 {filtered_by_date} 条因超过3天被过滤（如需显示更多可调整 DATE_RANGE_DAYS）")
        return unique

    except Exception as e:
        print(f"  [错误] {name} 抓取失败: {e}")
        traceback.print_exc()
        return []

def fetch_weibo_news() -> List[Dict]:
    all_weibo_news = []
    if not WEIBO_CONFIG['enabled'] or not WEIBO_CONFIG['monitor_uids']:
        print("[微博] 功能已禁用或未配置监控UID")
        return []
    print(f"[INFO] 开始抓取 {len(WEIBO_CONFIG['monitor_uids'])} 个微博用户 (Cookie: {'已配置' if WEIBO_CONFIG['headers'].get('Cookie') else '❌未配置'})")
    for idx, uid in enumerate(WEIBO_CONFIG['monitor_uids']):
        name = WEIBO_CONFIG['monitor_names'][idx] if idx < len(WEIBO_CONFIG['monitor_names']) else f'用户{uid}'
        posts = fetch_weibo_posts(uid, 15, name)
        all_weibo_news.extend(posts)
        time.sleep(2)

    # 汇总诊断
    if len(all_weibo_news) == 0:
        print(f"  [微博] ⚠️ 所有用户均未获取到数据！请检查：")
        print(f"    1. GitHub Secrets > WEIBO_COOKIE 是否已设置且未过期")
        print(f"    2. Cookie 获取方式：浏览器打开 m.weibo.cn 并登录 → F12 → Network → 刷新 → 任选请求 → 复制完整Cookie")
        print(f"    3. 确认监控的UID ({', '.join(WEIBO_CONFIG['monitor_uids'])}) 是否正确")
    else:
        print(f"  [OK] ✅ 微博: 共 {len(all_weibo_news)} 条动态")
    return all_weibo_news

# ==================== B站动态抓取 ====================

def fetch_bilibili_dynamics(uid: str, limit: int = 10) -> List[Dict]:
    dynamics = []
    try:
        url = "https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space"
        params = {"host_mid": uid, "offset": "", "page_size": limit}
        data = fetch_json(url, params)
        if not data or data.get('code') != 0:
            print(f"  [警告] B站 {uid} 返回错误: {data.get('message') if data else '空数据'}")
            return []
        items = data.get('data', {}).get('items', [])
        for item in items[:limit]:
            try:
                modules = item.get('modules', {}) or {}
                author = modules.get('module_author', {}) or {}
                name = author.get('name', f'UID{uid}')
                dynamic = modules.get('module_dynamic', {}) or {}
                desc = dynamic.get('desc', {}) or {}
                title = desc.get('text', '')
                if not title and 'orig' in item:
                    orig = item.get('orig', {}) or {}
                    orig_mod = orig.get('modules', {}) or {}
                    orig_dyn = orig_mod.get('module_dynamic', {}) or {}
                    title = orig_dyn.get('desc', {}).get('text', '')
                    if not title:
                        name = orig_mod.get('module_author', {}).get('name', name)
                if not title:
                    title = f"{name} 发布了新动态"
                if len(title) > 100:
                    title = title[:100] + '...'
                ts = author.get('pub_ts', 0)
                if ts:
                    dt = datetime.fromtimestamp(int(ts))
                    if dt < DATE_THRESHOLD:
                        continue
                    pub_date = dt.strftime('%Y-%m-%d %H:%M')
                else:
                    pub_date = ''
                id_str = item.get('id_str', '')
                link = f"https://t.bilibili.com/{id_str}" if id_str else ''
                dynamics.append({
                    'title': title,
                    'link': link,
                    'pub_date': pub_date,
                    'source': f'B站: {name}'
                })
            except Exception as e:
                continue
        print(f"  [OK] B站 {uid}: {len(dynamics)} 条")
        time.sleep(1)
    except Exception as e:
        print(f"  [错误] B站 {uid}: {e}")
    return dynamics

def fetch_bilibili_news() -> List[Dict]:
    all_news = []
    if not BILIBILI_UIDS:
        return []
    print(f"[INFO] 开始抓取 {len(BILIBILI_UIDS)} 个 B站 UP 主")
    for uid in BILIBILI_UIDS:
        all_news.extend(fetch_bilibili_dynamics(uid))
    print(f"  [OK] B站: 共 {len(all_news)} 条")
    return all_news

# ==================== 永雏塔菲统计 & 音频选择 ====================

def count_taffy_news(all_news: Dict[str, List[Dict]]) -> int:
    """统计所有新闻中包含「永雏塔菲」的条目数"""
    count = 0
    for source, items in all_news.items():
        for item in items:
            title = item.get('title', '')
            if '永雏塔菲' in title or '永雏塔菲' in source:
                count += 1
    return count

def generate_audio_html(use_taffy_audio: bool) -> str:
    """根据选择返回嵌入音频的 HTML 片段（使用 GitHub Raw URL）"""
    audio_url = AUDIO_URL_TAFFY if use_taffy_audio else AUDIO_URL_DEFAULT
    label = "永雏塔菲专属BGM" if use_taffy_audio else "VTuber综合BGM"
    return f'''
    <div style="margin: 15px 0; padding: 10px; background: #f0f0f0; border-radius: 8px;">
        <audio controls style="width: 100%; max-width: 300px;">
            <source src="{audio_url}" type="audio/mpeg">
            您的浏览器不支持音频播放。
        </audio>
        <p style="font-size:12px; color:#666; margin:5px 0 0 0;">🎵 {label}（点击播放）</p>
    </div>
    '''

# ==================== HTML 生成 ====================

def escape_html(text: str) -> str:
    if not text:
        return ''
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

def generate_html(all_news: Dict[str, List[Dict]], audio_html: str = "") -> str:
    total = sum(len(v) for v in all_news.values())
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>VTuber新闻汇总</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #ff6b6b; border-bottom: 3px solid #ff6b6b; }}
h2 {{ color: #4ecdc4; border-left: 4px solid #4ecdc4; padding-left: 12px; margin-top: 25px; }}
.news-item {{ background: #f9f9f9; margin: 15px 0; padding: 12px; border-radius: 8px; }}
.news-title a {{ color: #0066cc; text-decoration: none; }}
.news-meta {{ font-size: 12px; color: #888; }}
.footer {{ margin-top: 30px; padding-top: 15px; border-top: 1px solid #ddd; text-align: center; font-size: 12px; }}
</style>
</head>
<body>
<h1>🎮 VTuber新闻汇总</h1>
<p>抓取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<p>新鲜新闻总数 (近3天内): {total} 条</p>
{audio_html}
"""
    for source, items in all_news.items():
        html += f'<h2>📌 {source}</h2>'
        if not items:
            html += '<p style="color:#888; font-size:13px; padding-left:12px;">近3天暂无新动态</p>'
            continue
        for news in items[:50]:
            html += f'''<div class="news-item">
<div class="news-title"><a href="{news.get('link', '#')}" target="_blank">{escape_html(news.get('title', ''))}</a></div>
<div class="news-meta">日期: {escape_html(news.get('pub_date', ''))} | 来源: {escape_html(source)}</div>
</div>'''
    if total == 0:
        html += '<p>近3天内暂无新新闻，请访问官网查看最新动态。</p>'
    html += f'<div class="footer">本邮件由 GitHub Actions 自动生成 | VTuber新闻订阅 | 只显示最近3天内的新闻</div></body></html>'
    return html

# ==================== 邮件发送 ====================

def send_email(html_body: str, recipients: List[str]):
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS]) or not recipients:
        print("[错误] 邮件配置不完整")
        return
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = f"MoeFace News <{SMTP_USER}>"
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = f"VTuber新闻汇总 - {datetime.now().strftime('%Y-%m-%d')}"
        msg['Date'] = formatdate(localtime=True)
        msg.attach(MIMEText(html_body, 'html', 'utf-8'))
        print(f"[邮件] 连接 {SMTP_HOST}:{SMTP_PORT}")
        if SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
            server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
        server.quit()
        print("[邮件] ✅ 发送成功")
    except Exception as e:
        print(f"[邮件] ❌ 失败: {e}")

# ==================== 主函数 ====================

def main():
    print("=" * 50)
    print("VTuber News Fetcher (近3天新闻)")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"过滤器: 只收录 {DATE_THRESHOLD.strftime('%Y-%m-%d')} 之后的新闻")
    print("=" * 50)
    if not RECIPIENTS:
        print("[警告] 无收件人")
    else:
        print(f"[配置] 收件人数: {len(RECIPIENTS)}")

    all_news = {}
    
    # Hololive & Nijisanji
    for src in NEWS_SOURCES:
        html = fetch_html(src['url'])
        if not html:
            all_news[src['name']] = []
            continue
        if src['parser'] == 'hololive':
            news = parse_hololive_news(html)
        elif src['parser'] == 'nijisanji':
            news = parse_nijisanji_news(html)
        else:
            news = []
        all_news[src['name']] = news

    # B站动态
    bili = fetch_bilibili_news()
    all_news['国产 VTuber 动态'] = bili

    # 微博动态（始终添加分类，即使为空也在邮件中展示区块）
    weibo = fetch_weibo_news()
    all_news['微博 VTuber 动态'] = weibo

    total = sum(len(v) for v in all_news.values())
    print(f"\n总共抓取到 {total} 条近3天内新闻")

    # ---- 永雏塔菲统计与音频选择 ----
    taffy_cnt = count_taffy_news(all_news)
    print(f"[统计] 包含「永雏塔菲」的新闻数: {taffy_cnt} / {total}")
    # 决策条件：taffy_cnt >= TAFFY_THRESHOLD 且 占比 >= TAFFY_RATIO_THRESHOLD (且 total>0)
    use_taffy_audio = (total > 0 and 
                       taffy_cnt >= TAFFY_THRESHOLD and 
                       (taffy_cnt / total) >= TAFFY_RATIO_THRESHOLD)
    print(f"[音频] 选择: {'yctf.mp3 (永雏塔菲专属)' if use_taffy_audio else 'vtb.mp3 (综合)'} (永雏塔菲较多? {use_taffy_audio})")
    
    audio_html = generate_audio_html(use_taffy_audio)
    html_body = generate_html(all_news, audio_html)

    if SMTP_USER and RECIPIENTS:
        send_email(html_body, RECIPIENTS)
    else:
        with open('vtuber_news_report.md', 'w', encoding='utf-8') as f:
            f.write(f"# VTuber新闻汇总\n抓取时间: {datetime.now()}\n新闻数量: {total}\n\n")
            for src, items in all_news.items():
                if items:
                    f.write(f"## {src}\n")
                    for it in items:
                        f.write(f"- [{it['title']}]({it['link']})\n")
                    f.write("\n")
        print("已保存到 vtuber_news_report.md")

if __name__ == "__main__":
    main()