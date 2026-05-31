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
import feedparser
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

NEWS_SOURCES = [
    {'name': 'Hololive Production', 'url': 'https://hololivepro.com/news/', 'type': 'html', 'parser': 'hololive'},
    {'name': 'Nijisanji', 'url': 'https://www.nijisanji.jp/news', 'type': 'html', 'parser': 'nijisanji'}
]

BILIBILI_UIDS = ['33064694', '1265680561']

# 微博配置 - 多重备用策略
WEIBO_USERS = [
    {'uid': '7618923072', 'name': '永雏塔菲'}
]

# RSSHub 备用实例
RSSHUB_INSTANCES = [
    "https://rsshub.app",
    "https://rsshub.rssforever.com",
    "https://rsshub.feeded.xyz",
    "https://hub.slarker.me",
]

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
}
if BILI_COOKIE:
    HEADERS['Cookie'] = BILI_COOKIE
    print("[配置] ✅ B站 Cookie 已加载")
else:
    print("[配置] ⚠️ B站 Cookie 未配置")

# ==================== WBI 签名 ====================
MIXIN_KEY_ENC_TAB = [46,47,18,2,53,8,23,32,15,50,10,31,58,3,45,35,27,43,5,49,33,9,42,19,29,28,14,39,12,38,41,13,37,48,7,16,24,55,40,61,26,17,0,1,60,51,30,4,22,25,54,21,56,59,6,63,57,62,11,36,20,34,44,52]

def get_wbi_keys():
    try:
        r = requests.get('https://api.bilibili.com/x/web-interface/nav', headers=HEADERS, timeout=10)
        data = r.json()
        if data.get('code') != 0:
            return '', ''
        img = data['data']['wbi_img']['img_url'].rsplit('/',1)[-1].split('.')[0]
        sub = data['data']['wbi_img']['sub_url'].rsplit('/',1)[-1].split('.')[0]
        return img, sub
    except:
        return '', ''

def encrypt_wbi(params, img_key, sub_key):
    key_str = img_key + sub_key
    if len(key_str) < 64:
        return params
    mixin_key = ''.join(key_str[MIXIN_KEY_ENC_TAB[i]] for i in range(64))[:32]
    params['wts'] = int(time.time())
    sorted_params = sorted(params.items())
    query = urllib.parse.urlencode(sorted_params)
    params['w_rid'] = hashlib.md5((query + mixin_key).encode()).hexdigest()
    return params

def sign_request(url, params):
    img_key, sub_key = get_wbi_keys()
    if not img_key or not sub_key:
        return url, params
    return url, encrypt_wbi(params.copy(), img_key, sub_key)

def fetch_html(url, timeout=15):
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        r.raise_for_status()
        r.encoding = 'utf-8'
        return r.text
    except Exception as e:
        print(f"  [错误] 抓取失败 {url}: {e}")
        return None

def fetch_json(url, params=None, timeout=15):
    try:
        if params is None:
            params = {}
        signed_url, signed_params = sign_request(url, params)
        r = requests.get(signed_url, params=signed_params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [错误] 抓取失败 {url}: {e}")
        return None

# ==================== 日期辅助 ====================
def normalize_date(date_str):
    if not date_str:
        return None
    m = re.match(r'(\d{4})[./-](\d{1,2})[./-](\d{1,2})', date_str.strip())
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r'(\d{4})年(\d{1,2})月(\d{1,2})日', date_str)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None

def is_recent(date_str, days=3):
    if not date_str:
        return False
    try:
        if ' ' in date_str:
            dt = datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d')
        else:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
        return dt >= datetime.now() - timedelta(days=days)
    except:
        return False

def parse_weibo_datetime(dt_str):
    try:
        cleaned = re.sub(r'\+\d{4}', '', dt_str)
        dt = datetime.strptime(cleaned.strip(), '%a %b %d %H:%M:%S %Y')
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        m = re.search(r'(\d{4}-\d{2}-\d{2})', dt_str)
        if m:
            return m.group(1)
        return datetime.now().strftime('%Y-%m-%d %H:%M')

# ==================== Hololive ====================
def parse_hololive_news(html):
    news_list = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        articles = soup.find_all('div', class_=re.compile(r'news|post|article|entry', re.I))
        if not articles:
            articles = soup.find_all(['li', 'div'], class_=True)
        for article in articles[:15]:
            date_elem = article.find('time') or article.find(class_=re.compile(r'date|time', re.I))
            if date_elem:
                pub_date_raw = re.sub(r'[年月]', '.', date_elem.get_text(strip=True)).replace('日', '').strip('.')
            else:
                text = article.get_text()
                m = re.search(r'(\d{4}[.-]\d{1,2}[.-]\d{1,2})', text)
                if not m:
                    continue
                pub_date_raw = m.group(1)
            pub_date = normalize_date(pub_date_raw)
            if not pub_date or not is_recent(pub_date):
                continue
            link_elem = article.find('a', href=True)
            if not link_elem:
                continue
            link = link_elem.get('href')
            if link and not link.startswith('http'):
                link = urljoin('https://hololivepro.com', link)
            title = link_elem.get_text(strip=True)
            if not title or len(title) < 3:
                title_elem = article.find(['h2','h3','h4','p','span'])
                title = title_elem.get_text(strip=True) if title_elem else ""
            if title and len(title) > 5:
                title = re.sub(r'^\d{4}[.-]\d{1,2}[.-]\d{1,2}\s*', '', title)
                news_list.append({'title': title[:150], 'link': link, 'pub_date': pub_date, 'source': 'Hololive Production'})
        print(f"  [OK] Hololive Production: {len(news_list)} 条")
    except Exception as e:
        print(f"  [错误] Hololive 解析失败: {e}")
    return news_list

# ==================== Nijisanji ====================
def parse_nijisanji_news(html):
    news_list = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        items = soup.find_all('li', class_=re.compile(r'news|post|item', re.I))
        if not items:
            items = soup.find_all(['li','div'], class_=True)
        for item in items[:20]:
            text = item.get_text()
            m = re.search(r'(\d{4}[./-]\d{1,2}[./-]\d{1,2})', text)
            if not m:
                continue
            pub_date_raw = m.group(1).replace('/','.')
            pub_date = normalize_date(pub_date_raw)
            if not pub_date or not is_recent(pub_date):
                continue
            link_elem = item.find('a', href=True)
            if not link_elem:
                continue
            link = link_elem.get('href')
            if link and not link.startswith('http'):
                link = urljoin('https://www.nijisanji.jp', link)
            title = link_elem.get_text(strip=True)
            if not title or len(title) < 3:
                title = re.sub(r'^\d{4}[./-]\d{1,2}[./-]\d{1,2}\s*', '', text).strip()
            if title:
                news_list.append({'title': title[:150], 'link': link, 'pub_date': pub_date, 'source': 'Nijisanji'})
        print(f"  [OK] Nijisanji: {len(news_list)} 条")
    except Exception as e:
        print(f"  [错误] Nijisanji 解析失败: {e}")
    return news_list

# ==================== 微博抓取（多重备用策略）====================
def fetch_weibo_via_render_data(uid, name, limit=15):
    """策略1: 页面内嵌 $render_data"""
    try:
        url = f"https://m.weibo.cn/u/{uid}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9',
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        html = resp.text
        # 多种可能的匹配模式
        patterns = [
            r'window\.\$render_data\s*=\s*({.*?})\s*</script>',
            r'window\.\$render_data\s*=\s*({.*?});\s*</script>',
            r'<script>window\.\$render_data\s*=\s*({.*?})</script>',
        ]
        data = None
        for pattern in patterns:
            match = re.search(pattern, html, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    break
                except:
                    continue
        if not data:
            return None
        cards = None
        if 'data' in data and 'cards' in data['data']:
            cards = data['data']['cards']
        elif 'cards' in data:
            cards = data['cards']
        if not cards:
            return None
        return parse_weibo_cards(cards, uid, name, limit)
    except Exception as e:
        print(f"    [策略1失败] {e}")
        return None

def fetch_weibo_via_api(uid, name, limit=15):
    """策略2: 直接调用 API（无 Cookie，可能被限）"""
    try:
        containerid = f"107603{uid}"
        url = f"https://m.weibo.cn/api/container/getIndex?type=uid&value={uid}&containerid={containerid}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://m.weibo.cn/',
            'Accept': 'application/json, text/plain, */*',
            'X-Requested-With': 'XMLHttpRequest',
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get('ok') != 1:
            return None
        cards = data.get('data', {}).get('cards', [])
        if not cards:
            return None
        return parse_weibo_cards(cards, uid, name, limit)
    except Exception as e:
        print(f"    [策略2失败] {e}")
        return None

def fetch_weibo_via_rss(uid, name, limit=15):
    """策略3: RSSHub 公共实例"""
    for instance in RSSHUB_INSTANCES:
        try:
            rss_url = f"{instance}/weibo/user/{uid}"
            feed = feedparser.parse(rss_url)
            if feed.bozo == 0 and feed.entries:
                news = []
                for entry in feed.entries[:limit]:
                    title = entry.get('title', '').strip()
                    if not title:
                        continue
                    link = entry.get('link', '')
                    pub_date_raw = entry.get('published', '')
                    if pub_date_raw:
                        try:
                            from email.utils import parsedate_to_datetime
                            dt = parsedate_to_datetime(pub_date_raw)
                            if dt < datetime.now() - timedelta(days=3):
                                continue
                            pub_date = dt.strftime('%Y-%m-%d %H:%M')
                        except:
                            pub_date = pub_date_raw
                    else:
                        pub_date = datetime.now().strftime('%Y-%m-%d %H:%M')
                    if len(title) > 200:
                        title = title[:197] + "..."
                    news.append({'title': title, 'link': link, 'pub_date': pub_date, 'source': f'微博: {name}'})
                if news:
                    print(f"    [策略3成功] 使用 {instance}")
                    return news
        except:
            continue
    return None

def parse_weibo_cards(cards, uid, name, limit):
    """解析微博卡片数据"""
    news = []
    for card in cards[:limit*2]:
        mblog = card.get('mblog')
        if not mblog:
            continue
        raw_text = mblog.get('text', '')
        title = re.sub('<[^<]+?>', '', raw_text).strip()
        if not title:
            if mblog.get('retweeted_status'):
                title = "转发微博"
            else:
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
                continue
        else:
            pub_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        news.append({'title': title, 'link': link, 'pub_date': pub_date, 'source': f'微博: {name}'})
        if len(news) >= limit:
            break
    return news

def fetch_weibo_posts(uid, name, limit=15):
    """主函数：依次尝试三种策略"""
    print(f"  [微博] 抓取: {name} (UID: {uid})")
    
    # 策略1: 页面内嵌数据
    result = fetch_weibo_via_render_data(uid, name, limit)
    if result is not None:
        print(f"  [OK] {name}: 策略1成功，获取 {len(result)} 条")
        return result
    
    # 策略2: API 调用
    result = fetch_weibo_via_api(uid, name, limit)
    if result is not None:
        print(f"  [OK] {name}: 策略2成功，获取 {len(result)} 条")
        return result
    
    # 策略3: RSSHub
    result = fetch_weibo_via_rss(uid, name, limit)
    if result is not None:
        print(f"  [OK] {name}: 策略3成功，获取 {len(result)} 条")
        return result
    
    print(f"  [错误] {name}: 所有策略均失败")
    return []

def fetch_weibo_news():
    all_news = []
    print(f"[INFO] 开始抓取 {len(WEIBO_USERS)} 个微博用户 (多重策略)")
    for user in WEIBO_USERS:
        posts = fetch_weibo_posts(user['uid'], user['name'], 15)
        all_news.extend(posts)
        time.sleep(2)
    print(f"  [OK] 微博: 共 {len(all_news)} 条")
    return all_news

# ==================== B站动态 ====================
def fetch_bilibili_dynamics(uid, limit=10):
    dynamics = []
    try:
        url = "https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space"
        params = {"host_mid": uid, "offset": "", "page_size": limit}
        data = fetch_json(url, params)
        if not data or data.get('code') != 0:
            print(f"  [警告] B站 {uid} 错误: {data.get('message') if data else '空'}")
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
                dynamics.append({'title': title, 'link': link, 'pub_date': pub_date, 'source': f'B站: {name}'})
            except:
                continue
        print(f"  [OK] B站 {uid}: {len(dynamics)} 条")
        time.sleep(1)
    except Exception as e:
        print(f"  [错误] B站 {uid}: {e}")
    return dynamics

def fetch_bilibili_news():
    all_news = []
    if not BILIBILI_UIDS:
        return []
    print(f"[INFO] 开始抓取 {len(BILIBILI_UIDS)} 个 B站 UP 主")
    for uid in BILIBILI_UIDS:
        all_news.extend(fetch_bilibili_dynamics(uid))
    print(f"  [OK] B站: 共 {len(all_news)} 条")
    return all_news

# ==================== HTML 生成 ====================
def escape_html(text):
    if not text:
        return ''
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')

def generate_html(all_news):
    total = sum(len(v) for v in all_news.values())
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>VTuber新闻汇总</title>
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
"""
    for source, items in all_news.items():
        if not items:
            continue
        html += f'<h2>📌 {source}</h2>'
        for news in items[:50]:
            html += f'''<div class="news-item">
<div class="news-title"><a href="{news.get('link', '#')}" target="_blank">{escape_html(news.get('title', ''))}</a></div>
<div class="news-meta">日期: {escape_html(news.get('pub_date', ''))} | 来源: {escape_html(source)}</div>
</div>'''
    if total == 0:
        html += '<p>近3天内暂无新新闻，请访问官网查看最新动态。</p>'
    html += '<div class="footer">本邮件由 GitHub Actions 自动生成 | VTuber新闻订阅 | 只显示最近3天内的新闻</div></body></html>'
    return html

# ==================== 邮件发送 ====================
def send_email(html_body, recipients):
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
    print("VTuber News Fetcher (近3天新闻) - 多重策略版")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"过滤器: 只收录 {DATE_THRESHOLD.strftime('%Y-%m-%d')} 之后的新闻")
    print("=" * 50)
    if not RECIPIENTS:
        print("[警告] 无收件人")
    else:
        print(f"[配置] 收件人数: {len(RECIPIENTS)}")

    all_news = {}
    
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

    bili = fetch_bilibili_news()
    if bili:
        all_news['国产 VTuber 动态'] = bili

    weibo = fetch_weibo_news()
    if weibo:
        all_news['微博 VTuber 动态'] = weibo

    total = sum(len(v) for v in all_news.values())
    print(f"\n总共抓取到 {total} 条近3天内新闻")
    html_body = generate_html(all_news)

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