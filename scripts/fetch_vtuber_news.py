#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import smtplib
import time
from email.mime.text import MIMEText
from email.header import Header
from typing import List, Dict, Tuple
from datetime import datetime

import requests
import feedparser
from bs4 import BeautifulSoup

# ========== 配置部分 ==========
# 环境变量（由 GitHub Secrets 提供）
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.qq.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 465))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
TO_EMAIL = os.environ.get("TO_EMAIL", "")

# 新闻源配置
# 格式: "显示名称": {"type": "rss" 或 "web", "url": "地址", "selector": CSS选择器(仅web)}
# 注意：很多官方RSS已失效，这里使用已验证可用的源
NEWS_SOURCES = {
    "Hololive 官方新闻": {
        "type": "web",
        "url": "https://hololivepro.com/news/",
        "selector": "article"  # 通用article标签
    },
    "VTuber 行业动态 (PANORA)": {
        "type": "web",
        "url": "https://panora.tokyo/category/vtuber/",
        "selector": "article"
    },
    # 以下源已经失效，注释掉避免警告
    # "Hololive Production RSS": {"type": "rss", "url": "https://hololive.so-hololive.com/news.rss"},
    # "Nijisanji RSS": {"type": "rss", "url": "https://www.nijisanji.jp/rss.xml"},
}

# 需要过滤的关键词（不包含则过滤掉）
ALLOWED_KEYWORDS = ["hololive", "holostars", "nijisanji", "vtuber", "直播", "新衣服", "演唱会", "毕业", "联动"]

# ========== 工具函数 ==========
def fetch_rss(url: str) -> List[Dict]:
    """抓取RSS源，返回文章列表"""
    try:
        feed = feedparser.parse(url)
        entries = []
        for entry in feed.entries[:10]:  # 最多10条
            entries.append({
                'title': entry.get('title', '无标题'),
                'link': entry.get('link', '#'),
                'summary': entry.get('summary', ''),
                'published': entry.get('published', '')
            })
        return entries
    except Exception as e:
        print(f"    RSS抓取失败: {e}")
        return []

def fetch_website(url: str, selector: str) -> List[Dict]:
    """抓取网页，使用CSS选择器提取文章"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        articles = soup.select(selector)
        results = []
        for article in articles[:10]:
            # 尝试提取链接和标题
            a_tag = article.find('a')
            if a_tag:
                link = a_tag.get('href')
                if link and not link.startswith('http'):
                    link = requests.compat.urljoin(url, link)
                title = a_tag.get_text(strip=True)
            else:
                title = article.get_text(strip=True)
                link = '#'
            if title and len(title) > 5:
                results.append({
                    'title': title[:120],
                    'link': link,
                    'summary': '',
                    'published': datetime.now().strftime('%Y-%m-%d')
                })
        return results
    except Exception as e:
        print(f"    网页抓取失败: {e}")
        return []

def filter_news(articles: List[Dict]) -> List[Dict]:
    """根据关键词过滤新闻"""
    filtered = []
    for art in articles:
        text = (art['title'] + ' ' + art['summary']).lower()
        if any(kw.lower() in text for kw in ALLOWED_KEYWORDS):
            filtered.append(art)
    return filtered

def send_email(html_content: str):
    """发送HTML邮件（使用SSL）"""
    if not html_content or len(html_content) < 100:
        print("没有有效新闻内容，跳过邮件发送")
        return

    # 构造HTML邮件
    msg = MIMEText(html_content, 'html', 'utf-8')
    msg['From'] = Header(f"VTuber新闻机器人 <{SMTP_USER}>", 'utf-8')
    msg['To'] = TO_EMAIL
    msg['Subject'] = Header(f"VTuber 新闻汇总 - {datetime.now().strftime('%Y-%m-%d')}", 'utf-8')

    # 使用 SSL 连接
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [TO_EMAIL], msg.as_string())
        print(f"✅ 邮件发送成功至 {TO_EMAIL}")
    except smtplib.SMTPAuthenticationError:
        print("❌ SMTP 认证失败，请检查邮箱和授权码")
    except smtplib.SMTPServerDisconnected as e:
        print(f"❌ 服务器断开连接: {e}")
    except Exception as e:
        print(f"❌ 发送邮件失败: {e}")

def generate_html(news_data: Dict[str, List[Dict]]) -> str:
    """生成HTML报告"""
    html = f"""
    <html>
    <head><meta charset="UTF-8"></head>
    <body>
        <h1>🎤 VTuber 新闻汇总</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <hr/>
    """
    for source, articles in news_data.items():
        html += f"<h2>📌 {source}</h2><ul>"
        if not articles:
            html += "<li>暂无新闻</li>"
        for art in articles:
            html += f'<li><a href="{art["link"]}">{art["title"]}</a>'
            if art.get('published'):
                html += f' <small>({art["published"]})</small>'
            html += '</li>'
        html += "</ul>"
    html += "</body></html>"
    return html

# ========== 主流程 ==========
def main():
    print("=" * 50)
    print("VTuber News Fetcher 启动")
    print("=" * 50)

    all_news = {}
    total_raw = 0

    for name, config in NEWS_SOURCES.items():
        print(f"正在抓取: {name}")
        if config['type'] == 'rss':
            articles = fetch_rss(config['url'])
        else:
            articles = fetch_website(config['url'], config.get('selector', 'a'))
        raw_count = len(articles)
        total_raw += raw_count
        filtered = filter_news(articles)
        all_news[name] = filtered
        print(f"  原始: {raw_count} 条 → 过滤后: {len(filtered)} 条")

    print(f"\n总计抓取: {total_raw} 条原始新闻，过滤后有效新闻: {sum(len(v) for v in all_news.values())} 条")

    # 生成HTML并发送
    html_body = generate_html(all_news)
    send_email(html_body)

if __name__ == "__main__":
    main()