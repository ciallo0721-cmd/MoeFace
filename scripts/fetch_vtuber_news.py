#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import smtplib
import requests
import json
import time
from bs4 import BeautifulSoup
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin

# ==================== 配置 ====================
# 新闻来源配置
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

# 国产 VTuber B站 UID 列表（请替换为你关注的 VTuber UID）
# 获取方式：打开 VTuber 的 B站空间主页，URL 中的数字即为 UID
# 例如：https://space.bilibili.com/12345678 → UID 为 12345678
BILIBILI_UIDS = [
    '33064694',    # 请替换为实际 VTuber 的 UID
    '1265680561',  # 请替换为实际 VTuber 的 UID
]

# 邮件配置（从环境变量读取）
SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.qq.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '465'))
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PASS = os.getenv('SMTP_PASS', '')
TO_EMAIL = os.getenv('TO_EMAIL', '')

# User-Agent 伪装
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# ==================== 工具函数 ====================

def fetch_html(url: str, timeout: int = 15) -> Optional[str]:
    """抓取 HTML 内容"""
    try:
        response = requests.get(url, timeout=timeout, headers=HEADERS)
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.text
    except requests.RequestException as e:
        print(f"  [错误] 抓取失败 {url}: {e}")
        return None

def fetch_json(url: str, timeout: int = 15) -> Optional[dict]:
    """抓取 JSON 数据"""
    try:
        response = requests.get(url, timeout=timeout, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"  [错误] 抓取失败 {url}: {e}")
        return None

# ==================== Hololive 解析 ====================

def parse_hololive_news(html: str) -> List[Dict]:
    """
    解析 Hololive 新闻页面
    基于官方 HTML 结构：日期用 <time> 标签，标题在紧随其后的 <a> 或 <h2> 中
    """
    news_list = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # 查找所有新闻条目容器
        articles = soup.find_all('div', class_=re.compile(r'news|post|article|entry', re.I))
        
        # 备用：直接找带有日期和标题的结构
        if not articles or len(articles) < 2:
            articles = soup.find_all(['li', 'div'], class_=True)
        
        for article in articles[:15]:
            # 查找日期
            date_elem = None
            # 优先找 <time> 标签
            date_elem = article.find('time')
            if not date_elem:
                date_elem = article.find(class_=re.compile(r'date|time', re.I))
            if not date_elem:
                # 查找形如 2026.04.27 或 2026-04-27 格式的文本
                text = article.get_text()
                date_match = re.search(r'(\d{4}[.-]\d{1,2}[.-]\d{1,2})', text)
                if date_match:
                    pub_date = date_match.group(1)
                else:
                    continue
            else:
                pub_date = date_elem.get_text(strip=True)
                # 统一日期格式
                pub_date = re.sub(r'[年月]', '.', pub_date).replace('日', '').strip('.')

            # 查找标题和链接
            link_elem = article.find('a', href=True)
            if not link_elem:
                # 可能是标题在 h2/h3 内，但外层没有包裹 a 标签
                title_elem = article.find(['h2', 'h3', 'h4', 'p', 'span'])
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    # 过滤掉日期和过短的文本
                    if title and len(title) > 5 and title != pub_date:
                        news_list.append({
                            'title': title,
                            'link': '#',  # 无链接时占位
                            'pub_date': pub_date,
                            'source': 'Hololive Production'
                        })
                continue

            link = link_elem.get('href', '')
            if link and not link.startswith('http'):
                link = urljoin('https://hololivepro.com', link)

            # 标题优先从 a 标签获取
            title = link_elem.get_text(strip=True)
            if not title or len(title) < 3:
                # 从 h2/h3 等标签获取
                title_elem = article.find(['h2', 'h3', 'h4', 'p', 'span'])
                title = title_elem.get_text(strip=True) if title_elem else ""

            # 过滤掉无效条目
            if title and len(title) > 5 and title != pub_date:
                # 去除日期前缀
                title = re.sub(r'^\d{4}[.-]\d{1,2}[.-]\d{1,2}\s*', '', title)
                news_list.append({
                    'title': title,
                    'link': link if link != '#' else '',
                    'pub_date': pub_date,
                    'source': 'Hololive Production'
                })

        print(f"  [OK] Hololive Production: 解析到 {len(news_list)} 条新闻")
        
    except Exception as e:
        print(f"  [错误] Hololive 解析失败: {e}")
    
    return news_list

# ==================== Nijisanji 解析 ====================

def parse_nijisanji_news(html: str) -> List[Dict]:
    """
    解析 Nijisanji 新闻页面
    基于官方 HTML 结构：新闻以列表形式展示，包含日期和标题
    """
    news_list = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # 查找所有可能包含新闻的 li 标签
        items = soup.find_all('li', class_=re.compile(r'news|post|item', re.I))
        if not items:
            items = soup.find_all(['li', 'div'], class_=True)
        
        # 备用：直接找带日期的段落
        if not items:
            paragraphs = soup.find_all(['p', 'div'])
            items = [p for p in paragraphs if re.search(r'\d{4}[./]\d{1,2}[./]\d{1,2}', p.get_text())]
        
        for item in items[:20]:
            text = item.get_text()
            
            # 提取日期（格式：2026.04.27 或 2026-04-27）
            date_match = re.search(r'(\d{4}[./-]\d{1,2}[./-]\d{1,2})', text)
            if not date_match:
                continue
            pub_date = date_match.group(1).replace('/', '.')
            
            # 查找链接
            link_elem = item.find('a', href=True)
            if not link_elem:
                continue
            
            link = link_elem.get('href', '')
            if link and not link.startswith('http'):
                link = urljoin('https://www.nijisanji.jp', link)
            
            # 提取标题（排除日期部分）
            title = link_elem.get_text(strip=True)
            if not title or len(title) < 3:
                title = re.sub(r'^\d{4}[./-]\d{1,2}[./-]\d{1,2}\s*', '', text).strip()
            
            if title and len(title) > 3:
                news_list.append({
                    'title': title,
                    'link': link,
                    'pub_date': pub_date,
                    'source': 'Nijisanji'
                })
        
        print(f"  [OK] Nijisanji: 解析到 {len(news_list)} 条新闻")
        
    except Exception as e:
        print(f"  [错误] Nijisanji 解析失败: {e}")
    
    return news_list

# ==================== B站动态解析 ====================

def fetch_bilibili_dynamics(uid: str, limit: int = 5) -> List[Dict]:
    """
    通过 B站 API 获取指定 UP 主的动态
    API 来源：https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space
    """
    dynamics = []
    try:
        url = f"https://api.bilibili.com/x/polymer/web-dynamic/v1/feed/space?host_mid={uid}&offset=&page_size={limit}"
        data = fetch_json(url)
        
        if not data or data.get('code') != 0:
            print(f"  [警告] B站 API 返回错误，UID: {uid}")
            return []
        
        items = data.get('data', {}).get('items', [])
        for item in items[:limit]:
            # 提取动态内容
            modules = item.get('modules', {})
            desc = modules.get('module_dynamic', {}).get('desc', {})
            title = desc.get('text', '')
            
            # 如果是转发动态，提取原始内容
            if not title and 'orig' in modules:
                orig = modules.get('orig', {}).get('modules', {})
                orig_desc = orig.get('module_dynamic', {}).get('desc', {})
                title = orig_desc.get('text', '')
            
            if not title:
                # 尝试从其他字段提取
                title = modules.get('module_author', {}).get('name', '') + ' 发布了新动态'
            
            # 截取过长的内容
            if len(title) > 100:
                title = title[:100] + '...'
            
            # 获取发布时间
            timestamp = item.get('timestamp', 0)
            if timestamp:
                pub_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
            else:
                pub_date = ''
            
            # 获取链接
            link = f"https://t.bilibili.com/{item.get('id_str', '')}" if item.get('id_str') else ''
            
            if title:
                dynamics.append({
                    'title': title,
                    'link': link,
                    'pub_date': pub_date,
                    'source': f'B站 UP主 {uid}'
                })
        
        if dynamics:
            print(f"  [OK] B站 UID {uid}: 获取到 {len(dynamics)} 条动态")
        
    except Exception as e:
        print(f"  [错误] B站动态获取失败 UID {uid}: {e}")
    
    return dynamics

def fetch_bilibili_news() -> List[Dict]:
    """批量获取所有配置的 B站 UP 主动态"""
    all_dynamics = []
    if not BILIBILI_UIDS:
        print("  [提示] 未配置 B站 UID 列表，跳过国产 VTuber 动态")
        return []
    
    print(f"[INFO] 开始抓取 {len(BILIBILI_UIDS)} 个 B站 UP 主动态")
    for uid in BILIBILI_UIDS:
        dynamics = fetch_bilibili_dynamics(uid)
        all_dynamics.extend(dynamics)
        time.sleep(0.5)  # 避免请求过快
    
    print(f"  [OK] B站动态: 总共抓取到 {len(all_dynamics)} 条")
    return all_dynamics

# ==================== HTML 生成 ====================

def generate_html(all_news: Dict[str, List[Dict]]) -> str:
    """生成HTML邮件内容"""
    total_news = sum(len(items) for items in all_news.values())
    
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
        <p>新闻总数: {total_news} 条</p>
    """
    
    displayed_news = 0
    for source_name, news_list in all_news.items():
        if news_list:
            html += f'<h2>📌 {source_name}</h2>'
            for news in news_list:
                displayed_news += 1
                link = news.get('link', '')
                title = news.get('title', '无标题')
                pub_date = news.get('pub_date', '')
                html += f'''
                <div class="news-item">
                    <div class="news-title">
                        <a href="{link}" target="_blank">{title}</a>
                    </div>
                    <div class="news-meta">
                        日期: {pub_date if pub_date else '未知'} | 来源: {source_name}
                    </div>
                </div>
                '''
    
    if displayed_news == 0:
        html += '<p>暂无新新闻，请访问官网查看最新动态。</p>'
    
    html += f"""
        <div class="footer">
            <p>本邮件由 GitHub Actions 自动生成 | VTuber新闻订阅</p>
            <p>📧 如有问题请联系管理员</p>
        </div>
    </body>
    </html>
    """
    return html

# ==================== 邮件发送 ====================

def send_email(html_body: str):
    """发送邮件"""
    if not all([SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, TO_EMAIL]):
        print("[错误] 邮件配置不完整，请检查环境变量")
        print(f"  SMTP_HOST: {'已设置' if SMTP_HOST else '未设置'}")
        print(f"  SMTP_USER: {'已设置' if SMTP_USER else '未设置'}")
        print(f"  TO_EMAIL: {'已设置' if TO_EMAIL else '未设置'}")
        return
    
    try:
        # 创建邮件
        msg = MIMEMultipart('alternative')
        
        # ✅ 关键修复：From字段使用简单格式，不使用Header编码
        # QQ邮箱要求格式: "昵称 <邮箱地址>"，中间必须有空格
        msg['From'] = f"MoeFace News <{SMTP_USER}>"
        
        msg['To'] = TO_EMAIL
        msg['Subject'] = f"VTuber新闻汇总 - {datetime.now().strftime('%Y-%m-%d')}"
        msg['Date'] = formatdate(localtime=True)
        
        # 添加HTML内容
        html_part = MIMEText(html_body, 'html', 'utf-8')
        msg.attach(html_part)
        
        # 连接并发送
        print(f"[邮件] 正在连接 {SMTP_HOST}:{SMTP_PORT}")
        
        if SMTP_PORT == 465:
            # SSL连接
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30)
        else:
            # STARTTLS连接
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
            server.starttls()
        
        server.login(SMTP_USER, SMTP_PASS)
        print(f"[邮件] 登录成功，正在发送到 {TO_EMAIL}")
        
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

# ==================== 主函数 ====================

def main():
    print("=" * 50)
    print("VTuber News Fetcher 启动")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # 检查邮件配置
    if not SMTP_USER:
        print("[警告] SMTP_USER 未设置，将只抓取新闻不发送邮件")
    if not TO_EMAIL:
        print("[警告] TO_EMAIL 未设置，将只抓取新闻不发送邮件")
    
    # 抓取新闻
    all_news = {}
    
    # 抓取常规新闻源
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
    
    # 抓取 B站动态
    bilibili_news = fetch_bilibili_news()
    if bilibili_news:
        all_news['国产 VTuber 动态'] = bilibili_news
    
    total_news = sum(len(items) for items in all_news.values())
    print(f"\n总共抓取到 {total_news} 条新闻")
    
    # 生成HTML并发送
    html_body = generate_html(all_news)
    
    if SMTP_USER and TO_EMAIL:
        send_email(html_body)
    else:
        print(f"成功抓取 {total_news} 条新闻，但邮件配置不完整，未发送")
        # 保存到文件作为备份
        with open('vtuber_news_report.md', 'w', encoding='utf-8') as f:
            f.write(f"# VTuber新闻汇总\n\n")
            f.write(f"抓取时间: {datetime.now()}\n")
            f.write(f"新闻数量: {total_news}\n\n")
            for source_name, news_list in all_news.items():
                if news_list:
                    f.write(f"## {source_name}\n\n")
                    for news in news_list:
                        f.write(f"- [{news['title']}]({news['link']})\n")
                    f.write("\n")
        print("新闻已保存到 vtuber_news_report.md")

if __name__ == "__main__":
    main()