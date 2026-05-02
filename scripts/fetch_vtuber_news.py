#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import smtplib
import requests
from bs4 import BeautifulSoup
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate
from datetime import datetime
from typing import List, Dict, Tuple
import json
import time

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

# 邮件配置（从环境变量读取）
SMTP_HOST = os.getenv('SMTP_HOST', 'smtp.qq.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '465'))
SMTP_USER = os.getenv('SMTP_USER', '')
SMTP_PASS = os.getenv('SMTP_PASS', '')
TO_EMAIL = os.getenv('TO_EMAIL', '')

# ==================== 工具函数 ====================

def fetch_html(url: str, timeout: int = 10) -> str:
    """抓取HTML内容"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        response.encoding = 'utf-8'
        return response.text
    except Exception as e:
        print(f"  [错误] 抓取失败: {e}")
        return ""

def parse_hololive_news(html: str) -> List[Dict]:
    """解析 Hololive 新闻页面"""
    news_list = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        # 查找新闻列表项 - 根据实际HTML结构调整选择器
        articles = soup.find_all('a', class_='p-post-thumb')
        if not articles:
            # 备用选择器
            articles = soup.find_all('article', class_='p-post')
        if not articles:
            # 更通用的选择器
            articles = soup.find_all(['a', 'article'], href=True)
            articles = [a for a in articles if '/news/' in a.get('href', '')][:10]
        
        for article in articles[:10]:  # 最多取10条
            title_elem = article.find(['h2', 'h3', 'p', 'span'])
            title = title_elem.get_text(strip=True) if title_elem else "无标题"
            
            link = article.get('href', '')
            if link and not link.startswith('http'):
                link = 'https://hololivepro.com' + link
            
            # 尝试获取日期
            date_elem = article.find('time', class_='c-time')
            if not date_elem:
                date_elem = article.find(class_=re.compile(r'date|time'))
            pub_date = date_elem.get_text(strip=True) if date_elem else ""
            
            if title and title != "无标题":
                news_list.append({
                    'title': title,
                    'link': link,
                    'pub_date': pub_date,
                    'source': 'Hololive Production'
                })
    except Exception as e:
        print(f"  [错误] 解析失败: {e}")
    return news_list

def parse_nijisanji_news(html: str) -> List[Dict]:
    """解析 Nijisanji 新闻页面"""
    news_list = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        # 查找新闻链接
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link.get('href', '')
            # 过滤新闻链接
            if '/news/' in href or '/information/' in href:
                title = link.get_text(strip=True)
                if title and len(title) > 5:  # 过滤太短的文本
                    if not href.startswith('http'):
                        href = 'https://www.nijisanji.jp' + href
                    news_list.append({
                        'title': title,
                        'link': href,
                        'pub_date': "",
                        'source': 'Nijisanji'
                    })
                    if len(news_list) >= 10:
                        break
        
        # 如果还是没找到，尝试找所有带文本的链接
        if not news_list:
            for link in links:
                title = link.get_text(strip=True)
                if title and len(title) > 10 and title not in ['READ MORE', '続きを読む', '詳しくはこちら']:
                    href = link.get('href', '')
                    if href and not href.startswith('#') and not href.startswith('javascript'):
                        if not href.startswith('http'):
                            href = 'https://www.nijisanji.jp' + href
                        news_list.append({
                            'title': title,
                            'link': href,
                            'pub_date': "",
                            'source': 'Nijisanji'
                        })
                        if len(news_list) >= 8:
                            break
    except Exception as e:
        print(f"  [错误] 解析失败: {e}")
    return news_list

def fetch_source(source: Dict) -> Tuple[str, List[Dict]]:
    """抓取单个新闻源"""
    print(f"[INFO] 正在抓取: {source['name']} ({source['url']})")
    
    if source['type'] == 'html':
        html = fetch_html(source['url'])
        if not html:
            print(f"  [警告] 获取HTML失败")
            return source['name'], []
        
        if source['parser'] == 'hololive':
            news = parse_hololive_news(html)
        elif source['parser'] == 'nijisanji':
            news = parse_nijisanji_news(html)
        else:
            # 通用解析
            soup = BeautifulSoup(html, 'html.parser')
            news = []
            for link in soup.find_all('a', href=True):
                title = link.get_text(strip=True)
                if title and len(title) > 10:
                    news.append({
                        'title': title,
                        'link': link['href'],
                        'pub_date': "",
                        'source': source['name']
                    })
                    if len(news) >= 5:
                        break
    else:
        news = []
    
    print(f"  [OK] {source['name']}: 抓取到 {len(news)} 条新闻")
    return source['name'], news

def generate_html(all_news: Dict[str, List[Dict]], filtered_count: int) -> str:
    """生成HTML邮件内容"""
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
        <p>过滤后: {filtered_count} 条（原始总计 {sum(len(items) for items in all_news.values())} 条）</p>
    """
    
    total_news = 0
    for source_name, news_list in all_news.items():
        if news_list:
            html += f'<h2>📌 {source_name}</h2>'
            for news in news_list:
                total_news += 1
                link = news.get('link', '')
                title = news.get('title', '无标题')
                pub_date = news.get('pub_date', '')
                html += f'''
                <div class="news-item">
                    <div class="news-title">
                        <a href="{link}" target="_blank">{title}</a>
                    </div>
                    <div class="news-meta">
                        来源: {source_name}{f' | 日期: {pub_date}' if pub_date else ''}
                    </div>
                </div>
                '''
    
    if total_news == 0:
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
    total_raw = 0
    
    for source in NEWS_SOURCES:
        source_name, news_list = fetch_source(source)
        all_news[source_name] = news_list
        total_raw += len(news_list)
    
    print(f"\n总共抓取到 {total_raw} 条原始新闻")
    
    # 简单的过滤：去重
    filtered_count = total_raw
    
    if total_raw == 0:
        print("没有抓取到任何新闻数据")
        # 发送备选内容（可选）
        if SMTP_USER and TO_EMAIL:
            html_body = generate_html(all_news, filtered_count)
            send_email(html_body)
        else:
            print("邮件配置不完整，跳过发送")
    else:
        # 生成HTML并发送
        html_body = generate_html(all_news, filtered_count)
        
        if SMTP_USER and TO_EMAIL:
            send_email(html_body)
        else:
            print(f"成功抓取 {total_raw} 条新闻，但邮件配置不完整，未发送")
            # 保存到文件作为备份
            with open('vtuber_news_report.md', 'w', encoding='utf-8') as f:
                f.write(f"# VTuber新闻汇总\n\n")
                f.write(f"抓取时间: {datetime.now()}\n")
                f.write(f"新闻数量: {total_raw}\n\n")
                for source_name, news_list in all_news.items():
                    if news_list:
                        f.write(f"## {source_name}\n\n")
                        for news in news_list:
                            f.write(f"- [{news['title']}]({news['link']})\n")
                        f.write("\n")
            print("新闻已保存到 vtuber_news_report.md")

if __name__ == "__main__":
    main()