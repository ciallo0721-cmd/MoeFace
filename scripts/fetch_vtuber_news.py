#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
import feedparser
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
from datetime import datetime
import time
import sys
import traceback

# ========== 配置 ==========
RSS_FEEDS = {
    "Hololive Production": "https://hololivepro.com/news/",
    "Nijisanji": "https://www.nijisanji.jp/news",
}

def fetch_rss_news(url, source_name):
    """
    抓取新闻，如果 URL 是 RSS 链接 (feed) 则直接解析，
    如果是普通网页 (HTML) 则尝试抓取列表。
    """
    print(f"[INFO] 正在抓取: {source_name} ({url})")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        resp = requests.get(url, timeout=15, headers=headers)
        resp.raise_for_status()
        resp.encoding = 'utf-8'

        # 如果是 RSS/XML 内容
        if 'xml' in resp.headers.get('Content-Type', '').lower() or url.endswith('.rss') or url.endswith('.xml'):
            feed = feedparser.parse(resp.text)
            if feed.bozo:  # 如果解析出错
                print(f"[WARN] RSS 解析失败 {source_name}: {feed.bozo_exception}")
                return []
            entries = []
            for entry in feed.entries[:10]:   # 限制条目数量，避免过多
                entries.append({
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'summary': entry.get('summary', ''),
                    'published': entry.get('published', '')
                })
            print(f"[OK] {source_name}: 抓取到 {len(entries)} 条")
            return entries
        else:
            # 处理普通的 HTML 新闻页面
            soup = BeautifulSoup(resp.text, 'html.parser')
            articles = []
            # 针对 hololivepro.com/news/ 的结构解析
            if "hololivepro.com" in url and "/news/" in url:
                news_items = soup.find_all('a', class_='p-article__link')  # 根据实际页面结构调整选择器
                for item in news_items[:10]:
                    title = item.get_text(strip=True)
                    link = item.get('href')
                    if link and not link.startswith('http'):
                        link = "https://hololivepro.com" + link
                    if title and link:
                        articles.append({
                            'title': title,
                            'link': link,
                            'summary': '',
                            'published': ''
                        })
            # 针对 nijisanji.jp/news 的结构解析
            elif "nijisanji.jp" in url and "/news" in url:
                news_items = soup.find_all('div', class_='news-list-item')  # 假设的选择器
                for item in news_items[:10]:
                    title_elem = item.find('h3', class_='news-item-title')
                    link_elem = item.find('a')
                    if title_elem and link_elem:
                        title = title_elem.get_text(strip=True)
                        link = link_elem.get('href')
                        if link and not link.startswith('http'):
                            link = "https://www.nijisanji.jp" + link
                        articles.append({
                            'title': title,
                            'link': link,
                            'summary': '',
                            'published': ''
                        })
            else:
                print(f"[WARN] 未知的 HTML 页面结构: {source_name}")
            
            print(f"[OK] {source_name}: 抓取到 {len(articles)} 条 HTML 新闻")
            return articles
    except Exception as e:
        print(f"[WARN] 抓取失败 {source_name}: {str(e)}")
        return []

def filter_news(news_list, keywords):
    """
    根据关键词过滤新闻
    news_list: 新闻列表
    keywords: 关键词列表
    """
    filtered = []
    for news in news_list:
        title = news.get('title', '').lower()
        summary = news.get('summary', '').lower()
        # 检查标题或摘要是否包含任一关键词
        if any(kw.lower() in title or kw.lower() in summary for kw in keywords):
            filtered.append(news)
    return filtered

def generate_html_email_content(news_list):
    """
    生成 HTML 格式的邮件内容，遵循要求的格式
    """
    today_str = datetime.now().strftime('%Y-%m-%d')
    # 固定的 “发现”入口 URL，可以替换成你自己的
    detail_url = "https://github.com/ciallo0721-cmd/MoeFace"
    
    # 邮件头部说明
    html_content = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            h2 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
            .news-item {{ margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #f0f0f0; }}
            .news-title {{ font-size: 18px; font-weight: bold; margin-bottom: 5px; }}
            .news-title a {{ color: #2980b9; text-decoration: none; }}
            .news-title a:hover {{ text-decoration: underline; }}
            .news-summary {{ color: #666; font-size: 14px; }}
            .footer {{ margin-top: 30px; padding-top: 10px; border-top: 1px solid #ddd; font-size: 12px; color: #999; text-align: center; }}
        </style>
    </head>
    <body>
    <h2>—— {today_str} V圈事件 ——</h2>
    """
    
    if not news_list:
        # 无内容时的显示
        html_content += """
        <div class="news-item">
            <div class="news-title">📭 今天没有筛选到符合条件的VTuber新闻</div>
            <div class="news-summary">可能是新闻源暂无更新或关键词过滤条件较严格。</div>
        </div>
        """
    else:
        # 遍历新闻列表，按照编号格式生成
        for idx, news in enumerate(news_list, 1):
            title = news.get('title', '无标题')
            link = news.get('link', '#')
            summary = news.get('summary', '暂无详细摘要，请点击标题查看原文。')
            # 确保摘要长度适中并清洗 HTML 标签
            summary_text = BeautifulSoup(summary, 'html.parser').get_text().strip()
            if len(summary_text) > 300:
                summary_text = summary_text[:300] + '...'
            
            html_content += f"""
        <div class="news-item">
            <div class="news-title">{idx}. <a href='{link}' target='_blank' rel='noopener noreferrer'>{title}</a></div>
            <div class="news-summary">{summary_text}</div>
        </div>
        """
    
    html_content += f"""
        <div class="info-note">
            📧 此邮件由 GitHub Actions 自动生成，如需取消订阅请忽略本邮件。<br>
            🔍 更多内容请点击 <a href='{detail_url}' target='_blank' rel='noopener noreferrer'>🔗 查看详情</a>
        </div>
        <div class="footer">
            - GitHub 工作流<br>
            - Moe Face
        </div>
    </body>
    </html>
    """
    return html_content

def send_email(html_content):
    """
    发送邮件，修复 From 头部的编码问题
    """
    smtp_host = os.environ.get('SMTP_HOST')
    smtp_port = os.environ.get('SMTP_PORT')
    smtp_user = os.environ.get('SMTP_USER')
    smtp_pass = os.environ.get('SMTP_PASS')
    to_email = os.environ.get('TO_EMAIL')
    
    if not all([smtp_host, smtp_port, smtp_user, smtp_pass, to_email]):
        print("[ERROR] 邮件配置不完整，请检查环境变量 (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, TO_EMAIL)")
        sys.exit(1)
    
    # 关键修复：构造符合 QQ 邮箱要求的 From 格式 (昵称 <邮箱地址>)
    # 不要对 From 头部使用 utf-8 编码，否则 QQ 邮箱会识别失败
    from_display_name = "MoeFace"
    # QQ 邮箱要求格式: 昵称 <邮箱地址>
    from_addr = f"{from_display_name} <{smtp_user}>"
    
    msg = MIMEText(html_content, 'html', 'utf-8')
    # 关键：From 使用 formataddr 构造标准格式，不要对 From 做额外的 Header 编码，否则 QQ 会认为格式错误
    # 同时也确保 From 头符合 RFC5322 标准
    msg['From'] = from_addr
    msg['To'] = to_email
    # 对于 Subject 可以保留编码，它不影响 From 的识别
    subject_str = f'VTuber 新闻汇总 - {datetime.now().strftime("%Y-%m-%d")}'
    msg['Subject'] = Header(subject_str, 'utf-8')
    msg['Date'] = smtplib.formatdate()
    
    try:
        # 使用 SSL 连接 QQ 邮箱的 465 端口
        print(f"[INFO] 正在连接 SMTP 服务器 {smtp_host}:{smtp_port}")
        server = smtplib.SMTP_SSL(smtp_host, int(smtp_port))
        server.ehlo()
        server.login(smtp_user, smtp_pass)
        print("[INFO] SMTP 登录成功，正在发送邮件...")
        server.sendmail(smtp_user, [to_email], msg.as_string())
        server.quit()
        print(f"[SUCCESS] 邮件发送成功！收件人: {to_email}")
    except smtplib.SMTPAuthenticationError:
        print("[ERROR] SMTP 认证失败，请检查 SMTP_USER 或 SMTP_PASS (QQ邮箱需要授权码)")
        sys.exit(1)
    except smtplib.SMTPDataError as e:
        print(f"[ERROR] 邮件发送时数据错误: {e}")
        sys.exit(1)
    except smtplib.SMTPException as e:
        print(f"[ERROR] 发送邮件失败: {e}")
        sys.exit(1)

def main():
    print("==================================================")
    print("VTuber News Fetcher 启动")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==================================================")
    
    # 设置新闻关键词（中英文都可以，会自动转为小写匹配）
    keywords = ["hololive", "holostars", "nijisanji", "虚拟主播", "vtuber", "live", "concert", "演唱会", "新衣服", "毕业", "活动", "联动", "周边", "album", "单曲", "游戏", "直播计划", "节目"]
    
    # 抓取新闻
    all_news = []
    for source_name, url in RSS_FEEDS.items():
        news_list = fetch_rss_news(url, source_name)
        all_news.extend(news_list)
        time.sleep(0.5)   # 避免请求过快
    
    print(f"\n总共抓取到 {len(all_news)} 条原始新闻")
    
    # 过滤新闻
    if all_news:
        filtered_news = filter_news(all_news, keywords)
        print(f"过滤后保留 {len(filtered_news)} 条新闻")
    else:
        filtered_news = []
        print("没有抓取到任何新闻数据")
    
    # 生成 HTML 邮件内容
    html_body = generate_html_email_content(filtered_news)
    
    # 发送邮件
    send_email(html_body)

if __name__ == "__main__":
    main()