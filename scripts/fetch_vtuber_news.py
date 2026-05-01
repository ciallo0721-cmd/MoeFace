#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTuber 新闻抓取脚本
抓取 Hololive / Nijisanji / Vtuber Daily 等热门 V 圈事件
发送邮件 + 创建角色文件夹
"""

import os
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from datetime import datetime
from pathlib import Path

import feedparser
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ========== 配置区 ==========
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
TO_EMAIL = os.getenv("TO_EMAIL", "ciallo0721cmd@gmail.com")
REPO_OWNER = "ciallo0721-cmd"
REPO_NAME = "MoeFace"
DATA_DIR = Path("data")
# ============================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

# 已知的 Vtuber 团体/公司 → 官网公告 RSS / 页面
VTUBER_SOURCES = [
    {
        "name": "Hololive Production",
        "type": "rss",
        "url": "https://hololive.so-hololive.com/news.rss",
    },
    {
        "name": "Hololive English",
        "type": "rss",
        "url": "https://hololive-en.so-hololive.com/news.rss",
    },
    {
        "name": "Nijisanji",
        "type": "rss",
        "url": "https://www.nijisanji.com/rss.xml",
    },
    {
        "name": "Nijisanji EN",
        "type": "rss",
        "url": "https://www.nijisanji-en.com/rss.xml",
    },
    {
        "name": "Holostars",
        "type": "rss",
        "url": "https://holostars.so-hololive.com/news.rss",
    },
    {
        "name": "Bilibili VTuber 动态聚合",
        "type": "web",
        "url": "https://vtuber.techvsa.net/",
    },
]

# 过滤关键词（只保留包含这些词的条目）
INCLUDE_KEYWORDS = [
    "出道", "新成员", "新Vtuber", " Debut", "新 Liver",
    "预告", "活动", "演唱会", "live", "Anniversary",
    "合作", "联动", "collaboration",
]

# 排除词（噪音过滤）
EXCLUDE_KEYWORDS = [
    "抽奖", "广告", "推广", "【广告】",
]


def fetch_rss(url: str, source_name: str) -> list:
    """抓取 RSS 源"""
    news = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        for entry in feed.entries[:10]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            published = entry.get("published", datetime.now().isoformat())
            summary = ""
            if hasattr(entry, "summary"):
                summary = BeautifulSoup(entry.summary, "html.parser").get_text()[:200]
            news.append({
                "source": source_name,
                "title": title,
                "link": link,
                "published": published,
                "summary": summary,
            })
    except Exception as e:
        print(f"[WARN] RSS 抓取失败 {source_name}: {e}")
    return news


def fetch_web(url: str, source_name: str) -> list:
    """抓取网页（用于非 RSS 源）"""
    news = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # 简单策略：找所有 <a> 链接标题
        items = soup.select("a.post-title, a.item-title, h3 a, .entry-title a")
        for item in items[:15]:
            title = item.get_text(strip=True)
            link = item.get("href", "")
            if title and len(title) > 5:
                news.append({
                    "source": source_name,
                    "title": title,
                    "link": link,
                    "published": datetime.now().isoformat(),
                    "summary": "",
                })
    except Exception as e:
        print(f"[WARN] 网页抓取失败 {source_name}: {e}")
    return news


def filter_news(all_news: list) -> list:
    """过滤噪音，只保留有价值的事件"""
    filtered = []
    for item in all_news:
        title = item["title"]
        # 排除
        if any(kw in title for kw in EXCLUDE_KEYWORDS):
            continue
        # 优先保留包含关键词的
        if any(kw.lower() in title.lower() for kw in INCLUDE_KEYWORDS):
            item["priority"] = "HIGH"
            filtered.append(item)
        elif len(title) > 10:
            item["priority"] = "NORMAL"
            filtered.append(item)
    return filtered


def extract_vtuber_names(news: list) -> list:
    """从新闻标题/摘要中提取 VTuber 名字（简单关键词匹配）"""
    # 常见 VTuber 关键词（可扩展）
    KNOWN_VTUBERS = [
        "Hololive", "Nijisanji", "hololive", "nijisanji",
        "Gawr Gura", "Mori Calliope", "Watson Amelia",
        "Hoshimachi Suisei", "Usada Pekora", "Shirakami Fubuki",
        " Houshou Marine", "Miko", "Sakura Miko",
        "Vox Akuma", " Ike Eveland", "Luca Kaneshiro",
        "Shu Yamino", "Nina Kosaka", "Elena Lind",
        "Kson", "Mio", "Fubuki", "Aqua", "Rushia",
        "牡丹", "塔菲", "东雪莲", "A-Soul",
    ]
    found = set()
    for item in news:
        text = item["title"] + " " + item.get("summary", "")
        for vtuber in KNOWN_VTUBERS:
            if vtuber in text and vtuber not in ("hololive", "nijisanji", "Hololive", "Nijisanji"):
                found.add(vtuber)
    return list(found)


def create_vtuber_folders(names: list) -> list:
    """通过 GitHub API 创建角色文件夹"""
    created = []
    for name in names:
        # 规范化文件夹名（去掉空格里的特殊字符）
        safe_name = name.strip().replace("/", "_").replace("\\", "_")
        folder = DATA_DIR / safe_name
        try:
            folder.mkdir(parents=True, exist_ok=True)
            # 放一个 .gitkeep 防止空目录被 git 忽略
            gitkeep = folder / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.write_text("")
            created.append(safe_name)
            print(f"[OK] 创建文件夹: data/{safe_name}/")
        except Exception as e:
            print(f"[WARN] 创建失败 {safe_name}: {e}")
    return created


def build_email_body(news: list, created_folders: list) -> str:
    """生成邮件正文 HTML"""
    date_str = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    html = f"""
    <html>
    <body style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 700px; margin: auto; padding: 20px; background: #f9f9f9;">
      <div style="background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        <h2 style="color: #e74c3c; border-bottom: 3px solid #e74c3c; padding-bottom: 8px;">
          🔥 VTuber / V圈热门事件日报
        </h2>
        <p style="color: #888; font-size: 14px;">📅 {date_str} · 自动抓取</p>

        <h3 style="color: #2c3e50;">📣 今日事件 ({len(news)} 条)</h3>
        {"<hr>".join(
            f'''
            <div style="margin: 12px 0; padding: 12px; border-left: 4px solid {'#e74c3c' if n.get('priority')=='HIGH' else '#3498db'}; background: #f8f9fa; border-radius: 4px;">
              <span style="background: {'#e74c3c' if n.get('priority')=='HIGH' else '#3498db'}; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 12px;">
                {n["source"]}
              </span>
              <span style="background: #f39c12; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-left: 6px;">
                {"⭐优先" if n.get("priority")=="HIGH" else "普通"}
              </span>
              <h4 style="margin: 8px 0 4px;"><a href="{n["link"]}">{n["title"]}</a></h4>
              {f'<p style="color:#666; font-size:13px; margin:4px 0">{n["summary"]}</p>' if n.get("summary") else ""}
            </div>
            ''' for n in news
        ) if news else "<p style='color:#888'>今日暂无重大事件 ☁️</p>"}

        <h3 style="color: #2c3e50;">📁 已创建角色文件夹</h3>
        {"<ul>" + "".join(f"<li><code>{f}</code></li>" for f in created_folders) + "</ul>" if created_folders else "<p style='color:#888'>本次未发现新角色 🎵</p>"}

        <hr style="margin: 20px 0;">
        <p style="color: #aaa; font-size: 12px;">
          本邮件由 GitHub Actions 自动发送 · <a href="https://github.com/{REPO_OWNER}/{REPO_NAME}">{REPO_NAME}</a><br>
          如不想继续接收，请前往 GitHub 仓库取消 schedule workflow。
        </p>
      </div>
    </body>
    </html>
    """
    return html


def send_email(html_body: str, subject_extra: str = "") -> bool:
    """发送邮件"""
    if not SMTP_USER or not SMTP_PASS:
        print("[WARN] SMTP 未配置，跳过邮件发送")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = Header(f"🔥 V圈日报 {datetime.now().strftime('%m/%d')} {subject_extra}", "utf-8")
        msg["From"] = SMTP_USER
        msg["To"] = TO_EMAIL
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, TO_EMAIL, msg.as_string())
        print(f"[OK] 邮件已发送至 {TO_EMAIL}")
        return True
    except Exception as e:
        print(f"[ERROR] 邮件发送失败: {e}")
        return False


def save_report(news: list, created_folders: list):
    """保存新闻报告到 Markdown 文件（用于 artifact）"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# VTuber / V圈热门事件日报 — {date_str}",
        "",
        f"共抓取 **{len(news)}** 条事件",
        "",
        "## 📣 事件列表",
        "",
    ]
    for i, n in enumerate(news, 1):
        priority_tag = "⭐" if n.get("priority") == "HIGH" else "📌"
        lines.append(f"{i}. {priority_tag} **[{n['source']}]** {n['title']}")
        if n.get("summary"):
            lines.append(f"   > {n['summary']}")
        lines.append(f"   🔗 {n['link']}")
        lines.append("")

    if created_folders:
        lines += ["## 📁 已创建角色文件夹", ""]
        for f in created_folders:
            lines.append(f"- `data/{f}/`")
        lines.append("")

    report_path = "vtuber_news_report.md"
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] 报告已保存: {report_path}")


def main():
    print("=" * 50)
    print("VTuber News Fetcher 启动")
    print("=" * 50)

    # 1. 抓取所有来源
    all_news = []
    for src in VTUBER_SOURCES:
        if src["type"] == "rss":
            items = fetch_rss(src["url"], src["name"])
        else:
            items = fetch_web(src["url"], src["name"])
        all_news.extend(items)
        print(f"[OK] {src['name']}: {len(items)} 条")

    # 2. 过滤
    filtered = filter_news(all_news)
    print(f"\n过滤后: {len(filtered)} 条（原始 {len(all_news)} 条）")

    # 3. 提取 VTuber 名字 → 建文件夹
    vtuber_names = extract_vtuber_names(filtered)
    created = create_vtuber_folders(vtuber_names)

    # 4. 发邮件
    html_body = build_email_body(filtered, created)
    send_email(html_body)

    # 5. 保存报告
    save_report(filtered, created)

    print("\n全部完成！")


if __name__ == "__main__":
    main()
