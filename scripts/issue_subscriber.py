#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Issue 评论自动处理脚本
- Issue #1（征集邮箱订阅）：提取邮箱 → 写入 subscribers.txt → 回复确认
- 其他 Issue：自动回复"正在发送到仓库所有者的邮箱..."

环境变量（由 GitHub Actions 注入）：
  ISSUE_NUMBER, COMMENT_ID, COMMENT_BODY, COMMENT_USER,
  ISSUE_TITLE, ISSUE_AUTHOR, REPO_OWNER, REPO_NAME, GITHUB_TOKEN
"""

import os
import re
import sys
import requests

# ==================== 配置 ====================

# 订阅 Issue 编号（征集邮箱的 Issue）
SUBSCRIPTION_ISSUE_NUMBER = 1

# 订阅者列表文件路径（仓库根目录，相对 scripts/ 目录）
SUBSCRIBERS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'subscribers.txt'
)

# GitHub API 基地址
API_BASE = 'https://api.github.com'

# 非 #1 Issue 的自动回复消息
AUTO_REPLY_MSG = """您好！感谢您的留言 🙌

您的反馈已收到，正在发送到仓库所有者的邮箱进行处理。

如果您也想接收 **VTuber 新闻邮件订阅**，请前往 [Issue #{sub}](../../issues/{sub}) 留下您的邮箱地址 ✉️""".format(sub=SUBSCRIPTION_ISSUE_NUMBER)

# ==================== 工具函数 ====================


def get_env(key: str, required: bool = True) -> str:
    val = os.getenv(key, '')
    if required and not val:
        print(f"[错误] 缺少环境变量: {key}")
        sys.exit(1)
    return val


def api_request(method: str, path: str, **kwargs) -> dict:
    token = get_env('GITHUB_TOKEN')
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'issue-subscriber-bot'
    }
    url = f'{API_BASE}/{path.lstrip("/")}'
    resp = requests.request(method, url, headers=headers, timeout=15, **kwargs)

    if resp.status_code >= 400:
        print(f'[API 错误] {method} {path} → HTTP {resp.status_code}')
        print(f'  响应: {resp.text[:300]}')
        return {}

    if resp.headers.get('content-type', '').startswith('application/json'):
        return resp.json()
    return {'status': resp.status_code, 'text': resp.text}


def post_comment(issue_number: int, body: str) -> bool:
    repo = f"{get_env('REPO_OWNER')}/{get_env('REPO_NAME')}"
    result = api_request('POST', f'/repos/{repo}/issues/{issue_number}/comments', json={'body': body})
    if result.get('id'):
        print(f"[回复] 已在 Issue #{issue_number} 发表评论")
        return True
    else:
        print(f"[回复失败] Issue #{issue_number}: {result}")
        return False


def extract_emails(text: str) -> list:
    """
    从文本中提取邮箱地址。
    支持格式：
      - 纯邮箱：xxx@xxx.com
      - 带前后缀：email: xxx@xxx.com / 邮箱=xxx@xxx.com / 我的邮箱是 xxx@xxx.com
    """
    if not text:
        return []

    # 匹配常见邮箱模式
    patterns = [
        r'(?:邮箱|email|e-mail|mail|邮件)\s*[:：=]\s*([\w\.\+\-]+\@[\w\-]+\.[\w\.]{2,})',
        r'([\w\.\+\-]+\@[\w\-]+\.(?:com|cn|net|org|edu|gov|io|co|dev|me|info|cc|top|xyz))',  # 常见域名后缀优先匹配
    ]

    emails = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            email = m.strip().lower().rstrip('.')
            # 基本合法性检查
            if '@' in email and '.' in email.split('@')[-1] and len(email) < 100:
                emails.add(email)
            elif '@' in email and len(email) > 5:
                # 不太确定但也可能有效
                pass  # 跳过可疑的

    # 如果上面的精确匹配没找到，用宽泛兜底
    if not emails:
        loose_matches = re.findall(r'[\w\.\+\-]+\@[\w\-]+\.[\w\.]{2,}', text, re.IGNORECASE)
        for m in loose_matches:
            email = m.strip().lower().rstrip('.')
            if len(email) > 6 and len(email) < 100 and not email.startswith(('http', '//')):
                emails.add(email)

    return list(emails)


def load_subscribers() -> list:
    """加载现有订阅者列表"""
    if os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        # 每行一个邮箱，跳过 # 开头的注释行
        return [line for line in lines if not line.startswith('#')]
    return []


def save_subscribers(emails: list):
    """保存订阅者列表（保留注释）"""
    existing_comments = []
    if os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):
                    existing_comments.append(line.rstrip())

    with open(SUBSCRIBERS_FILE, 'w', encoding='utf-8') as f:
        # 写入文件头注释
        f.write('# VTuber 新闻邮件订阅者列表\n')
        f.write('# 每行一个邮箱，由 issue_subscriber.yml 自动维护\n')
        f.write('# 请勿手动编辑此文件，使用 Issue #1 进行订阅/退订\n')
        f.write('#\n')

        for email in sorted(set(emails)):
            f.write(f'{email}\n')


# ==================== 主逻辑 ====================


def handle_subscription():
    """处理 Issue #1（邮箱征集）的评论"""
    comment_body = get_env('COMMENT_BODY')
    user_login = get_env('COMMENT_USER')

    print(f"[订阅] 用户 @{user_login} 在 Issue #{SUBSCRIPTION_ISSUE_NUMBER} 发表了评论")
    print(f"[订阅] 评论内容: {comment_body[:200]}")

    # 提取邮箱
    emails = extract_emails(comment_body)

    if not emails:
        print("[订阅] ⚠️ 未检测到有效邮箱地址")
        post_comment(SUBSCRIPTION_ISSUE_NUMBER, (
            f"@{user_login} 您好！😊\n\n"
            "抱歉，没有从您的评论中识别到有效的邮箱地址。\n\n"
            "**请在评论中包含您的邮箱**，例如：\n"
            "- `我的邮箱是 example@gmail.com`\n"
            "- `email: user@qq.com`\n\n"
            "收到后您将开始接收 VTuber 新闻每日汇总邮件 ✉️"
        ))
        return

    print(f"[订阅] 提取到邮箱: {emails}")

    # 加载现有订阅者
    existing = load_subscribers()

    # 去重并合并
    new_emails = [e for e in emails if e not in existing]
    all_emails = list(set(existing + emails))

    if not new_emails:
        print("[订阅] 这些邮箱已经在订阅列表中了")
        post_comment(SUBSCRIPTION_ISSUE_NUMBER, (
            f"@{user_login} 您好！😊\n\n"
            f"以下邮箱**已经订阅**过了：\n"
            + '\n'.join(f'- `{e}`' for e in emails) +
            "\n\n无需重复操作～您将继续收到 VTuber 新闻邮件 📬"
        ))
        return

    # 保存新列表
    save_subscribers(all_emails)

    # 构建确认回复
    email_list = '\n'.join(f'- `{e}`' for e in new_emails)
    total_count = len(all_emails)
    post_comment(SUBSCRIPTION_ISSUE_NUMBER, (
        f"@{user_login} 您好！✅ **订阅成功！** 🎉\n\n"
        f"已添加以下邮箱到订阅列表：\n{email_list}\n\n"
        f"当前共有 **{total_count}** 位订阅者\n\n"
        f"您将在下次定时任务运行后（北京时间 8:00 / 12:00 / 20:00）收到第一封邮件 ✉️\n\n"
        f"_如需退订，请在评论区回复「取消订阅」_"
    ))

    print(f"[订阅] ✅ 新增 {len(new_emails)} 个订阅者: {new_emails}")


def handle_other_issue():
    """处理非订阅 Issue 的评论——自动回复"""
    issue_number = int(get_env('ISSUE_NUMBER'))
    user_login = get_env('COMMENT_USER')
    comment_body = get_env('COMMENT_BODY')

    print(f"[自动回复] Issue #{issue_number} 收到来自 @{user_login} 的评论")

    # 如果是自己（bot）的回复，不处理（避免无限循环）
    bot_name = f"{get_env('REPO_OWNER')}[bot]"
    if user_login.endswith('[bot]') or 'github-actions' in user_login.lower():
        print(f"[自动回复] 跳过机器人自己的评论")
        return

    # 如果评论者是仓库所有者，不自动回复
    owner = get_env('ISSUE_AUTHOR')
    if user_login == owner:
        print(f"[自动回复] 跳过仓库作者 @{owner} 的评论")
        return

    # 检查是否包含"取消订阅"/"unsubscribe"关键词（跨 issue 也支持）
    unsub_keywords = ['取消订阅', 'unsubscribe', '退订', '不要发了', '停止发送']
    if any(kw in comment_body.lower() for kw in unsub_keywords):
        handle_unsubscribe(user_login, issue_number)
        return

    # 标准自动回复
    reply_body = AUTO_REPLY_MSG.replace('\n', '\n')
    post_comment(issue_number, reply_body)


def handle_unsubscribe(username: str, issue_number: int):
    """处理取消订阅请求"""
    subscribers = load_subscribers()
    comment_user = get_env('COMMENT_USER')

    # 尝试根据用户名找关联邮箱（通过历史记录或直接搜索）
    removed = []
    remaining = []

    # 简单方案：如果用户留了邮箱在评论中就删那个
    comment_body = get_env('COMMENT_BODY')
    emails_in_comment = extract_emails(comment_body)

    if emails_in_comment:
        for e in emails_in_comment:
            if e in subscribers:
                subscribers.remove(e)
                removed.append(e)
            else:
                remaining.append(e)
    else:
        # 没给具体邮箱，提示用户
        post_comment(issue_number, (
            f"@{comment_user} 您好 👋\n\n"
            "如需取消订阅，请在评论中**附上您之前订阅时使用的邮箱**，例如：\n"
            "`取消订阅 my@email.com`\n\n"
            "或者您也可以直接去 [Issue #{sub}](../../issues/{sub}) 处理。".format(sub=SUBSCRIPTION_ISSUE_NUMBER)
        ))
        return

    if removed:
        save_subscribers(subscribers)
        post_comment(issue_number, (
            f"@{comment_user} 已为您取消订阅 ✅\n\n"
            "以下邮箱已从列表中移除：\n"
            + '\n'.join(f'- ~~`{e}`~~' for e in removed) +
            "\n\n感谢您过去的订阅！如有需要随时可以重新订阅 🙌"
        ))
        print(f"[退订] 移除 {len(removed)} 个邮箱: {removed}")
    else:
        post_comment(issue_number, (
            f"@{comment_user} 您好 👋\n\n"
            "未找到与这些邮箱匹配的订阅记录：\n"
            + '\n'.join(f'- `{e}`' for e in emails_in_comment) +
            "\n\n可能您之前并未使用这些邮箱订阅？请核对后再试。"
        ))


def main():
    issue_number_str = os.getenv('ISSUE_NUMBER', '0')
    try:
        issue_number = int(issue_number_str)
    except ValueError:
        print(f"[错误] 无效的 Issue 编号: {issue_number_str}")
        sys.exit(1)

    comment_body = os.getenv('COMMENT_BODY', '')

    print("=" * 50)
    print("Issue Subscriber Handler")
    print(f"Issue #{issue_number} | 用户: @{{}}".format(os.getenv('COMMENT_USER', '?')))
    print("=" * 50)

    # 检查退订指令（全局生效，不限 issue）
    unsub_keywords = ['取消订阅', 'unsubscribe', '退订']
    if any(kw in comment_body.lower() for kw in unsub_keywords):
        if issue_number != SUBSCRIPTION_ISSUE_NUMBER:
            handle_unsubscribe(os.getenv('COMMENT_USER'), issue_number)
            return

    if issue_number == SUBSCRIPTION_ISSUE_NUMBER:
        handle_subscription()
    else:
        handle_other_issue()

    print("\n[完成] 处理结束")


if __name__ == '__main__':
    main()
