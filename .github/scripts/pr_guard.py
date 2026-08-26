#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PR 内容审查脚本（配合 .github/workflows/pr-guard.yml 使用）

逻辑：
  1. 从 stdin 读取 `gh pr diff` 的输出
  2. 只提取「新增行」（以单个 + 开头，不含 +++ 文件头）
  3. 对新增行做敏感词匹配：
       - 普通违禁词：    .github/blocklist_banned.txt
       - 政治类敏感词：  .github/blocklist_political.txt
  4. 命中则写报告文件 + 设置 outputs.violations=true（由 workflow 决定是否留言/关 PR/标红）

词表由使用者自行维护，本脚本不内置任何具体词。
两个 txt 每行一个词；以 # 开头的行为注释，自动跳过。
"""
import os
import re
import sys


def load_terms(path):
    """读取词表文件，返回去空白、去注释后的词列表。文件不存在则返回空列表。"""
    terms = []
    if not os.path.exists(path):
        return terms
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            terms.append(s)
    return terms


def make_pattern(term):
    """
    构造匹配正则：
      - 纯 ASCII 词：加 \\b 词边界，避免 'av' 误伤 'avatar'
      - 含非 ASCII（如中文）：直接子串匹配（中文无词边界概念）
    """
    if re.fullmatch(r"[\x00-\x7F]+", term):
        return re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    return re.compile(re.escape(term), re.IGNORECASE)


def parse_diff(lines):
    """解析 diff 文本，yield (文件路径, 新增行内容)。"""
    cur = None
    out = []
    for raw in lines:
        if raw.startswith("diff --git"):
            m = re.search(r"\s b/(.+)$", raw)
            if m:
                cur = m.group(1)
            continue
        if raw.startswith("+++"):
            m = re.search(r"\+{3} b/(.+)$", raw)
            if m:
                cur = m.group(1)
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            out.append((cur, raw[1:]))
    return out


def main():
    banned_file = os.environ.get("BANNED_FILE", ".github/blocklist_banned.txt")
    political_file = os.environ.get("POLITICAL_FILE", ".github/blocklist_political.txt")
    auto_close = os.environ.get("AUTO_CLOSE", "false").lower() == "true"

    banned = load_terms(banned_file)
    political = load_terms(political_file)
    banned_patterns = [(t, make_pattern(t)) for t in banned]
    political_patterns = [(t, make_pattern(t)) for t in political]

    diff_text = sys.stdin.read().splitlines()
    added = parse_diff(diff_text)

    matches = []  # (filepath, term, category)
    for filepath, line in added:
        for term, pat in banned_patterns:
            if pat.search(line):
                matches.append((filepath, term, "违禁词"))
                break
        for term, pat in political_patterns:
            if pat.search(line):
                matches.append((filepath, term, "政治类敏感词"))
                break

    # 去重
    seen = set()
    uniq = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            uniq.append(m)

    gh_output = os.environ.get("GITHUB_OUTPUT", "")
    if uniq:
        lines_out = [
            "## ⚠️ PR 内容审查未通过",
            "",
            "本 PR 新增内容命中以下敏感词，已**拒绝合并**：",
            "",
            "| 文件 | 命中词 | 类别 |",
            "| --- | --- | --- |",
        ]
        for fp, term, cat in uniq:
            lines_out.append(f"| `{fp or '未知'}` | `{term}` | {cat} |")
        lines_out.append("")
        lines_out.append(
            "请移除相关敏感词后重新提交。词表维护在 "
            "`.github/blocklist_banned.txt` 与 `.github/blocklist_political.txt`。"
        )
        report = "\n".join(lines_out)
        with open("pr_guard_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        if gh_output:
            with open(gh_output, "a", encoding="utf-8") as f:
                f.write("violations=true\n")
                f.write(f"auto_close={'true' if auto_close else 'false'}\n")

        print(f"[PR Guard] 命中 {len(uniq)} 条敏感词，拒绝合并。")
        for fp, term, cat in uniq:
            print(f"  - [{cat}] {term}  @ {fp}")
    else:
        if gh_output:
            with open(gh_output, "a", encoding="utf-8") as f:
                f.write("violations=false\n")
                f.write("auto_close=false\n")
        print("[PR Guard] 未命中敏感词，通过。")


if __name__ == "__main__":
    main()
