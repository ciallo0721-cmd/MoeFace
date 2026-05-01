# VTuber 新闻监控工作流配置指南

## 文件说明

```
.github/workflows/vtuber_news_fetcher.yml   ← GitHub Actions 调度配置
scripts/fetch_vtuber_news.py                ← 爬虫 + 邮件 + 建文件夹脚本
```

## 首次配置步骤（必读）

### Step 1：在 GitHub 仓库添加 Secrets

进入你的仓库 → **Settings → Secrets and variables → Actions**，
依次添加以下 4 个 Secret：

| Secret 名称 | 值 | 说明 |
|---|---|---|
| `SMTP_HOST` | `smtp.qq.com` | QQ邮箱 SMTP 服务器 |
| `SMTP_PORT` | `587` | TLS 端口 |
| `SMTP_USER` | 你的 QQ 邮箱地址 | 发件人 |
| `SMTP_PASS` | **QQ邮箱授权码** | 不是登录密码！ |
| `TO_EMAIL` | `ciallo0721cmd@gmail.com` | 收件人 |

> ⚠️ **QQ邮箱授权码获取方法：**
> 1. 打开 QQ邮箱网页版 → 右上角 **设置**
> 2. 点击 **账户** 标签
> 3. 找到 **POP3/IMAP/SMTP/Exchange/CardDAV/CalDAV服务**
> 4. 开启 **SMTP 服务**，按提示发送短信验证
> 5. 页面会生成一个 **16位授权码**，复制填入 `SMTP_PASS`

### Step 2：启用 Actions

进入 GitHub 仓库 → **Actions** 标签，GitHub 会自动检测到工作流文件。
点击 "VTuber News Fetcher" → 点击右侧 "Enable workflow"。

### Step 3：手动测试一次

点击 "VTuber News Fetcher" → 点击右侧 **"Run workflow"** → 运行。
查看 Run 日志确认：
- 爬虫是否成功抓取
- 邮件是否发出
- 是否创建了文件夹

## 监控的来源

| 来源 | 类型 | 说明 |
|---|---|---|
| Hololive Production | RSS | 日/英 |
| Nijisanji | RSS | 日/英 |
| Holostars | RSS | 男团 |
| Bilibili VTuber 聚合 | 网页 | 综合 |
| *(可自行扩展)* | | |

## 过滤规则

- **优先推送**：包含「出道」「新成员」「 Debut 」「演唱会」「Anniversary」等关键词
- **普通事件**：其余事件
- **自动排除**：含「广告」「推广」「抽奖」等噪音词

## 添加新的 VTuber 关键词

编辑 `scripts/fetch_vtuber_news.py` 中的 `KNOWN_VTUBERS` 列表，
可以添加任何你想监控的角色名/社团名。

## 自定义运行时间

修改 `.github/workflows/vtuber_news_fetcher.yml` 中的 cron：

```yaml
schedule:
  - cron: '0 0 * * *'          # 每天 UTC 0:00 = 北京时间 8:00
  # - cron: '0 9 * * *'       # 每天 UTC 9:00 = 北京时间 17:00
  # - cron: '0 */4 * * *'     # 每 4 小时一次
```

## 常见问题

**Q: 邮件发不出去？**
A: 确认 Gmail 开启了「低安全性应用访问」或使用了 App 密码，推荐后者。

**Q: GitHub Actions 报错 "Permission denied"?**
A: `GITHUB_TOKEN` 默认有读写仓库的权限，首次 push 需要仓库 Settings → Actions → 确认 "Read and write permissions"。

**Q: 想监控更多来源？**
A: 在 `VTUBER_SOURCES` 列表中按格式添加即可，参考文件中已有示例。
