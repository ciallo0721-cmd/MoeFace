# MEMORY.md — MoeFace 项目长期记忆

## 项目概况

- **项目名**: MoeFace（EmoScan Pro / MoeFace）
- **路径**: `G:/EmoScan Pro/MoeFace`
- **功能**: 动漫人脸识别系统，使用 FaceNet + lbpcascade_animeface，通过 JSON 特征库匹配角色

## 核心文件

- `recognize.py` — 主程序（2026-04-04 已改造为 Tkinter GUI），模型懒加载，支持拖拽识别图片/视频
- `cname/name.json` — 角色别名配置（模块化，JSON 格式），取代原 KEYWORD_MAPPING 硬编码
- `data/` — 各角色训练图片（每个角色一个子文件夹）
- `features/` — JSON 格式特征库缓存

## 别名系统（cname/name.json）规则

- 格式：`[{"db_name": "角色名", "aliases": ["别名1", "别名2", ...]}]`
- 用于根据视频/图片文件名中的关键词，自动选择加载哪个角色特征库
- 匹配逻辑：文件名（小写）包含任意 alias（小写）即匹配，返回 `db_name`
- `DEFAULT_DB_NAME = "全部特征库"` 表示加载所有角色
- 可在 GUI 内通过「管理角色别名」按钮直接编辑

## data/ 目录新增角色文件夹（2026-04-01，更新版）

以下为空文件夹（待采集训练图片），全部为全球热门角色：

**动漫角色**（MyWaifuList 全球热门榜）：Megumin、Rem、Asuna、Zero_Two、Emilia、Kurisu_Makise、Nezuko、Saber、初音未来（已有）

**Hololive VTuber**（订阅数百万级）：Gawr_Gura、Houshou_Marine、Hoshimachi_Suisei、Usada_Pekora、Shirakami_Fubuki、Inugami_Korone、Minato_Aqua、Nekomata_Okayu、Shirogane_Noel、Oozora_Subaru、Sakura_Miko、Amane_Kanata、Nakiri_Ayame、Tsunomaki_Watame、Takanashi_Kiara、Mori_Calliope、Watson_Amelia、Aki_Marine

⚠️ 注意：之前错误创建了 Bocchi、Kaguya_Luna、Subaru_Hololive、Yuki_Noa 等冷门/不规范命名的文件夹，已全部删除。数据投毒不可取！

## 已配置角色及别名（截至 2026-03-29）

| db_name | 关键词示例 | 来源 |
|---|---|---|
| 永雏塔菲 | 塔菲、雏草姬、Taffy | VTuber |
| 东雪莲 | 东雪莲、罕见、Azuma Lim | VTuber |
| 丛雨 | 丛雨、Murasame | 千恋*万花 |
| 棍母 | 棍母、Konbu、棍棍 | VTuber |
| Ayachi_Nene | Ayachi、Nene、绫地宁宁、宁宁、绫地 | VTuber |
| Neuro-sama | Neuro、牛肉、Neurosama | VTuber |
| otto | Otto | VTuber |
| ShikiNatsume | 夏目、四季夏目、枣子姐、Shiki | VTuber |
| Monika | Monika、莫妮卡、Just Monika | DDLC |
| Natsuki | Natsuki、夏树 | DDLC |
| Sayori | Sayori、纱世里、小夜 | DDLC |
| 三司绫濑 | 三司绫濑、三司、锉刀、Ayase、Misumi | RIDDLE JOKER |
| 初音未来 | 初音未来、初音、Miku、Hatsune、葱娘、米库 | VOCALOID |
| 千早爱音 | 千早爱音、爱音、Anon、Chihaya | MyGO!!!!! |
| 喜多郁代 | 喜多郁代、喜多、Ikuyo、Kita、归去来兮 | 孤独摇滚 |
| 安和昴 | 安和昴、昴、Subaru、Yasunaga | Girls Band Cry |
| 常陆茉子 | 常陆茉子、茉子、Mako、Hitachi | 千恋*万花 |
| 明月栞那 | 明月栞那、栞那、Kanna、Akizuki | 星光咖啡馆与死神之蝶 |
| 朝武芳乃 | 朝武芳乃、芳乃、Yoshino、Tomotake、Ciallo | 千恋*万花 |
| 河原木桃香 | 河原木桃香、桃香、Momoka、Kawaraoki | Girls Band Cry |
| 海老冢智 | 海老冢智、海老塚智、Tomo、Ebizuka | Girls Band Cry |
| 神乐Mea | 神乐Mea、Mea、咩啊、屑女仆 | VTuber |
| 绪山真寻 | 绪山真寻、真寻、Mahiro、欧尼酱 | 别当欧尼酱了！ |
| 蕾娜·列支敦瑙尔 | 蕾娜、列支敦瑙尔、Rena、Lichtennauer | 千恋*万花 |
| 高松灯 | 高松灯、灯、Tomori、Tomorin | MyGO!!!!! |
| 井芹仁菜 | 井芹仁菜、仁菜、Nina、Iseri | Girls Band Cry |
