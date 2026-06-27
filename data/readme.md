
# MoeFace 人脸识别数据集

MoeFace 是一个用于人脸特征学习与风格化研究的数据集，主要面向人脸识别、特征提取以及生成模型训练等任务。

每个文件夹代表一个独立角色，内部包含该角色的面部图像样本。


---

## ✨ 数据集特性

- 📁 **结构清晰**：按角色分类整理
- ⚡ **可直接使用**：兼容 PyTorch / TensorFlow ImageFolder
- 🔄 **持续维护**：定期更新与修正标注
- 🎨 **多用途**：可用于人脸识别、特征学习与风格迁移研究

---

## ⚖️ 许可说明

本数据集整体采用 **CC0 1.0 Universal（Public Domain Dedication）** 发布：


在法律允许的最大范围内，数据集提供者放弃所有版权与相关权利。

---

## ⚠️ 数据来源声明

- 数据来源于公开可访问的网络内容
- 通过关键词检索与筛选整理而成
- 已尽力过滤明显的隐私或非公开内容
- 数据仅用于研究、学习与非恶意用途

> 若权利方认为某些内容存在侵权问题，可通过 Issue 联系删除。

---

## 🚀 使用方式

### 📌 PyTorch 示例

```python
from torchvision.datasets import ImageFolder

dataset = ImageFolder("data/")
print(dataset.classes)  # 输出角色列表
````

---

## 🧠 使用建议

* 建议在训练前进行：

  * 人脸检测裁剪
  * 去重处理
  * 分辨率统一
* 可用于：

  * 人脸识别模型训练
  * embedding 学习
  * 风格迁移 / LoRA 数据准备

---

## 🤝 贡献方式

欢迎提交：

* 新角色数据
* 错误标注修正
* 去重或清洗优化
* 数据结构改进建议

---

## 📜 更新日志

* 持续更新中（自动同步 + 人工审核）
* 提交记录：[https://github.com/ciallo0721-cmd/MoeFace/commits/main](https://github.com/ciallo0721-cmd/MoeFace/commits/main)

---

## ⭐ 支持

如果这个数据集对你有帮助，可以点个 Star ⭐
