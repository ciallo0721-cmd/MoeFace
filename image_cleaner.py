#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片分辨率分类清理工具 — MoeFace 动漫人脸识别项目配套工具
按分辨率分组显示动漫/二次元图片，每次显示2张，可选择保留或删除
支持两种扫描模式：单文件夹模式（不遍历子文件夹）或递归模式（遍历所有子文件夹）
用于整理 VTuber / 动漫角色图片数据集
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
from tkinter import Tk, ttk, filedialog, messagebox, StringVar, IntVar, BooleanVar
from tkinter import Label, Button, Frame, Canvas, Scrollbar, Checkbutton
from PIL import Image, ImageTk
import threading


class ImageCleanerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片分辨率分类清理工具")
        self.root.geometry("1200x800")

        self.data_dir = Path("data")  # 使用实例变量，方便修改
        self.recursive_mode = BooleanVar(value=False)  # 递归遍历模式，默认关闭
        self.resolution_groups = defaultdict(list)
        self.sorted_resolutions = []
        self.current_res_index = 0
        self.current_img_index = 0
        self.deleted_files = []
        self.thumbnail_cache = {}

        self.setup_ui()
        self.scan_images()

    def setup_ui(self):
        """设置UI界面"""
        # 顶部信息栏
        top_frame = Frame(self.root, bg="#f0f0f0", height=60)
        top_frame.pack(fill="x", padx=10, pady=5)
        top_frame.pack_propagate(False)

        self.info_var = StringVar(value="正在扫描图片...")
        Label(top_frame, textvariable=self.info_var, font=("微软雅黑", 12), bg="#f0f0f0").pack(side="left", padx=10)

        # 分辨率导航
        nav_frame = Frame(top_frame, bg="#f0f0f0")
        nav_frame.pack(side="right", padx=10)

        Button(nav_frame, text="◀ 上一组", command=self.prev_group, font=("微软雅黑", 10), width=10).pack(side="left", padx=5)
        self.res_var = StringVar(value="")
        Label(nav_frame, textvariable=self.res_var, font=("微软雅黑", 11, "bold"), bg="#f0f0f0", width=20).pack(side="left", padx=5)
        Button(nav_frame, text="下一组 ▶", command=self.next_group, font=("微软雅黑", 10), width=10).pack(side="left", padx=5)

        # 图片显示区域
        img_frame = Frame(self.root, bg="#2b2b2b")
        img_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # 左侧图片
        left_frame = Frame(img_frame, bg="#1a1a1a")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.left_img_label = Label(left_frame, bg="#1a1a1a", text="", font=("微软雅黑", 14))
        self.left_img_label.pack(fill="both", expand=True)
        self.left_info_var = StringVar(value="")
        Label(left_frame, textvariable=self.left_info_var, font=("微软雅黑", 9), bg="#1a1a1a", fg="white").pack(pady=5)

        # 右侧图片
        right_frame = Frame(img_frame, bg="#1a1a1a")
        right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        self.right_img_label = Label(right_frame, bg="#1a1a1a", text="", font=("微软雅黑", 14))
        self.right_img_label.pack(fill="both", expand=True)
        self.right_info_var = StringVar(value="")
        Label(right_frame, textvariable=self.right_info_var, font=("微软雅黑", 9), bg="#1a1a1a", fg="white").pack(pady=5)

        # 底部操作栏
        bottom_frame = Frame(self.root, bg="#f0f0f0", height=100)
        bottom_frame.pack(fill="x", padx=10, pady=5)
        bottom_frame.pack_propagate(False)

        # 模式选择区域
        mode_frame = Frame(bottom_frame, bg="#f0f0f0")
        mode_frame.pack(side="top", pady=5)

        Checkbutton(mode_frame, text="递归遍历子文件夹（扫描所有子目录）", 
                   variable=self.recursive_mode, 
                   command=self.toggle_mode,
                   font=("微软雅黑", 10), bg="#f0f0f0").pack(side="left", padx=10)

        # 进度标签
        self.progress_var = StringVar(value="")
        Label(bottom_frame, textvariable=self.progress_var, font=("微软雅黑", 10), bg="#f0f0f0").pack(side="top", pady=2)

        # 按钮区域
        btn_frame = Frame(bottom_frame, bg="#f0f0f0")
        btn_frame.pack(side="bottom", pady=5)

        Button(btn_frame, text="✅ 保留这两张", command=self.keep_images, font=("微软雅黑", 12, "bold"),
               bg="#4CAF50", fg="white", width=18, height=2, cursor="hand2").pack(side="left", padx=20)

        Button(btn_frame, text="🗑️ 删除这两张", command=self.delete_images, font=("微软雅黑", 12, "bold"),
               bg="#f44336", fg="white", width=18, height=2, cursor="hand2").pack(side="left", padx=20)

        Button(btn_frame, text="📁 选择数据目录", command=self.choose_directory, font=("微软雅黑", 10),
               width=15).pack(side="left", padx=20)

        # 显示路径信息
        self.path_var = StringVar(value=f"当前目录: {self.data_dir}")
        Label(bottom_frame, textvariable=self.path_var, font=("微软雅黑", 8), bg="#f0f0f0", fg="gray").pack(side="bottom", pady=2)

    def toggle_mode(self):
        """切换扫描模式时重新扫描"""
        # 显示提示
        mode_name = "递归遍历所有子文件夹" if self.recursive_mode.get() else "单文件夹模式（不遍历子文件夹）"
        if messagebox.askyesno("切换扫描模式", f"切换到 {mode_name} 吗？\n这将重新扫描当前目录下的所有图片。"):
            self.thumbnail_cache.clear()
            self.deleted_files.clear()
            self.resolution_groups.clear()
            self.sorted_resolutions.clear()
            self.current_res_index = 0
            self.current_img_index = 0
            self.scan_images()

    def scan_images(self):
        """扫描目录下的所有图片，按分辨率分组"""
        if not self.data_dir.exists():
            messagebox.showerror("错误", f"数据目录不存在: {self.data_dir}")
            return

        self.info_var.set("正在扫描图片，请稍候...")
        self.root.update()

        def do_scan():
            self.resolution_groups.clear()
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}

            if self.recursive_mode.get():
                # 递归模式：遍历所有子文件夹
                for img_path in self.data_dir.rglob("*"):
                    if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                        self._add_image_to_group(img_path)
            else:
                # 单文件夹模式：只扫描当前文件夹，不遍历子文件夹
                for item in self.data_dir.iterdir():
                    if item.is_file() and item.suffix.lower() in image_extensions:
                        self._add_image_to_group(item)
                    elif item.is_dir():
                        # 单文件夹模式下，也扫描第一层子文件夹中的图片（保持原有行为）
                        for img_path in item.iterdir():
                            if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                                self._add_image_to_group(img_path)

            self.sorted_resolutions = sorted(self.resolution_groups.keys(),
                                            key=lambda x: int(x.split('x')[0]) * int(x.split('x')[1]),
                                            reverse=True)

            if not self.sorted_resolutions:
                self.info_var.set("未找到任何图片！")
                return

            self.current_res_index = 0
            self.current_img_index = 0
            total_images = sum(len(v) for v in self.resolution_groups.values())
            mode_text = "递归" if self.recursive_mode.get() else "单文件夹"
            self.info_var.set(f"扫描完成！共找到 {total_images} 张图片，{len(self.sorted_resolutions)} 种分辨率 [{mode_text}模式]")
            self.show_current_group()

        threading.Thread(target=do_scan, daemon=True).start()

    def _add_image_to_group(self, img_path):
        """将单张图片添加到分辨率组"""
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                res_key = f"{width}x{height}"
                self.resolution_groups[res_key].append(str(img_path))
        except Exception as e:
            print(f"无法读取图片 {img_path}: {e}")

    def show_current_group(self):
        """显示当前分辨率组的2张图片"""
        if not self.sorted_resolutions:
            return

        res = self.sorted_resolutions[self.current_res_index]
        img_list = self.resolution_groups[res]
        total_img = len(img_list)

        self.res_var.set(f"分辨率: {res} (共{total_img}张)")

        # 计算实际可以显示的图片数量
        start = self.current_img_index
        end = min(start + 2, total_img)

        # 显示左侧图片
        if start < total_img:
            self.show_image(img_list[start], self.left_img_label, self.left_info_var)
        else:
            self.left_img_label.config(image="", text="无图片")
            self.left_info_var.set("")

        # 显示右侧图片
        if start + 1 < total_img:
            self.show_image(img_list[start + 1], self.right_img_label, self.right_info_var)
        else:
            self.right_img_label.config(image="", text="")
            self.right_info_var.set("")

        # 更新进度
        group_num = self.current_res_index + 1
        total_groups = len(self.sorted_resolutions)
        mode_text = "递归" if self.recursive_mode.get() else "单文件夹"
        self.progress_var.set(f"[{mode_text}] 分辨率组: {group_num}/{total_groups} | 当前组图片: {start+1}-{end}/{total_img}")

    def show_image(self, img_path, label, info_var):
        """在Label中显示图片缩略图"""
        try:
            if img_path in self.thumbnail_cache:
                thumb = self.thumbnail_cache[img_path]
            else:
                img = Image.open(img_path)
                # 计算缩略图大小，保持比例
                img.thumbnail((550, 500), Image.Resampling.LANCZOS)
                thumb = ImageTk.PhotoImage(img)
                self.thumbnail_cache[img_path] = thumb

            label.config(image=thumb, text="")
            label.image = thumb  # 保持引用

            # 显示图片信息
            img_name = os.path.basename(img_path)
            if len(img_name) > 40:
                img_name = img_name[:37] + "..."
            # 获取相对路径或子文件夹名
            try:
                rel_path = Path(img_path).relative_to(self.data_dir)
                folder_name = str(rel_path.parent) if rel_path.parent != Path('.') else "根目录"
            except:
                folder_name = os.path.dirname(img_path).split(os.sep)[-1]
            info_var.set(f"{img_name}\n{folder_name}")

        except Exception as e:
            label.config(image="", text=f"加载失败\n{str(e)[:30]}")
            info_var.set("")

    def get_current_images(self):
        """获取当前显示的两张图片路径"""
        if not self.sorted_resolutions:
            return []
        res = self.sorted_resolutions[self.current_res_index]
        img_list = self.resolution_groups[res]
        start = self.current_img_index
        result = []
        if start < len(img_list):
            result.append(img_list[start])
        if start + 1 < len(img_list):
            result.append(img_list[start + 1])
        return result

    def keep_images(self):
        """保留当前显示的两张图片，显示下一组"""
        # 移动到下一张/下一对
        self.advance()

    def delete_images(self):
        """删除当前显示的两张图片"""
        images_to_delete = self.get_current_images()
        if not images_to_delete:
            return

        # 确认对话框
        res = messagebox.askyesno("确认删除", f"确定要删除这 {len(images_to_delete)} 张图片吗？\n此操作不可撤销！")
        if not res:
            return

        for img_path in images_to_delete:
            try:
                # 从分辨率组中移除
                res_key = self.sorted_resolutions[self.current_res_index]
                if img_path in self.resolution_groups[res_key]:
                    self.resolution_groups[res_key].remove(img_path)
                # 删除文件
                os.remove(img_path)
                self.deleted_files.append(img_path)
                print(f"已删除: {img_path}")
            except Exception as e:
                messagebox.showerror("删除失败", f"无法删除 {img_path}\n{str(e)}")

        # 如果当前组没有图片了，移除这个分辨率组
        res_key = self.sorted_resolutions[self.current_res_index]
        if not self.resolution_groups[res_key]:
            del self.resolution_groups[res_key]
            self.sorted_resolutions.pop(self.current_res_index)
            if self.current_res_index >= len(self.sorted_resolutions):
                self.current_res_index = max(0, len(self.sorted_resolutions) - 1)
            self.current_img_index = 0
        else:
            # 调整当前图片索引
            img_list = self.resolution_groups[res_key]
            if self.current_img_index >= len(img_list):
                self.current_img_index = max(0, len(img_list) - 2)

        self.show_current_group()

    def advance(self):
        """前进到下一组图片"""
        if not self.sorted_resolutions:
            return

        res = self.sorted_resolutions[self.current_res_index]
        img_list = self.resolution_groups[res]

        # 如果当前组还有更多图片，显示下一对
        if self.current_img_index + 2 < len(img_list):
            self.current_img_index += 2
        else:
            # 切换到下一个分辨率组
            if self.current_res_index + 1 < len(self.sorted_resolutions):
                self.current_res_index += 1
                self.current_img_index = 0
            else:
                messagebox.showinfo("完成", "所有分辨率组已处理完毕！")
                return

        self.show_current_group()

    def prev_group(self):
        """显示上一组"""
        if not self.sorted_resolutions:
            return

        res = self.sorted_resolutions[self.current_res_index]
        img_list = self.resolution_groups[res]

        # 如果当前组可以往前翻
        if self.current_img_index >= 2:
            self.current_img_index -= 2
        else:
            # 切换到上一个分辨率组
            if self.current_res_index > 0:
                self.current_res_index -= 1
                res = self.sorted_resolutions[self.current_res_index]
                img_list = self.resolution_groups[res]
                # 定位到上一组的最后一对
                self.current_img_index = max(0, len(img_list) - 2)
            else:
                messagebox.showinfo("提示", "已经是第一组了！")
                return

        self.show_current_group()

    def next_group(self):
        """显示下一组（按钮触发）"""
        self.advance()

    def choose_directory(self):
        """选择新的数据目录"""
        new_dir = filedialog.askdirectory(initialdir=str(self.data_dir.parent))
        if new_dir:
            self.data_dir = Path(new_dir)
            self.path_var.set(f"当前目录: {self.data_dir}")
            self.thumbnail_cache.clear()
            self.deleted_files.clear()
            self.resolution_groups.clear()
            self.sorted_resolutions.clear()
            self.current_res_index = 0
            self.current_img_index = 0
            self.scan_images()


def main():
    root = Tk()
    app = ImageCleanerApp(root)
    root.mainloop()

    # 打印删除记录
    if app.deleted_files:
        print(f"\n共删除了 {len(app.deleted_files)} 张图片:")
        for f in app.deleted_files:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
