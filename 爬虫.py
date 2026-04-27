import os
import sys
import requests
import time
import random
import urllib.parse
import json
import re
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup

# ======================== 配置区域 ========================
DOWNLOAD_THREADS = 5
REQUEST_DELAY = 0.5
REQUEST_TIMEOUT = (5, 10)          # (连接超时, 读取超时) 单位秒

ENABLE_BING = True
ENABLE_GOOGLE = True
ENABLE_BAIDU = True
ENABLE_PIXIV = False
ENABLE_SAFEBOORU = True

PIXIV_COOKIE = ""

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}

ROLE_SUFFIXES = {
    "丛雨": ["立绘","壁纸","千恋万花","Murasame","ムラサメ","丛雨丸"],
    "Neuro-sama": ["vtuber","art","fanart","Neuro sama","Neurosama","AI vtuber"],
    "永雏塔菲": ["虚拟主播","立绘","Taffy","Ace Taffy","唐人塔菲"],
    "东雪莲": ["虚拟主播","立绘","東雪蓮","Yukiren","罕见"],
    "ShikiNatsume": ["立绘","壁纸","棗シキ","Shiki Natsume","枣子姐"],
    "棍母": ["棍娘","Gun Mu","电棍母亲","电棍妈妈"],
    "otto": ["吉吉国王","电棍","otto lol","帅,otto"],
    "Ayachi_Nene": ["立绘","壁纸","綾地寧々","Nene Ayachi","桌角战士"]
}

# ======================== 日志函数（必须定义在使用之前） ========================
def log(msg):
    """带时间戳的日志"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ======================== 加载负面词列表 ========================
NEGATIVE_WORDS_FILE = "./cname/pachong.json"
negative_words = []

def load_negative_words():
    """从JSON文件加载负面词列表，返回字符串列表"""
    global negative_words
    if not os.path.exists(NEGATIVE_WORDS_FILE):
        log(f"警告: 负面词文件 {NEGATIVE_WORDS_FILE} 不存在，将不进行内容过滤")
        return []
    try:
        with open(NEGATIVE_WORDS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            words = data
        elif isinstance(data, dict) and 'words' in data:
            words = data['words']
        else:
            words = []
        # 统一转为小写，便于匹配
        negative_words = [str(w).lower() for w in words if w]
        log(f"已加载 {len(negative_words)} 个负面词")
        return negative_words
    except json.JSONDecodeError as e:
        log(f"负面词文件 JSON 格式错误: {e}")
        log(f"请检查文件 {NEGATIVE_WORDS_FILE} 的格式，应为 ['词1', '词2'] 或 {{'words': ['词1', '词2']}}")
        return []
    except Exception as e:
        log(f"加载负面词文件失败: {e}")
        return []

def contains_negative_word(text):
    """检查文本中是否包含任意负面词（不区分大小写）"""
    if not negative_words:
        return False
    lower_text = text.lower()
    return any(word in lower_text for word in negative_words)

# 程序启动时加载负面词
load_negative_words()

# ======================== 图源函数（增加超时和日志） ========================
def get_bing_images(keyword, num):
    log(f"Bing 开始搜索: {keyword}")
    urls = []
    start = 1
    max_pages = 3  # 最多翻3页
    page = 0
    while len(urls) < num and page < max_pages:
        search_url = f"https://cn.bing.com/images/search?q={urllib.parse.quote(keyword)}&first={start}"
        try:
            log(f"  Bing 请求第{page+1}页: {search_url[:80]}...")
            resp = requests.get(search_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            soup = BeautifulSoup(resp.text, 'html.parser')
            links = soup.find_all('a', class_='iusc')
            if not links:
                log(f"  Bing 第{page+1}页没有找到链接")
                break
            for link in links:
                m = link.get('m')
                if m:
                    try:
                        img_url = json.loads(m).get('murl')
                        if img_url and img_url not in urls:
                            urls.append(img_url)
                            if len(urls) >= num:
                                break
                    except:
                        continue
            start += len(links)
            page += 1
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            log(f"  Bing 出错: {e}")
            break
    log(f"Bing 结束，获得 {len(urls)} 个URL")
    return urls[:num]

def get_baidu_images(keyword, num):
    log(f"百度 开始搜索: {keyword}")
    urls = []
    pn = 0
    max_pages = 3
    page = 0
    while len(urls) < num and page < max_pages:
        search_url = f"https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word={urllib.parse.quote(keyword)}&pn={pn}"
        try:
            log(f"  百度请求第{page+1}页: {search_url[:80]}...")
            resp = requests.get(search_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            pattern = r'"objURL":"(https?://[^"]+)"'
            matches = re.findall(pattern, resp.text)
            if not matches:
                log(f"  百度第{page+1}页没有找到图片链接")
                break
            for url in matches:
                url = url.replace('\\/', '/')
                if url and url not in urls:
                    urls.append(url)
                    if len(urls) >= num:
                        break
            pn += 60
            page += 1
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            log(f"  百度出错: {e}")
            break
    log(f"百度结束，获得 {len(urls)} 个URL")
    return urls[:num]

def get_google_images(keyword, num):
    log(f"Google 开始搜索: {keyword}")
    urls = []
    start = 0
    max_pages = 2
    page = 0
    while len(urls) < num and page < max_pages:
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(keyword)}&tbm=isch&start={start}"
        try:
            log(f"  Google请求第{page+1}页")
            resp = requests.get(search_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            soup = BeautifulSoup(resp.text, 'html.parser')
            img_tags = soup.find_all('img')
            for img in img_tags:
                src = img.get('data-src') or img.get('src')
                if src and src.startswith('http') and src not in urls:
                    urls.append(src)
                    if len(urls) >= num:
                        break
            if not img_tags:
                break
            start += 20
            page += 1
            time.sleep(REQUEST_DELAY + random.uniform(0.5, 1.5))
        except Exception as e:
            log(f"  Google出错: {e}")
            break
    log(f"Google结束，获得 {len(urls)} 个URL")
    return urls[:num]

def get_pixiv_images(keyword, num):
    if not PIXIV_COOKIE:
        log("Pixiv 未配置Cookie，跳过")
        return []
    log(f"Pixiv 开始搜索: {keyword}")
    urls = []
    headers = HEADERS.copy()
    headers['Cookie'] = PIXIV_COOKIE
    headers['Referer'] = 'https://www.pixiv.net/'
    page = 1
    max_pages = 2
    while len(urls) < num and page <= max_pages:
        api_url = f"https://www.pixiv.net/ajax/search/artworks/{urllib.parse.quote(keyword)}?word={urllib.parse.quote(keyword)}&order=date_d&mode=all&p={page}"
        try:
            log(f"  Pixiv请求第{page}页")
            resp = requests.get(api_url, headers=headers, timeout=REQUEST_TIMEOUT)
            data = resp.json()
            if data.get('error'):
                break
            works = data.get('body', {}).get('illustManga', {}).get('data', [])
            if not works:
                break
            for work in works:
                illust_id = work.get('id')
                if illust_id:
                    detail_url = f"https://www.pixiv.net/ajax/illust/{illust_id}"
                    detail_resp = requests.get(detail_url, headers=headers, timeout=REQUEST_TIMEOUT)
                    detail_data = detail_resp.json()
                    urls_big = detail_data.get('body', {}).get('urls', {}).get('original')
                    if urls_big and urls_big not in urls:
                        urls.append(urls_big)
                        if len(urls) >= num:
                            break
            page += 1
            time.sleep(REQUEST_DELAY * 2)
        except Exception as e:
            log(f"  Pixiv出错: {e}")
            break
    log(f"Pixiv结束，获得 {len(urls)} 个URL")
    return urls[:num]

def get_safebooru_images(keyword, num):
    log(f"Safebooru 开始搜索: {keyword}")
    urls = []
    page = 0
    max_pages = 3
    while len(urls) < num and page < max_pages:
        api_url = f"https://safebooru.org/index.php?page=dapi&s=post&q=index&tags={urllib.parse.quote(keyword)}&pid={page}&limit=40"
        try:
            log(f"  Safebooru请求第{page+1}页")
            resp = requests.get(api_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            soup = BeautifulSoup(resp.text, 'xml')
            posts = soup.find_all('post')
            if not posts:
                log(f"  Safebooru第{page+1}页没有post")
                break
            for post in posts:
                file_url = post.get('file_url')
                if file_url and file_url.startswith('https://') and file_url not in urls:
                    urls.append(file_url)
                    if len(urls) >= num:
                        break
            page += 1
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            log(f"  Safebooru出错: {e}")
            break
    log(f"Safebooru结束，获得 {len(urls)} 个URL")
    return urls[:num]

SOURCE_GETTERS = []
if ENABLE_BING:
    SOURCE_GETTERS.append(('Bing', get_bing_images))
if ENABLE_BAIDU:
    SOURCE_GETTERS.append(('百度', get_baidu_images))
if ENABLE_GOOGLE:
    SOURCE_GETTERS.append(('Google', get_google_images))
if ENABLE_PIXIV:
    SOURCE_GETTERS.append(('Pixiv', get_pixiv_images))
if ENABLE_SAFEBOORU:
    SOURCE_GETTERS.append(('Safebooru', get_safebooru_images))

def collect_urls_from_sources(keyword, target_num, suffixes):
    all_urls = []
    keywords_to_try = [keyword] + [f"{keyword} {suffix}" for suffix in suffixes]
    for kw in keywords_to_try:
        if len(all_urls) >= target_num:
            break
        log(f"尝试关键词: {kw}")
        for src_name, getter in SOURCE_GETTERS:
            if len(all_urls) >= target_num:
                break
            need = target_num - len(all_urls)
            log(f"  使用图源 {src_name} 获取最多 {need*2} 个URL...")
            urls = getter(kw, need * 2) if need > 0 else []
            # 过滤包含负面词的URL
            filtered_urls = []
            for u in urls:
                if contains_negative_word(u):
                    log(f"    负面词过滤跳过URL: {u[:80]}...")
                elif u not in all_urls:
                    filtered_urls.append(u)
            new_urls = filtered_urls
            all_urls.extend(new_urls)
            log(f"    {src_name} 新增 {len(new_urls)} 个URL（共获得{len(urls)}，过滤掉{len(urls)-len(new_urls)}个），累计 {len(all_urls)}")
            time.sleep(0.2)
    return all_urls[:target_num]

def download_image(url, dir_path, base_name):
    # 下载前再次检查负面词（双重保险）
    if contains_negative_word(url):
        log(f"下载跳过（负面词）: {url[:80]}...")
        return False
    try:
        resp = requests.get(url, headers=HEADERS, timeout=(5, 15), stream=True)
        if resp.status_code != 200:
            return False
        content_type = resp.headers.get('content-type', '')
        ext = None
        if 'jpeg' in content_type or 'jpg' in content_type:
            ext = '.jpg'
        elif 'png' in content_type:
            ext = '.png'
        elif 'gif' in content_type:
            ext = '.gif'
        elif 'webp' in content_type:
            ext = '.webp'
        else:
            path = urllib.parse.urlparse(url).path
            guess = os.path.splitext(path)[1].lower()
            ext = guess if guess in ['.jpg','.jpeg','.png','.gif','.webp'] else '.jpg'
        
        file_path = os.path.join(dir_path, base_name + ext)
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(dir_path, f"{base_name}_{counter}{ext}")
            counter += 1
        
        with open(file_path, 'wb') as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
        
        if os.path.getsize(file_path) > 1024:
            return True
        else:
            os.remove(file_path)
            return False
    except Exception:
        return False

def get_roles_from_dir(root_dir):
    if not os.path.isdir(root_dir):
        return []
    items = os.listdir(root_dir)
    subdirs = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
    if subdirs:
        roles = []
        for sub in subdirs:
            role_path = os.path.join(root_dir, sub)
            roles.append((sub, role_path))
        return roles
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
    has_images = any(f.lower().endswith(image_extensions) for f in items if os.path.isfile(os.path.join(root_dir, f)))
    if has_images:
        role_name = os.path.basename(root_dir)
        return [(role_name, root_dir)]
    return []

def crawl_all_roles(root_dir, max_images_per_role):
    if not os.path.exists(root_dir):
        print(f"错误：目录不存在 {root_dir}")
        return
    if not os.path.isdir(root_dir):
        print(f"错误：路径不是目录 {root_dir}")
        return
    
    roles = get_roles_from_dir(root_dir)
    if not roles:
        print(f"警告：在 {root_dir} 下没有找到角色文件夹或图片")
        return
    
    print(f"共找到 {len(roles)} 个角色：{', '.join([name for name, _ in roles])}")
    
    for role_name, role_path in roles:
        print(f"\n{'='*40}\n处理角色: {role_name}")
        target_dir = role_path
        os.makedirs(target_dir, exist_ok=True)
        
        existing = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.gif','.webp'))]
        base_index = len(existing)
        print(f"  已有 {base_index} 张")
        
        suffixes = ROLE_SUFFIXES.get(role_name, ["立绘","壁纸","art","fanart","illustration"])
        print(f"  搜索关键词: {role_name}, 后缀: {suffixes}")
        urls = collect_urls_from_sources(role_name, max_images_per_role, suffixes)
        print(f"共获取到 {len(urls)} 个有效URL（已过滤负面词）")
        
        if not urls:
            print("  没有获取到任何URL，跳过下载")
            continue
        
        success = 0
        with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as executor:
            future_to_idx = {}
            for idx, img_url in enumerate(urls):
                base_name = f"{role_name}_{base_index + idx}"
                future = executor.submit(download_image, img_url, target_dir, base_name)
                future_to_idx[future] = (idx, img_url, base_name)
            
            for future in as_completed(future_to_idx):
                idx, img_url, base_name = future_to_idx[future]
                try:
                    if future.result():
                        success += 1
                        print(f"  ✅ {idx+1}: {base_name}.*")
                    else:
                        print(f"  ❌ {idx+1}: 下载失败")
                except Exception as e:
                    print(f"  ❌ {idx+1}: 异常 - {e}")
        
        print(f"角色 {role_name} 完成，成功 {success} 张")
    print("\n所有角色处理完毕！")

# ======================== GUI 界面（与原版相同） ========================
class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    def write(self, msg):
        self.text_widget.after(0, self._insert, msg)
    def _insert(self, msg):
        self.text_widget.insert(tk.END, msg)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()
    def flush(self):
        pass

class CrawlerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("简陋图片爬虫")
        self.root.geometry("700x500")
        self.running = False
        self.original_stdout = sys.stdout
        
        frame_path = tk.Frame(self.root)
        frame_path.pack(pady=10, padx=10, fill=tk.X)
        tk.Label(frame_path, text="目标文件夹:").pack(side=tk.LEFT, padx=5)
        self.path_var = tk.StringVar()
        self.entry_path = tk.Entry(frame_path, textvariable=self.path_var, width=50)
        self.entry_path.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.btn_browse = tk.Button(frame_path, text="浏览", command=self.browse_folder)
        self.btn_browse.pack(side=tk.LEFT, padx=5)
        
        frame_num = tk.Frame(self.root)
        frame_num.pack(pady=5, padx=10, fill=tk.X)
        tk.Label(frame_num, text="每个角色图片数:").pack(side=tk.LEFT, padx=5)
        self.num_var = tk.StringVar(value="10")
        self.entry_num = tk.Entry(frame_num, textvariable=self.num_var, width=10)
        self.entry_num.pack(side=tk.LEFT, padx=5)
        
        self.btn_start = tk.Button(self.root, text="开始爬取", command=self.start_crawling, bg="lightgreen")
        self.btn_start.pack(pady=10)
        
        self.log_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=25)
        self.log_text.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        self.log_text.insert(tk.END, "请选择存放角色图片的根目录。\n")
        self.log_text.insert(tk.END, "支持两种结构：\n")
        self.log_text.insert(tk.END, "  1. 根目录下直接包含多个角色文件夹（每个文件夹一个角色）\n")
        self.log_text.insert(tk.END, "  2. 根目录本身就是某个角色的目录（目录内已有图片）\n")
        self.log_text.insert(tk.END, "然后点击「开始爬取」\n")
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.path_var.set(folder)
    
    def start_crawling(self):
        if self.running:
            self.log_text.insert(tk.END, "爬取任务已在运行中，请稍后...\n")
            return
        root_dir = self.path_var.get().strip()
        if not root_dir:
            self.log_text.insert(tk.END, "请先选择目标文件夹！\n")
            return
        try:
            max_num = int(self.num_var.get())
            if max_num <= 0:
                raise ValueError
        except ValueError:
            self.log_text.insert(tk.END, "图片数量必须是正整数！\n")
            return
        
        self.running = True
        self.btn_start.config(state=tk.DISABLED, text="爬取中...")
        self.log_text.insert(tk.END, f"开始爬取，根目录：{root_dir}，每个角色最多 {max_num} 张\n")
        self.log_text.see(tk.END)
        
        thread = threading.Thread(target=self._run_crawler, args=(root_dir, max_num))
        thread.daemon = True
        thread.start()
    
    def _run_crawler(self, root_dir, max_num):
        sys.stdout = StdoutRedirector(self.log_text)
        try:
            crawl_all_roles(root_dir, max_num)
        except Exception as e:
            print(f"爬取过程中发生异常: {e}")
        finally:
            sys.stdout = self.original_stdout
            self.root.after(0, self._crawl_finished)
    
    def _crawl_finished(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL, text="开始爬取")
        self.log_text.insert(tk.END, "\n爬取任务结束。\n")
        self.log_text.see(tk.END)
    
    def on_close(self):
        if self.running:
            self.log_text.insert(tk.END, "正在爬取中，请等待完成后再关闭...\n")
            return
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = CrawlerGUI()
    app.run()
