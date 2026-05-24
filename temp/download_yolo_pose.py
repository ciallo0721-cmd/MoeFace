import urllib.request, ssl, os

ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.onnx'
dest = r'G:\EmoScan Pro\MoeFace\yolo11n-pose.onnx'

# 删除旧文件
if os.path.exists(dest):
    try:
        os.remove(dest)
    except PermissionError:
        import subprocess
        subprocess.run(['cmd', '/c', 'del', '/f', dest], capture_output=True)

print('Downloading yolo11n-pose.onnx...')
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
urllib.request.install_opener(opener)
urllib.request.urlretrieve(url, dest)
size = os.path.getsize(dest)
print(f'Done: {size//1024//1024} MB')

# 验证 ONNX 文件头
with open(dest, 'rb') as f:
    header = f.read(4)
print(f'Header: {header.hex()} (should start with 08 for valid ONNX)')
