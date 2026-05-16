"""
导出 FaceNet 模型为 TensorFlow.js 格式
以及导出预计算的 512 维特征库

运行方式：
    python export_tfjs.py

输出：
    web/model/           - TensorFlow.js 模型
    web/features/        - 512维特征库（JSON）
"""
import os
import json
import numpy as np
from pathlib import Path
import torch
from facenet_pytorch import InceptionResnetV1
import torch.nn as nn

# 路径配置
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "web"
MODEL_DIR = OUTPUT_DIR / "model"
FEATURES_DIR = OUTPUT_DIR / "features"

def export_facenet_to_tfjs():
    """导出 FaceNet 模型为 TensorFlow.js 格式"""
    print("正在加载 FaceNet 模型...")
    
    device = torch.device("cpu")  # 导出时用CPU
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    
    # 创建输出目录
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 导出手写数字测试输入 (batch_size=1, channels=3, height=160, width=160)
    # FaceNet 期望输入是 160x160
    dummy_input = torch.randn(1, 3, 160, 160)
    
    # 导出为 TorchScript
    print("正在导出为 TorchScript...")
    scripted_model = torch.jit.trace(model, dummy_input)
    scripted_model.save(str(MODEL_DIR / "facenet.pt"))
    
    # 导出为 ONNX
    print("正在导出为 ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        str(MODEL_DIR / "facenet.onnx"),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11
    )
    
    print(f"模型已导出到: {MODEL_DIR}")
    print("提示: 需要使用 onnx-tf 或 tf2onnx 转换为 TensorFlow.js 格式")
    print("pip install onnx onnx-tf tensorflow tfjs")
    
    return MODEL_DIR / "facenet.onnx"

def export_features_for_browser():
    """导出浏览器可用的特征库（与本地相同格式）"""
    print("\n正在导出浏览器特征库...")
    
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 读取现有的512维特征库
    local_features_path = BASE_DIR / "features" / "全部特征库.json"
    if local_features_path.exists():
        with open(local_features_path, 'r', encoding='utf-8') as f:
            features = json.load(f)
        
        # 复制到 web 目录
        web_features_path = FEATURES_DIR / "全部特征库.json"
        with open(web_features_path, 'w', encoding='utf-8') as f:
            json.dump(features, f, ensure_ascii=False, indent=2)
        
        print(f"特征库已复制到: {web_features_path}")
        print(f"角色数量: {len(features)}")
        print(f"特征维度: {len(list(features.values())[0])}")
    else:
        print("警告: 未找到本地特征库，跳过导出")

def convert_onnx_to_tfjs():
    """尝试将 ONNX 转换为 TensorFlow.js"""
    try:
        import onnx
        from onnx_tf import backend as tf_backend
        import tensorflow as tf
        import tfjs as tfjs_converter
        
        onnx_path = MODEL_DIR / "facenet.onnx"
        tf_model_path = MODEL_DIR / "facenet_tf"
        
        if not onnx_path.exists():
            print("ONNX 模型不存在，跳过转换")
            return
        
        print("\n正在转换为 TensorFlow 格式...")
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = tf_backend.prepare(onnx_model)
        tf_rep.export_graph(str(tf_model_path))
        
        print("正在转换为 TensorFlow.js 格式...")
        tfjs_converter.convert_from_saved_model(
            str(tf_model_path),
            output_dir=str(MODEL_DIR / "tfjs"),
            signature="serving_default"
        )
        
        print(f"TensorFlow.js 模型已导出到: {MODEL_DIR / 'tfjs'}")
        
    except ImportError as e:
        print(f"\n转换工具未安装: {e}")
        print("请手动运行以下命令转换模型:")
        print("  pip install onnx onnx-tf tensorflow tfjs")
        print(f"  python -c \"from onnx_tf import backend as tf_backend; tf_backend.prepare(onnx.load('{MODEL_DIR / 'facenet.onnx'}')).export_graph('{MODEL_DIR / 'facenet_tf'}')\"")
        print(f"  tensorflowjs_converter --input_format=tf_saved_model {MODEL_DIR / 'facenet_tf'} {MODEL_DIR / 'tfjs'}")
    except Exception as e:
        print(f"\n转换失败: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("MoeFace TensorFlow.js 导出工具")
    print("=" * 50)
    
    # 导出特征库（必需）
    export_features_for_browser()
    
    # 导出模型（可选，需要转换）
    try:
        export_facenet_to_tfjs()
        convert_onnx_to_tfjs()
    except Exception as e:
        print(f"\n模型导出失败: {e}")
        print("\n请确保安装了必要的库:")
        print("  pip install torch facenet-pytorch onnx onnx-tf tensorflow tfjs")
    
    print("\n" + "=" * 50)
    print("导出完成!")
    print("=" * 50)
