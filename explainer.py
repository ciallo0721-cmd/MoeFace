"""
MoeFace 模型可解释性模块 — 注意力热力图 (Grad-CAM)

通过修改 FaceNet 模型的前向传播钩子，提取最后一层卷积的梯度，
生成注意力热力图叠加在输入图像上。
"""
import numpy as np
import torch
import cv2

# ── 全局缓存 ──
_hook_handles = []
_gradients = None
_activations = None


def _forward_hook(module, input, output):
    """存储最后一层卷积的激活值"""
    global _activations
    _activations = output


def _backward_hook(module, grad_in, grad_out):
    """存储梯度"""
    global _gradients
    _gradients = grad_out[0]


def register_hooks(model):
    """注册 Grad-CAM 钩子到 FaceNet 模型（InceptionResnetV1）"""
    global _hook_handles
    _clear_hooks()

    # InceptionResnetV1 使用 nn.Sequential，找到 Conv2d 层
    for name, module in model.named_modules():
        # 取最后一个 Conv2d 层（Block8 或 Mixed_7a 等）
        if isinstance(module, torch.nn.Conv2d):
            continue  # 先扫描
    # 取 model.last_linear 之前的特征层
    if hasattr(model, 'logits'):
        target = model.logits
    else:
        # 找最后一个激活函数前的 conv 层
        target = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target = module
                break

    if target is not None:
        h1 = target.register_forward_hook(_forward_hook)
        h2 = target.register_full_backward_hook(_backward_hook)
        _hook_handles = [h1, h2]
        return True
    return False


def _clear_hooks():
    global _hook_handles
    for h in _hook_handles:
        try:
            h.remove()
        except Exception:
            pass
    _hook_handles = []


def generate_gradcam(model, input_tensor: torch.Tensor, target_class=None):
    """
    生成 Grad-CAM 热力图

    Args:
        model: FaceNet 模型
        input_tensor: [1, 3, 160, 160] 归一化张量
        target_class: 目标类别索引（None=使用最高得分）

    Returns:
        heatmap: (160, 160) 热力图（0~1）
    """
    global _gradients, _activations
    _gradients = None
    _activations = None

    # 确保钩子已注册
    if not _hook_handles:
        register_hooks(model)

    # 前向传播
    model.zero_grad()
    output = model(input_tensor)

    if _activations is None:
        # 钩子未触发，返回空热力图
        return np.zeros((160, 160), dtype=np.float32)

    # 构造目标：取最高激活的神经元
    if target_class is None:
        # FaceNet 输出 512 维嵌入，取最大激活维度
        target = output[0].max()
        model.zero_grad()
        target.backward(retain_graph=True)
    else:
        target = output[0, target_class]
        model.zero_grad()
        target.backward(retain_graph=True)

    if _gradients is None:
        return np.zeros((160, 160), dtype=np.float32)

    # Grad-CAM 计算
    weights = _gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = (weights * _activations).sum(dim=1)  # [1, H, W]
    cam = torch.relu(cam)  # 只保留正激活

    # 缩放到输入尺寸
    cam = cam.squeeze().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    # 缩放到 160x160
    cam = cv2.resize(cam, (160, 160), interpolation=cv2.INTER_LINEAR)
    return cam


def overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5, colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """
    将热力图叠加到原图上

    Args:
        image_bgr: BGR 图像
        heatmap: (H, W) 热力图，值域 [0, 1]
        alpha: 叠加透明度
        colormap: OpenCV 颜色映射

    Returns:
        叠加后的 BGR 图像
    """
    h, w = image_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), colormap)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def explain_face(model, face_img_rgb: np.ndarray) -> np.ndarray:
    """
    对人脸图像一键生成热力图叠加图

    Args:
        model: FaceNet 模型
        face_img_rgb: RGB 人脸图像 (任意尺寸)

    Returns:
        overlay_bgr: 叠加热力图的 BGR 图像
    """
    # 预处理
    face = cv2.resize(face_img_rgb, (160, 160))
    tensor = (torch.tensor(face).permute(2, 0, 1)
              .float().unsqueeze(0) / 255.0)

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    heatmap = generate_gradcam(model, tensor)
    face_bgr = cv2.cvtColor(face_img_rgb, cv2.COLOR_RGB2BGR)
    result = overlay_heatmap(face_bgr, heatmap)
    return result
