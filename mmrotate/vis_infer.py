import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import cm
from mmdet.apis import init_detector, inference_detector

# ====== 导入自定义 Detector 类 ======
sys.path.append('/root/mmdetection')  # 改为你的 mmdetection 根目录
from mmrotate.models.detectors import my_detector_SGM_CAMM_GNN_Constrastive

# ====== 参数配置 ======
config_file = '/root/mmdetection/mmrotate/configs/custom_cfg_new_SGM_CAMM_GNN_Contrastive.py'
checkpoint_file = '/root/mmdetection/work_dirs/custom_cfg_new_SGM_CAMM_GNN_Con_ablation-study_54turns-82.3mAP/epoch_54.pth'
img_dir = '/root/mmdetection/data/Test/JPEGImages'
save_dir = '/root/mmdetection/camm_best_vis'
device = 'cuda:0'

alpha = 0.3     # 方差 + 高响应占比综合权重
conf_thr = 0.3  # 检测框分数阈值
blur_radius = 25  # 控制船舶与背景的平滑过渡

os.makedirs(save_dir, exist_ok=True)

# ====== 初始化模型 ======
model = init_detector(config_file, checkpoint_file, device=device)
model.eval()

def process_image(img_path):
    """推理选最佳尺度的 CAMM mask，并返回检测结果"""
    best_mask = [None]
    best_idx = [-1]
    det_result_holder = [None]

    def camm_hook(module, inp, out):
        masks_tensor_list = out['masks']  # list[num_scales]
        masks_numpy = []
        for m in masks_tensor_list:
            m_np = m.detach().cpu().numpy()
            if m_np.ndim == 4:  # [B,C,H,W]
                m_np = m_np[0]
            if m_np.ndim == 3:  # [C,H,W]
                m_np = m_np.mean(axis=0)
            masks_numpy.append(m_np)

        # 根据方差+高响应比例选最佳尺度
        scores = []
        for idx, m in enumerate(masks_numpy):
            m_norm = (m - m.min()) / (m.max() - m.min() + 1e-8)
            var = np.var(m_norm)
            high_ratio = np.mean(m_norm > 0.5)
            score = var + alpha * high_ratio
            scores.append(score)

        sel_idx = int(np.argmax(scores))
        best_idx[0] = sel_idx
        best_mask[0] = masks_numpy[sel_idx]

    # 注册hook获取CAMM mask
    hook_handle = model.camm_module.register_forward_hook(camm_hook)
    det_result = inference_detector(model, img_path)  # 推理获取检测框
    det_result_holder[0] = det_result
    hook_handle.remove()

    return best_mask[0], best_idx[0], det_result_holder[0]

def save_camm_vis_gradcam_style(img_path, mask, best_idx, det_result,
                                 conf_thr=0.3, blur_radius=45):
    """
    CAMM + Grad-CAM风格显著性可视化
    - 船体内部红黄渐变
    - 船体外有渐变halo
    - 船外低响应为深蓝
    """
    orig_img = cv2.imread(img_path)
    H, W = orig_img.shape[:2]

    # 归一化 mask 并插值到原图大小
    mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask_resized = cv2.resize(mask_norm, (W, H))

    # -------- 生成船舶掩膜 --------
    ship_mask = np.zeros((H, W), dtype=np.uint8)
    for cls_result in det_result:
        if cls_result is None or len(cls_result) == 0:
            continue
        for box in cls_result:
            if box[-1] < conf_thr:
                continue
            cx, cy, bw, bh, angle, score = box
            rect = ((float(cx), float(cy)),
                    (float(bw), float(bh)),
                    float(angle) * 180.0 / np.pi)
            poly_pts = cv2.boxPoints(rect)
            poly_pts = np.int0(poly_pts)
            cv2.fillPoly(ship_mask, [poly_pts], 255)

    # -------- 模糊船舶掩膜，形成 halo --------
    halo_mask = cv2.GaussianBlur(ship_mask, (blur_radius, blur_radius), 0)
    halo_mask = halo_mask.astype(np.float32) / 255.0
    halo_mask = np.clip(halo_mask, 0, 1)

    # -------- 用 halo 扩展显著性范围 --------
    mask_with_halo = mask_resized * 0.7 + halo_mask * 0.3  # 原显著性+halo混合
    mask_with_halo = np.clip(mask_with_halo, 0, 1)

    # -------- 生成 Jet 热力图 --------
    heatmap_color = (cm.jet(mask_with_halo)[:, :, :3] * 255).astype(np.uint8)
    heatmap_bgr = cv2.cvtColor(heatmap_color, cv2.COLOR_RGB2BGR)

    # -------- Overlay（Grad-CAM 风格）--------
    overlay = cv2.addWeighted(heatmap_bgr, 0.6, orig_img, 0.4, 0)

    # 保存结果
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    overlay_path = os.path.join(save_dir, f"{base_name}_bestscale{best_idx}_gradcam.png")
    cv2.imwrite(overlay_path, overlay)


# ====== 批量处理 ======
img_files = [
    os.path.join(img_dir, f)
    for f in os.listdir(img_dir)
    if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif'))
]

print(f"共 {len(img_files)} 张图片，开始处理...")
for img_path in tqdm(img_files):
    best_mask, best_idx, det_result = process_image(img_path)
    save_camm_vis_gradcam_style(img_path, best_mask, best_idx, det_result)

print(f"处理完成！结果保存在 {save_dir}")
