import os
import yaml
import torch
import cv2
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 配置 
class Config:
    # 原始数据根目录
    RAW_ROOT = r"D:\University\Code\CallPhone+Smoking"
    # 合并后的 YOLO 数据集存放路径
    MERGED_ROOT = "./merged_smoking_phone"

    CLASS_MAPPING = {
        (0, "SmokingData"): 0,   # SmokingData 中的 class 0 映射为新类别 0 (smoking)
        (0, "CallPhoneData"): 1  # CallPhoneData 中的 class 0 映射为新类别 1 (phone_call)
    }
    CLASS_NAMES = ['smoking', 'phone_call']
    
    # 训练参数
    EPOCHS = 1                 
    BATCH_SIZE = 16
    IMG_SIZE = 640
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 预训练权重路径
    YOLOV5_WEIGHT = "./yolov5s.pt"
    YOLOV8_WEIGHT = "./yolov8n.pt"
    
    # 输出目录
    OUTPUT_DIR = "./runs"
    BEST_WEIGHT_DIR = "./best_weights"

cfg = Config()
os.makedirs(cfg.BEST_WEIGHT_DIR, exist_ok=True)
os.makedirs(cfg.MERGED_ROOT, exist_ok=True)

# 1. 合并数据集 
def merge_datasets():
    """将 CallPhoneData 和 SmokingData 合并为统一的 YOLO 格式数据集"""
    splits = ['Train', 'Val', 'Test']  # 原始数据中的划分名称
    target_splits = {'Train': 'train', 'Val': 'val', 'Test': 'test'}
    
    for split in splits:
        target = target_splits[split]
        img_target = Path(cfg.MERGED_ROOT) / target / 'images'
        label_target = Path(cfg.MERGED_ROOT) / target / 'labels'
        img_target.mkdir(parents=True, exist_ok=True)
        label_target.mkdir(parents=True, exist_ok=True)
        
        # 处理两个子数据集
        for sub in ['CallPhoneData', 'SmokingData']:
            src_img_dir = Path(cfg.RAW_ROOT) / sub / split / 'images'
            src_label_dir = Path(cfg.RAW_ROOT) / sub / split / 'labels'
            if not src_img_dir.exists():
                print(f"警告：{src_img_dir} 不存在，跳过")
                continue
            
            # 复制图片并转换标签
            img_files = list(src_img_dir.glob('*.jpg')) + list(src_img_dir.glob('*.png'))
            for img_path in tqdm(img_files, desc=f"合并 {sub} {split}"):
                # 复制图片
                new_img_name = f"{sub}_{split}_{img_path.name}"
                shutil.copy(img_path, img_target / new_img_name)
                
                # 处理标签文件
                label_path = src_label_dir / (img_path.stem + '.txt')
                if not label_path.exists():
                    continue
                
                new_label_path = label_target / (Path(new_img_name).stem + '.txt')
                with open(label_path, 'r') as f_in, open(new_label_path, 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            old_cls = int(parts[0])
                            # 根据子数据集和原始类别映射到新类别
                            key = (old_cls, sub)
                            new_cls = cfg.CLASS_MAPPING.get(key)
                            if new_cls is None:
                                # 如果映射未定义，打印并跳过该标注
                                print(f"警告：未找到映射规则 (old_cls={old_cls}, sub={sub})，文件 {label_path} 中的该行被忽略")
                                continue
                            f_out.write(f"{new_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
        print(f"完成 {split} -> {target}")
    
    # 生成 data.yaml
    data_yaml = {
        'path': cfg.MERGED_ROOT,
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(cfg.CLASS_NAMES),
        'names': cfg.CLASS_NAMES
    }
    yaml_path = Path(cfg.MERGED_ROOT) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"已生成 data.yaml: {yaml_path}")
    return str(yaml_path)

# 2. 数据集可视化 
def visualize_dataset(sample_count=9):
    img_dir = Path(cfg.MERGED_ROOT) / 'train' / 'images'
    label_dir = Path(cfg.MERGED_ROOT) / 'train' / 'labels'
    if not img_dir.exists():
        print("未找到训练集图片目录，跳过可视化")
        return
    
    img_paths = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    if len(img_paths) == 0:
        print("训练集无图片")
        return
    
    samples = random.sample(img_paths, min(sample_count, len(img_paths)))
    cols = 3
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
    axes = axes.flatten() if rows*cols > 1 else [axes]
    
    for idx, img_path in enumerate(samples):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        label_path = label_dir / (img_path.stem + '.txt')
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id, xc, yc, bw, bh = map(float, parts)
                        cls_id = int(cls_id)
                        x1 = int((xc - bw/2) * w)
                        y1 = int((yc - bh/2) * h)
                        x2 = int((xc + bw/2) * w)
                        y2 = int((yc + bh/2) * h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                        label = cfg.CLASS_NAMES[cls_id] if cls_id < len(cfg.CLASS_NAMES) else str(cls_id)
                        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(img_path.name)
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')
    plt.tight_layout()
    plt.savefig('dataset_visualization.png', dpi=150)
    plt.show()
    print("数据集可视化已保存为 dataset_visualization.png")

# 3. 模型训练 
def train_model(model_name, weight_path, yaml_path):
    print(f"\n{'='*30} 开始训练 {model_name} {'='*30}")
    model = YOLO(weight_path)
    results = model.train(
        data=yaml_path,
        epochs=cfg.EPOCHS,
        imgsz=cfg.IMG_SIZE,
        batch=cfg.BATCH_SIZE,
        device=cfg.DEVICE,
        project=cfg.OUTPUT_DIR,
        name=model_name,
        exist_ok=True,
        verbose=True
    )
    val_results = model.val()
    # 保存最佳权重
    best_src = Path(cfg.OUTPUT_DIR) / model_name / 'weights' / 'best.pt'
    if best_src.exists():
        best_dst = Path(cfg.BEST_WEIGHT_DIR) / f'best_{model_name}.pt'
        shutil.copy(best_src, best_dst)
        print(f"最佳权重已保存至 {best_dst}")
    return model, results, val_results

# 4. 绘制对比曲线 
def plot_training_curves():
    model_names = ['YOLOv5s', 'YOLOv8n']
    dfs = {}
    for name in model_names:
        csv_path = Path(cfg.OUTPUT_DIR) / name / 'results.csv'
        if csv_path.exists():
            dfs[name] = pd.read_csv(csv_path)
        else:
            print(f"未找到 {name} 的训练记录，跳过曲线绘制")
            return
    
    metrics = [
        ('train/box_loss', 'Box Loss'),
        ('metrics/mAP50(B)', 'mAP@0.5'),
        ('metrics/precision(B)', 'Precision'),
        ('metrics/recall(B)', 'Recall')
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        for name in model_names:
            if col in dfs[name].columns:
                y = dfs[name][col].dropna()
                ax.plot(range(1, len(y)+1), y, label=name)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()
    print("对比曲线已保存为 model_comparison.png")

# 5. 检测结果可视化 
def detect_and_visualize(model_path, num_images=6):
    test_img_dir = Path(cfg.MERGED_ROOT) / 'test' / 'images'
    if not test_img_dir.exists():
        print("未找到测试图片目录，跳过检测可视化")
        return
    img_paths = list(test_img_dir.glob('*.jpg')) + list(test_img_dir.glob('*.png'))
    if len(img_paths) == 0:
        print("测试集无图片")
        return
    model = YOLO(model_path)
    selected = img_paths[:num_images]
    rows = (num_images + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5*rows))
    axes = axes.flatten()
    for i, img_path in enumerate(selected):
        results = model(img_path, conf=0.25)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        axes[i].imshow(annotated)
        axes[i].axis('off')
        axes[i].set_title(img_path.name)
    for i in range(len(selected), len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('detection_results.png', dpi=150)
    plt.show()
    print("检测结果图已保存为 detection_results.png")

# 6. 视频检测
def detect_video(model_path, video_path, output_path, conf_thres=0.25):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf_thres)
        annotated = results[0].plot()
        out.write(annotated)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"已处理 {frame_count} 帧")
    cap.release()
    out.release()
    print(f"视频检测完成，保存至 {output_path}")

# 主程序 
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    
    # 1. 合并数据集
    yaml_path = Path(cfg.MERGED_ROOT) / 'data.yaml'
    if not yaml_path.exists():
        print("开始合并数据集...")
        yaml_path = merge_datasets()
    else:
        print("数据集已合并，跳过合并步骤。")
    
    # 2. 数据集可视化
    visualize_dataset()
    
    # 3. 训练 YOLOv5 和 YOLOv8
    model_v5, _, _ = train_model("YOLOv5s", cfg.YOLOV5_WEIGHT, yaml_path)
    model_v8, _, _ = train_model("YOLOv8n", cfg.YOLOV8_WEIGHT, yaml_path)
    
    # 4. 绘制训练曲线
    plot_training_curves()
    
    # 5. 图片检测结果可视化
    best_v8 = Path(cfg.BEST_WEIGHT_DIR) / 'best_YOLOv8n.pt'
    if best_v8.exists():
        detect_and_visualize(str(best_v8))
    else:
        print("未找到最佳权重，跳过检测可视化")
        # 6. 视频检测
    detect_video(str(best_v8), "./test_video.mp4", "./detected_video.mp4")
    print("\n所有流程完成")