# YOLOv5 vs YOLOv8：抽烟 / 接打电话行为目标检测对比实验

> Object detection benchmark of YOLOv5s and YOLOv8n on a smoking & phone-call behavior dataset (custom 2-class annotation, transfer learning with official pretrained weights, 640×640, 50 epochs, RTX 4060 Laptop).

深度学习实验课程结课项目 · 广州商学院 信息技术与工程学院 人工智能专业

---

## 1. 项目简介

本项目在同一份自建双类别数据集上，完整对比 **YOLOv5s** 与 **YOLOv8n** 两个单阶段检测模型（均基于 Ultralytics 官方预训练权重做迁移学习），从数据预处理、训练过程、验证集指标到实际检测效果进行了端到端比较，并输出可视化对比结论。

实验目标不是追求绝对精度，而是回答一个工程问题：**在中等规模、标注质量参差的行为数据集上，两代模型各自的取舍是什么。**

## 2. 数据集说明

| 项目 | 内容 |
| --- | --- |
| 数据来源 | CSDN datacanvas 公开分享数据集（[来源页面](https://datacanvas.csdn.net/691e79955511483559ec11f3.html)） |
| 图像数量 | 1559 张（已标注，YOLO txt 格式） |
| 检测类别 | 2 类：`smoking`（抽烟）、`phone_call`（接打电话） |
| 图像尺寸 | 原始尺寸不一，训练前统一缩放至 **640×640** |
| 划分方式 | 沿用原始 Train / Val / Test 划分 |
| 合并方式 | 将 `SmokingData`、`CallPhoneData` 两个子集的原始 class 0 分别重映射为 `smoking`(0) 与 `phone_call`(1)，合并为统一数据集 |

> 已知数据问题：`phone_call` 类别的原始标注存在异常，导致模型实际上主要学习到 `smoking` 类别的特征；部分图片文件损坏，修复后引入少量噪声。详见「实验结果分析」。

## 3. 项目结构

```
.
├── dl_ex_final.py                 # 主流程脚本（数据合并 → 训练 → 评估 → 可视化）
├── YOLOv5和YOLOv8对比实现...docx   # 完整实验报告
├── merged_smoking_phone/          # 合并后的 YOLO 格式数据集（脚本自动生成）
│   ├── train/{images,labels}
│   ├── val/{images,labels}
│   ├── test/{images,labels}
│   └── data.yaml
├── runs/                          # Ultralytics 训练输出（results.csv / 权重 / 曲线）
├── best_weights/                  # 两个模型的最优权重
│   ├── best_YOLOv5s.pt
│   └── best_YOLOv8n.pt
├── dataset_visualization.png      # 数据集标注可视化
├── model_comparison.png           # 训练指标对比曲线
├── detection_results.png          # 测试集检测效果
└── detected_video.mp4             # 视频推理结果（可选）
```

## 4. 环境依赖

```bash
# Python >= 3.8，建议使用独立虚拟环境
pip install ultralytics opencv-python matplotlib pandas scikit-learn tqdm pyyaml torch
```

- 实测硬件：NVIDIA GeForce RTX 4060 Laptop GPU（device 自动检测 CUDA，无 GPU 时回落 CPU，但训练会非常慢）
- 预训练权重：`yolov5s.pt`、`yolov8n.pt`（Ultralytics 官方权重，置于项目根目录）

## 5. 快速开始

1. 修改 `dl_ex_final.py` 中的 `Config` 配置：

```python
RAW_ROOT = r"<你的原始数据集根目录>"   # 需包含 CallPhoneData / SmokingData 两个子目录
EPOCHS   = 50                        # 复现实验请设为 50；脚本默认 1 用于快速跑通流程
BATCH_SIZE = 16
IMG_SIZE   = 640
```

2. 运行主流程：

```bash
python dl_ex_final.py
```

脚本将依次执行：**数据集合并与标签重映射 → data.yaml 生成 → 数据集可视化 → 训练 YOLOv5s → 训练 YOLOv8n → 指标曲线对比 → 测试集检测可视化 → 视频推理**，各阶段产物见上节目录说明。

3. 单独推理：

```python
from ultralytics import YOLO
model = YOLO("best_weights/best_YOLOv8n.pt")
model("your_image.jpg", conf=0.25)      # 图片
model("your_video.mp4", save=True)      # 视频
```

> 说明：主流程中的视频检测默认调用 `best_weights/best_YOLOv8n.pt`，如需用 YOLOv5 推理，修改 `detect_video()` 的 `model_path` 参数即可。

## 6. 训练配置

| 参数 | 取值 |
| --- | --- |
| 预训练权重 | `yolov5s.pt` / `yolov8n.pt`（官方 COCO 权重） |
| Epochs | 50 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Device | CUDA（RTX 4060 Laptop） |
| 框架 | Ultralytics（同一套 API 调用两代模型，保证对比公平） |

## 7. 实验结果

### 7.1 验证集指标对比（第 50 轮）

| 模型 | Box Loss (val) | mAP@0.5 | Precision | Recall |
| --- | --- | --- | --- | --- |
| YOLOv5s | 1.66137 | 0.4659 | 0.4742 | 0.5582 |
| YOLOv8n | 1.68860 | 0.4555 | 0.4889 | 0.5379 |

**训练耗时（50 轮）**：YOLOv5s ≈ 948 s，YOLOv8n ≈ 709 s，**YOLOv8 提速约 25%**。

### 7.2 同一图片的检测效果对比

对验证集图片 `SmokingData_Val_2171.jpg` 分别用两个模型推理：两者均能正确定位吸烟行为；**YOLOv5 置信度 0.7，YOLOv8 置信度 0.8**。

### 7.3 模型优缺点总结

| 模型 | 优点 | 缺点 |
| --- | --- | --- |
| YOLOv5 | 召回率较高，对多数目标能稳定检出 | 部分样本置信度偏低，偶尔漏检 |
| YOLOv8 | 精确率高，预测置信度普遍更高，训练速度更快 | 极少数复杂场景下召回率略低于 YOLOv5 |

## 8. 结果分析

1. **指标层面**：YOLOv5 的 mAP@0.5 与 Recall 分别高出约 1% 和 2%，说明其正样本召回能力稍强；YOLOv8 的 Precision 高出约 1.5%，虚警更少。两者的验证框损失非常接近，说明两代模型在本数据集上都已收敛到相近水平。
2. **效率层面**：YOLOv8 训练耗时减少约 25%，同时预测置信度更高（0.8 vs 0.7），工程上更友好。
3. **误差来源**：① `phone_call` 类别原始标注异常，模型未有效学习该类；② 部分损坏图片修复后引入噪声；③ 目标过小、光线过暗、遮挡等情况造成特征提取困难。
4. **结论**：本任务上两者性能接近、各有取舍——**更看重漏检率（安全类场景）选 YOLOv5；更看重虚警率与训练/推理效率选 YOLOv8**。若只需单模型部署，在本数据集条件下推荐 **YOLOv8n**（速度、精确率、置信度占优，且精度损失在 1% 量级）。

## 9. 不足与后续改进

- 数据集规模偏小（1559 张）且类别不平衡，`phone_call` 类几乎未习得 → 需清洗/重标该类别或补充数据；
- 仅对比了 s / n 两个最小规格模型，未覆盖 m / l 规模与更细的超参搜索；
- 未做数据增强策略的消融实验（Mosaic、MixUp 等）；
- 后续可补充：清洗数据集后重训、引入验证集置信度阈值调优、增加 YOLOv9 / RT-DETR 等新架构对照。

## 10. 声明

数据集来源于公开分享页面，版权归原作者所有；本仓库仅用于课程学习与实验对比，不含原始数据集的批量再分发。

---

**作者**：林启烨 · 人工智能专业 · 2025–2026 学年第二学期
