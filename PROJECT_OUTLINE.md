# 摄像头物品分类项目 — 完整大纲

---

## 一、项目概述

| 条目 | 说明 |
|------|------|
| **目标** | 训练一个图像分类模型，部署到边缘设备，通过摄像头实时识别物品（以水果为例） |
| **核心流程** | 摄像头采集 → 图像预处理 → 模型推理 → 输出分类结果 |
| **平台** | 训练在 Ubuntu PC（有 NVIDIA 显卡），推理部署在 ARM Linux 板子 |
| **预算** | 先 0 元本机跑通，后续板子预算 200-300 元 |

---

## 二、技术栈总览

| 层 | 技术 | 用途 |
|----|------|------|
| **训练框架** | PyTorch | 定义模型、训练、验证、保存权重 |
| **模型** | MobileNetV2（预训练权重） | 轻量级 CNN，适合边缘设备，水果分类准确率 95%+ |
| **模型导出** | ONNX | 把 PyTorch 模型转成跨平台通用格式 |
| **推理引擎** | ONNX Runtime | 在板子上加载 ONNX 模型跑推理，轻量、快速 |
| **图像采集** | OpenCV | 读取摄像头、图像缩放/裁剪/归一化、显示结果 |
| **数据来源** | Fruits 360 数据集 (Kaggle) | 131 种水果，90000+ 张图片，开箱即用 |
| **操作系统** | Ubuntu（训练机）/ Debian/Armbian（板子） | 全链路 Linux |

### 为什么选这些？

```
训练：PyTorch 生态最好，你想做 AI Infra 方向这是必修
模型：MobileNetV2 在精度和速度之间平衡最好，专为移动端设计
部署：ONNX 让模型脱离 PyTorch，推理快 1.5-2 倍，板子上只装几十 MB 的 ONNX Runtime
摄像头：OpenCV 是工业标准，但你只需要学 5 个函数
```

---

## 三、硬件方案

### 阶段 1：本机跑通（0 元，马上开始）

| 设备 | 状态 |
|------|------|
| 现有 Ubuntu 电脑（NVIDIA 显卡） | ✅ 已有 |
| USB 摄像头（720P） | 需购买，约 30-50 元，或用手边现有 |

**这步的目标**：把全链路跑通，确认效果满意。

### 阶段 2：边缘部署（跑通阶段 1 后再买）

| 设备 | 价格 | 说明 |
|------|------|------|
| **香橙派 Orange Pi 3B** | ~150-200 元 | 性价比之王 |
| 树莓派 4B 2GB | ~250 元 | 社区最成熟，备选 |
| USB 摄像头 720P | ~30-50 元 | 越便宜越好，不需要高清 |
| TF 卡 32GB+ | ~30 元 | 装系统和模型 |
| 散热片 + 小风扇 | ~15 元 | 足够用了 |
| **总计** | **200-300 元** | |

### 散热说明

- 项目推理负载不重（单次推理 0.1-0.3 秒），不是持续满载
- 十几块的散热片+小风扇套装就够了
- 即使不加散热，CPU 到 80°C 会自动降频保护，不会烧

---

## 四、各技术点如何理解

### 4.1 ONNX（Open Neural Network Exchange）

**一句话**：模型的中转格式，训练用 PyTorch，部署用 ONNX。

```
类比：
  你写了一本书（PyTorch 模型）
  → 翻译成英文版（导出 ONNX）
  → 在任何国家都能读（PC、手机、ARM 板子、浏览器）
  → 不需要把中文老师也带过去（不需要装 PyTorch）
```

**为什么用它：**

| 不用 ONNX | 用 ONNX |
|-----------|---------|
| 板子上装 PyTorch（几 GB） | 板子上装 ONNX Runtime（几十 MB） |
| PyTorch 推理参数多、占内存 | ONNX Runtime 轻量、启动快 |
| ARM 板装 PyTorch 容易踩坑 | ONNX Runtime 官方支持 ARM |

**你只需要学一个操作**：怎么导出 ONNX，代码就 5 行。

### 4.2 OpenCV 和 PyTorch 的关系

| | PyTorch | OpenCV |
|------|---------|--------|
| **是什么** | 深度学习框架 | 计算机视觉工具库 |
| **核心能力** | 训练神经网络、GPU 计算、自动求导 | 读图片、开摄像头、图像缩放/裁剪/显示 |
| **本项目负责** | 模型推理（判断「这是苹果」） | 所有模型之外的事（取画面、预处理、显示结果） |
| **你需要学的** | 迁移学习、训练、导出 | 5 个函数就够 |

**协作关系：**
```
摄像头 → OpenCV 抓画面 → 缩放/转 tensor → ONNX 推理 → OpenCV 写文字显示
        |                  |                |              |
     读取帧            预处理         模型分类        展示结果
```

### 4.3 MobileNetV2

- Google 2018 提出，专为移动/嵌入式设备设计
- 在 ImageNet 上预训练过，**直接从预训练权重迁移学习**，不需要从头训练
- 参数量 ~3.5M（ResNet18 是 11M），在 ARM 板子上推理快
- 水果分类准确率轻松 95%+

---

## 五、项目实施步骤（分 4 个阶段）

### 阶段 1：模型训练（在本机）

**目标**：训练一个能分类水果的模型

| 步骤 | 做什么 | 知识点 |
|------|--------|--------|
| 1.1 | 下载 Fruits 360 数据集 | Kaggle API / 手动下载 |
| 1.2 | 配置训练环境 | `pip install torch torchvision matplotlib tensorboard` |
| 1.3 | 写数据加载代码 | `ImageFolder`、数据增强、train/val split |
| 1.4 | 写模型定义 | 加载预训练 MobileNetV2、替换分类头 |
| 1.5 | 写训练循环 | train/val loop、保存最佳模型、loss 曲线 |
| 1.6 | 评估模型 | 准确率、混淆矩阵、单张图片测试 |

**预计耗时**：写完代码 1-2 天，训练几十分钟到 1-2 小时（取决于 GPU）

### 阶段 2：导出 ONNX + 本地推理验证（在本机）

**目标**：把模型导出 ONNX，在本机用 ONNX Runtime 跑推理

| 步骤 | 做什么 | 知识点 |
|------|--------|--------|
| 2.1 | 导出 ONNX | `torch.onnx.export()` |
| 2.2 | 装 ONNX Runtime | `pip install onnxruntime` |
| 2.3 | 加载 ONNX 模型 | 创建 session、准备输入 |
| 2.4 | 单张图片推理 | 对比 PyTorch 和 ONNX 结果是否一致 |
| 2.5 | 测速对比 | PyTorch vs ONNX Runtime 推理速度 |

### 阶段 3：摄像头实时分类（在本机）

**目标**：接上摄像头，实时识别画面中的物品

| 步骤 | 做什么 | 知识点 |
|------|--------|--------|
| 3.1 | 测试摄像头 | `cv2.VideoCapture(0)` 读画面 |
| 3.2 | 预处理管线 | 缩放、裁剪、归一化、转 tensor |
| 3.3 | 实时推理循环 | 读帧 → 推理 → 显示结果 |
| 3.4 | 显示优化 | 画框、写文字、显示置信度 |
| 3.5 | 交互（可选） | 按键截图保存、切换模型 |

### 阶段 4：搬到板子上（ARM Linux 部署）

**目标**：把整套东西搬到香橙派/树莓派上独立运行

| 步骤 | 做什么 | 知识点 |
|------|--------|--------|
| 4.1 | 烧录系统 | Armbian/Debian、SSH 连接 |
| 4.2 | 装依赖 | Python、ONNX Runtime（ARM 版）、OpenCV |
| 4.3 | 传模型和代码 | scp / U盘 |
| 4.4 | 跑起来 | 接摄像头、运行推理脚本 |
| 4.5 | 开机自启动 | systemd 服务 |

---

## 六、数据集：Fruits 360

| 属性 | 值 |
|------|-----|
| **来源** | [Kaggle - Fruits 360](https://www.kaggle.com/datasets/moltean/fruits) |
| **类别数** | 131 种水果 |
| **总图片数** | 90000+ 张 |
| **图片尺寸** | 100×100 像素 |
| **数据划分** | 已分好 Training / Test |
| **文件大小** | 约 1GB |

### 数据特点

- 白底拍摄，背景干净 → 训练难度低，适合入门
- 类别多 → 换分类头的全连接层时注意输出 131
- 图片尺寸小 → 需要 resize 到 224×224（MobileNetV2 的输入要求）

---

## 七、模型训练策略

### 迁移学习方案

```
MobileNetV2（ImageNet 预训练）
  ├── backbone：冻结，不要动（已经会认形状/纹理/边缘）
  └── 分类头：替换成 131 类，只训这层
```

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| 输入尺寸 | 224×224 | MobileNetV2 标准输入 |
| Batch Size | 32 | GPU 显存够用 |
| Learning Rate | 0.001 | Adam 优化器 |
| Epochs | 10-20 | 数据简单，收敛快 |
| 优化器 | Adam | 最省心 |
| 损失函数 | CrossEntropyLoss | 多分类标准 |

### 数据增强

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),   # 水平翻转
    transforms.RandomRotation(15),        # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 色彩扰动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 标准归一化
])
```

---

## 八、项目目录结构

```
Camera-recognition-project/
├── PROJECT_OUTLINE.md          # 本文件，项目大纲
│
├── data/                       # 数据集（gitignore 掉）
│   └── fruits-360/
│       ├── Training/
│       └── Test/
│
├── models/                     # 模型文件（gitignore，太大）
│   ├── best_model.pth          # PyTorch 权重
│   └── model.onnx              # ONNX 导出模型
│
├── src/
│   ├── config.py               # 所有配置（路径、超参数、类别名）
│   ├── dataset.py              # 数据加载
│   ├── model.py                # 模型定义
│   ├── train.py                # 训练脚本
│   ├── evaluate.py             # 评估脚本（准确率、混淆矩阵）
│   ├── export_onnx.py          # 导出 ONNX
│   ├── infer_single.py         # 单张图片推理测试
│   └── camera_infer.py         # 摄像头实时推理
│
├── deploy/                     # 板子部署用
│   ├── requirements.txt        # 板子依赖
│   ├── run.sh                  # 启动脚本
│   └── autostart.service       # 开机自启 systemd 配置
│
├── notebooks/                  # Jupyter notebook（实验用）
│   └── explore_data.ipynb      # 数据探索
│
├── requirements.txt            # 训练机依赖
└── README.md
```

---

## 九、依赖清单

### 训练机（你现在电脑）

```
torch>=2.0.0
torchvision>=0.15.0
onnx>=1.14.0
onnxruntime>=1.15.0
opencv-python>=4.8.0
matplotlib>=3.7.0
tensorboard>=2.13.0
numpy>=1.24.0
Pillow>=9.5.0
```

### 板子（ARM Linux）

```
onnxruntime         # ARM 版，装法不同
opencv-python-headless  # 无需 GUI 依赖
numpy
Pillow
```

### 安装命令（训练机）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # 用 CUDA 版
pip install onnx onnxruntime opencv-python matplotlib tensorboard
```

> **提醒**：用 pip 先换国内镜像源：
> ```bash
> pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

## 十、关键代码框架预览

### 导出 ONNX（5 行代码）

```python
import torch
from src.model import FruitClassifier

model = FruitClassifier(num_classes=131)
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)  # 构造一个假输入
torch.onnx.export(model, dummy_input, "models/model.onnx",
                  input_names=["input"], output_names=["output"])
```

### OpenCV 摄像头推理循环（核心骨架）

```python
import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("models/model.onnx")
class_names = [...]  # 131 个水果名

cap = cv2.VideoCapture(0)  # 打开摄像头

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理：缩放 + 归一化
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # 加 batch 维度

    # 推理
    outputs = session.run(None, {"input": img})
    pred_idx = np.argmax(outputs[0])
    confidence = np.max(outputs[0])

    # 显示结果
    label = f"{class_names[pred_idx]}: {confidence:.2%}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 十一、可能踩的坑 & 提前知道

| 坑 | 怎么回事 | 怎么解决 |
|----|----------|---------|
| 数据集类别不对 | Fruits 360 类别数可能随版本变化 | 读数据时先 `print(len(classes))`，模型输出和这个数对齐 |
| ONNX 导出 shape 报错 | pytorch 模型用了动态操作 | 导出时指定 `dynamic_axes` 或用固定输入尺寸 |
| 板子上装不了某些 pip 包 | ARM 架构没有预编译 wheel | 用 apt-get 装系统包，或用 piwheels 源 |
| 摄像头打不开 | /dev/video0 权限或设备冲突 | `sudo usermod -aG video $USER` 然后重启 |
| 推理结果和训练时不一样 | 预处理不一致（归一化参数、颜色通道顺序） | 训练和推理用同一套预处理函数 |
| 板子太烫 | 持续满载 + 不加散热 | 十几块钱买散热片+风扇 |

### 降温手段（如果板子烫手）

```
1. 贴散热片在 CPU 上                        → 降温 5-10°C
2. 接 5V 小风扇对着板子吹                     → 降温 10-15°C
3. 推理间隔加 time.sleep(0.1)，降低负载        → 降温 3-5°C
4. 不要放在封闭盒子里，保持通风                 → 基本的
```

---

## 十二、学习路径一目了然

```
你现在在这里 ↓

    [掌握 PyTorch 迁移学习]      ← 你最需要扎实掌握的
            ↓
    [学会 ONNX 导出和推理]       ← 就几行代码
            ↓
    [学会 OpenCV 5 个基本函数]   ← 半小时搞定
            ↓
    [本机跑通全流程]             ← 里程碑：0元达成
            ↓
    [买板子、烧系统、部署]       ← 搬上 ARM Linux
            ↓
    [完成！摄像头拍啥认啥]       ← 拿出去炫
```

---

## 十三、下一步行动

- [ ] 确认第一步：开始写训练代码（数据下载 + 模型训练）
- [ ] 购买 USB 摄像头（30 元级别即可）
- [ ] （跑通后）决定买不买板子

---

> 最后更新：2026-06-27
