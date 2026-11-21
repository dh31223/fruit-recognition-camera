# 水果识别相机 🍎📷

> 50 类果蔬实时识别 · ResNet18 + ONNX Runtime + 香橙派 4 Pro 边缘部署，91.4% 准确率 / 9 FPS

USB 摄像头对准水果，实时显示名称和置信度。50 类水果蔬菜，91.4% 准确率，不联网，不依赖 GPU 服务器，一个香橙派 4 Pro 搞定一切。

---

## 效果

```
┌──────────────────────────────────────────┐
│  Apple         89.3%   ← 绿色（高置信度）  │
│  Banana         3.2%                      │
│  Orange         2.1%                      │
│  Pear           1.8%                      │
│  Peach          0.9%                      │
│                          FPS: 9           │
└──────────────────────────────────────────┘
```

## 技术栈

| 环节 | 技术 |
|------|------|
| 模型 | ResNet18（PyTorch） |
| 数据集 | Kaggle Fruits & Vegetables (50类) |
| 训练策略 | 两阶段：冻结 backbone → 全模型微调 |
| 推理加速 | ONNX Runtime |
| 部署 | 香橙派 4 Pro (Armbian) |
| 摄像头 | OpenCV |

## 准确率

- **验证集**：93.0%（Stage 2 微调后）
- **测试集**：91.4%，平均 F1 = 0.915

## 项目结构

```
src/
├── MyModel.py          # 模型定义 (ResNet18)
├── MyTrans.py          # 数据增强
├── MyDataset.py        # 数据集加载
├── MyDataLoader.py     # DataLoader 配置
├── Parameters.py       # 超参数
├── Train.py            # 训练脚本（两阶段）
├── Test.py             # 测试脚本
├── Predict.py          # 摄像头实时推理
├── Export.py           # ONNX 导出
├── OnnxVerify.py       # ONNX 验证 + 速度对比
└── Convert.py          # (占位)
Test_results/
├── classification_report.csv
└── Specific_prediction_results.csv
```

## 快速开始

### 训练

```bash
pip install torch torchvision
python src/Train.py
```

### 摄像头实时推理 (PC)

```bash
pip install opencv-python pillow
python src/Predict.py
```

### 部署到香橙派 4 Pro

```bash
# 板子上
pip install onnxruntime opencv-python pillow numpy

# PC 上传模型
scp models/model.onnx src/Predict.py orangepi@<IP>:~/

# 运行
python Predict.py
```

## 硬件清单

| 设备 | 价格 |
|------|------|
| 香橙派 4 Pro (4GB) | ¥541 |
| TF 卡 (32GB) | ¥41 |
| HDMI 线 | ¥8 |
| Type-C 电源 | ¥27 |
| 散热片 | ¥3 |
| USB 摄像头 | ¥20 |
| **总计** | **¥640** |

## 踩过的坑

- **NPU**：香橙派 4 Pro 的 vivante NPU 驱动与用户态库不兼容，最终放弃，ONNX CPU 推理 9 FPS 也够用
- **BGR→RGB**：OpenCV 读进来是 BGR，模型吃 RGB，忘了转换识别率暴跌
- **颜色增强**：hue 设太大（如 0.5）会导致苹果和梨无法区分
- **真实场景**：梨常被误认为苹果，倒置水果识别差——训练数据没有倒着的样本

## 相关链接

- 📝 知乎文章：[大二，预算 ¥650，从零手搓水果识别相机](https://zhuanlan.zhihu.com/p/2059697618398270453)

## License

MIT
