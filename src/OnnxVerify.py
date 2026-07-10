# ONNX 推理验证 + 速度对比

import torch
import onnxruntime as ort
import numpy as np
import time
from MyModel import MyModel
from MyDataset import test_dataset
from Parameters import gpu

print("=" * 50)
print("ONNX 模型验证")
print("=" * 50)

# 1. 加载 PyTorch 模型（CPU）
print("\n加载 PyTorch 模型...")
pytorch_model = MyModel
pytorch_model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
pytorch_model.to("cpu")
pytorch_model.eval()

# 2. 加载 ONNX 模型
print("加载 ONNX 模型...")
ort_session = ort.InferenceSession("models/model.onnx",
                                   providers=['CPUExecutionProvider'])

# 3. 取一张测试图片
img_tensor, label = test_dataset[0]
img_tensor = img_tensor.unsqueeze(0)  # (1, 3, 224, 224)

true_class = test_dataset.classes[label]
print(f"\n测试图片: {true_class}")

# 4. PyTorch 推理
with torch.no_grad():
    pytorch_output = pytorch_model(img_tensor)
    pytorch_pred = pytorch_output.argmax(dim=1).item()
    pytorch_prob = torch.softmax(pytorch_output, dim=1)

# 5. ONNX 推理
ort_input = img_tensor.numpy()
ort_output = ort_session.run(["output"], {"input": ort_input})[0]
onnx_pred = ort_output.argmax(axis=1)[0]

# softmax
exp = np.exp(ort_output - ort_output.max(axis=1, keepdims=True))
onnx_prob = exp / exp.sum(axis=1, keepdims=True)

# 6. 对比
print(f"\nPyTorch 预测: {test_dataset.classes[pytorch_pred]} ({pytorch_prob[0, pytorch_pred].item():.2%})")
print(f"ONNX   预测: {test_dataset.classes[onnx_pred]} ({onnx_prob[0, onnx_pred]:.2%})")
print(f"\nTop-1 一致: {'✅' if pytorch_pred == onnx_pred else '❌'}")

diff = np.abs(pytorch_output.numpy() - ort_output)
print(f"输出最大误差: {diff.max():.6f}")

# 7. 速度对比
print("\n" + "=" * 50)
print("速度对比（CPU，100 次取平均）")
print("=" * 50)

runs = 100

# 预热
for _ in range(5):
    with torch.no_grad():
        pytorch_model(img_tensor)
    ort_session.run(["output"], {"input": ort_input})

# PyTorch
torch_start = time.time()
for _ in range(runs):
    with torch.no_grad():
        pytorch_model(img_tensor)
torch_ms = (time.time() - torch_start) / runs * 1000

# ONNX
onnx_start = time.time()
for _ in range(runs):
    ort_session.run(["output"], {"input": ort_input})
onnx_ms = (time.time() - onnx_start) / runs * 1000

print(f"\nPyTorch (CPU): {torch_ms:.2f} ms  →  {1000 / torch_ms:.1f} FPS")
print(f"ONNX    (CPU): {onnx_ms:.2f} ms  →  {1000 / onnx_ms:.1f} FPS")
print(f"\nONNX 比 PyTorch 快 {torch_ms / onnx_ms:.1f} 倍")
