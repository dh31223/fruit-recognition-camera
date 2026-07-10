
import torch
from MyModel import MyModel



# 1. 加载模型（移到 CPU，导出 ONNX 不需要 GPU）
model = MyModel
model.load_state_dict(torch.load("models/best_model.pth", map_location="cpu"))
model.to("cpu")
model.eval()

# 2. 造一个假图片（CPU 上，batch_size=1, 3通道, 224×224）
dummy = torch.randn(1, 3, 224, 224)

# 3. 导出
torch.onnx.export(model,
                  dummy, "models/model.onnx",
                  input_names=["input"],
                  output_names=["output"]
                )

print("导出成功！")
