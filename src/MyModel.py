

# Model 文件

# 导入包

from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from Parameters import num_class, gpu

# 模型

MyModel = resnet18(weights = ResNet18_Weights.DEFAULT)
MyModel.fc = nn.Linear(in_features = MyModel.fc.in_features, out_features = num_class)
MyModel.to(gpu)



