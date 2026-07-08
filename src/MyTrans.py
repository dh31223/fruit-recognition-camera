

# Transforms 文件

# 导入包

import torchvision.transforms.v2 as v2
import torch

train_transforms = v2.Compose([
    v2.Resize(256),            
    v2.CenterCrop(224),        
    v2.RandomHorizontalFlip(0.5), 
    v2.RandomRotation(15), 
    v2.ColorJitter(
        brightness = 0.2, 
        contrast = 0.2, 
        saturation = 0.2, 
        hue = 0.05
    ), 
    v2.ToImage(),
    v2.ToDtype(dtype = torch.float32, scale = True), 
    v2.Normalize(
        mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225]
    )
])



val_transforms = v2.Compose([
    v2.Resize(256),            
    v2.CenterCrop(224), 
    v2.ToImage(),
    v2.ToDtype(dtype = torch.float32, scale = True), 
    v2.Normalize(
        mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225]
    )
])


test_transforms = v2.Compose([
    v2.Resize(256),            
    v2.CenterCrop(224), 
    v2.ToImage(),
    v2.ToDtype(dtype = torch.float32, scale = True), 
    v2.Normalize(
        mean = [0.485, 0.456, 0.406], 
        std = [0.229, 0.224, 0.225]
    )
])







