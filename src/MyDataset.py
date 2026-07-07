

# Dataset 文件

# 导入包


from torchvision.datasets import ImageFolder
from MyTrans import train_transforms, val_transforms, test_transforms

train_dataset = ImageFolder(root = r"data/archive/train", transform = train_transforms)

val_dataset   = ImageFolder(root = r"data/archive/validation", transform = val_transforms)

test_dataset  = ImageFolder(root = r"data/archive/test", transform = test_transforms)











