# 测试脚本

# 导入包
import torch
from MyModel import MyModel
from MyDataLoader import test_dataloader
from MyDataset import test_dataset
from Parameters import gpu, num_class
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import warnings
import csv
import os
warnings.filterwarnings("ignore")

# 加载模型
model = MyModel
model.load_state_dict(torch.load(r"models/best_model.pth"))
model.eval()

print("=" * 50)
print("开始测试...")
print("=" * 50)

all_preds = []
all_labels = []

with torch.no_grad():

    for batch_idx, (feature, true_label) in enumerate(test_dataloader):

        feature, true_label = feature.to(gpu), true_label.to(gpu)

        predict_label = model(feature)

        pred_class = predict_label.argmax(dim=1)

        all_preds.append(pred_class.cpu())
        all_labels.append(true_label.cpu())

# 合并所有 batch
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)

# 准确率
accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
print(f"\n测试集准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# 分类报告
class_names = test_dataset.classes
report = classification_report(all_labels, all_preds, target_names=class_names, digits=3, output_dict=True)
print("\n各类别指标:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(22, 20))
plt.imshow(cm, cmap="Blues", aspect="auto")
plt.colorbar(label="Sample Count")
plt.xlabel("Predicted", fontsize=14)
plt.ylabel("True", fontsize=14)
plt.title(f"Confusion Matrix (50 Classes, Acc={accuracy * 100:.1f}%)", fontsize=16)
plt.xticks(range(num_class), class_names, rotation=90, fontsize=7)
plt.yticks(range(num_class), class_names, fontsize=7)
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
print("\n混淆矩阵已保存至 models/confusion_matrix.png")

# 混淆矩阵导出 CSV
os.makedirs("Test_results", exist_ok=True)
with open("Test_results/Specific_prediction_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([""] + class_names)
    for i, row in enumerate(cm):
        writer.writerow([class_names[i]] + list(row))
print("混淆矩阵 CSV 已保存至 Test_results/Specific_prediction_results.csv")

# 分类报告导出 CSV
with open("Test_results/classification_report.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
    for name, metrics in report.items():
        if name == "accuracy":
            writer.writerow([name, "", "", f"{metrics:.3f}", ""])
        else:
            writer.writerow([name, f"{metrics['precision']:.3f}", f"{metrics['recall']:.3f}", f"{metrics['f1-score']:.3f}", metrics['support']])
print("分类报告 CSV 已保存至 Test_results/classification_report.csv")
