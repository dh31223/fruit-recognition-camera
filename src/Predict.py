import cv2
from PIL import Image
from MyTrans import val_transforms
from Parameters import gpu
from MyModel import MyModel
import torch
from MyDataset import test_dataset
import time

cap = cv2.VideoCapture(2)

model = MyModel
model.load_state_dict(torch.load(r"models/best_model.pth"))
model.eval()

class_names = test_dataset.classes

# 预热摄像头（前几帧曝光不准）
for _ in range(10):
    cap.read()

print("模型已加载，摄像头已就绪")

# FPS 统计
fps_history = [0.0] * 10
fps_idx = 0


def get_confidence_color(conf):
    """置信度 → 颜色：>80% 绿，50-80% 黄，<50% 红"""
    if conf > 0.8:
        return (0, 255, 0)      # 绿
    elif conf > 0.5:
        return (0, 255, 255)    # 黄
    else:
        return (0, 0, 255)      # 红


while True:
    loop_start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = val_transforms(pil).unsqueeze(0)
    tensor = tensor.to(gpu)

    predict = model(tensor)
    probs = torch.softmax(predict, dim=1)
    top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)

    # ===== 左上角：Top-5 预测 =====
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (260, 165), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # Top 1 — 大字，置信度配色
    top1_name = class_names[top5_indices[0, 0].item()]
    top1_conf = top5_probs[0, 0].item()
    top1_color = get_confidence_color(top1_conf)
    cv2.putText(frame, f"{top1_name}  {top1_conf:.1%}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, top1_color, 2)

    # Top 2~5 — 小字，各自的置信度配色
    for i in range(1, 5):
        name = class_names[top5_indices[0, i].item()]
        conf = top5_probs[0, i].item()
        color = get_confidence_color(conf)
        y = 32 + i * 28
        cv2.putText(frame, f"{name}  {conf:.1%}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # ===== 右上角：FPS =====
    frame_h, frame_w = frame.shape[:2]
    fps = 1.0 / (time.time() - loop_start) if (time.time() - loop_start) > 0 else 0
    fps_history[fps_idx % 10] = fps
    fps_idx += 1
    avg_fps = sum(fps_history) / len(fps_history)

    fps_text = f"FPS: {avg_fps:.0f}"
    (text_w, text_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    fps_x = frame_w - text_w - 15
    fps_y = 30

    # 半透明背景
    overlay2 = frame.copy()
    cv2.rectangle(overlay2,
                  (fps_x - 8, fps_y - text_h - 5),
                  (fps_x + text_w + 8, fps_y + 5),
                  (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0)

    cv2.putText(frame, fps_text, (fps_x, fps_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
        break





