


# 训练Python脚本




# 导入包
import warnings
warnings.filterwarnings("ignore")

from MyModel import MyModel
from MyDataLoader import train_dataloader, val_dataloader
from Parameters import first_step_epochs, first_step_learn_rate, second_step_epochs, second_step_learn_rate, weight_decay, gpu, first_step_patience, second_step_patience
import time
import torch.nn as nn
import torch
import os
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import matplotlib.pyplot as plt





if __name__ == '__main__':

    model = MyModel
    loss_func = nn.CrossEntropyLoss()

    # 第一阶段训练，只训练分类头，防止随机初始化的分类头通过反向传播污染整体预训练参数。

    # 先冻结所有参数，然后只解冻fc参数
    for pa in model.parameters():
        pa.requires_grad = False

    for pa in model.fc.parameters():
        pa.requires_grad = True

    optimizer = torch.optim.Adam(
        params = model.parameters(), 
        lr = first_step_learn_rate, 
        weight_decay = weight_decay
    )

    scheduler = StepLR(
        optimizer = optimizer, 
        step_size = 5, 
        gamma = 0.5
    )

    # 开始第一阶段训练
    
    first_step_best_loss = float("inf")
    this_time_avg_loss = 0
    unImprove = 0
    first_step_train_loss_num = []
    first_step_val_loss_num = []
    print("="*26 + "第一阶段训练" + "=" * 27)
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Test Loss':<12} {'Accuracy':<10} {'耗时(秒)':<10} {'总计':<10}")
    print("-" * 65)
    all_time = 0
    for epoch in range(first_step_epochs):

        # 开始第epoch + 1次训练
        model.train()
        train_loss_sum = 0
        start_time = time.time()
        for feature, true_label in train_dataloader:

            feature, true_label = feature.to(gpu), true_label.to(gpu)

            predict_label = model(feature)

            loss = loss_func(predict_label, true_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            first_step_train_loss_num.append(loss.item())
            train_loss_sum += loss.item()
        
        # 开始第epoch + 1次测试

        with torch.no_grad():
            
            model.eval()
            val_loss_sum = 0
            correct_count = 0

            for feature, true_label in val_dataloader:

                feature, true_label = feature.to(gpu), true_label.to(gpu)

                predict_label = model(feature)

                loss = loss_func(predict_label, true_label)

                first_step_val_loss_num.append(loss.item())
                val_loss_sum += loss.item()

                correct_count += (predict_label.argmax(dim = 1) == true_label).sum().item()

        # 第epoch + 1轮训练和测试结束，反馈结果
        end_time = time.time()
        all_time += (end_time - start_time)
        print(f"{epoch + 1:<6} {train_loss_sum / len(train_dataloader):<12.6f} {val_loss_sum / len(val_dataloader):<12.6f} {correct_count / len(val_dataloader.dataset):<10.4f} {int(end_time - start_time):<10} {int(all_time // 60)}分{int(all_time % 60)}秒")

        this_time_avg_loss = val_loss_sum / len(val_dataloader)

        if first_step_best_loss > this_time_avg_loss:

            first_step_best_loss = this_time_avg_loss
            os.makedirs("models", exist_ok = True)
            torch.save(model.state_dict(), r"models/first_best_model.pth")
            unImprove = 0

        else:
            unImprove += 1
            if unImprove == first_step_patience:
                print(f"早停于{epoch + 1}, 正确率为{correct_count / len(val_dataloader.dataset)}")
                print()
                break
        
        scheduler.step()
        
        

    print()
    print()
    print()

    # 开始第二阶段训练

    model.load_state_dict(torch.load(r'models/first_best_model.pth'))

    # 解锁全部参数
    for pa in model.parameters():
        pa.requires_grad = True

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = second_step_learn_rate, 
        weight_decay = weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer = optimizer, 
        mode = "min", 
        factor = 0.5, 
        patience = 2
    )

    second_step_best_loss = float("inf")
    this_time_avg_loss = 0
    unImprove = 0
    second_step_train_loss_num = []
    second_step_val_loss_num = []
    print("="*26 + "第二阶段训练" + "=" * 27)
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Test Loss':<12} {'Accuracy':<10} {'耗时(秒)':<10} {'总计':<10}")
    print("-" * 65)

    for epoch in range(second_step_epochs):

        # 开始第 epoch + 1 次训练
        model.train()
        train_loss_sum = 0
        start_time = time.time()

        for feature, true_label in train_dataloader:

            feature, true_label = feature.to(gpu), true_label.to(gpu)

            predict_label = model(feature)

            loss = loss_func(predict_label, true_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            second_step_train_loss_num.append(loss.item())
            train_loss_sum += loss.item()

        # 开始第 epoch + 1 次测试
        with torch.no_grad():

            model.eval()
            val_loss_sum = 0
            correct_count = 0

            for feature, true_label in val_dataloader:

                feature, true_label = feature.to(gpu), true_label.to(gpu)

                predict_label = model(feature)

                loss = loss_func(predict_label, true_label)

                second_step_val_loss_num.append(loss.item())
                val_loss_sum += loss.item()

                correct_count += (predict_label.argmax(dim = 1) == true_label).sum().item()

        # 第epoch + 1轮训练和测试结束，反馈结果
        end_time = time.time()
        all_time += (end_time - start_time)
        print(f"{epoch + 1:<6} {train_loss_sum / len(train_dataloader):<12.6f} {val_loss_sum / len(val_dataloader):<12.6f} {correct_count / len(val_dataloader.dataset):<10.4f} {int(end_time - start_time):<10} {int(all_time // 60)}分{int(all_time % 60)}秒")
                
        # 早停
        this_time_avg_loss = val_loss_sum / len(val_dataloader)

        scheduler.step(this_time_avg_loss)

        if second_step_best_loss > this_time_avg_loss:
            second_step_best_loss = this_time_avg_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), r"models/best_model.pth")
            unImprove = 0
        else:
            unImprove += 1
            if unImprove == second_step_patience:
                print(f"早停于{epoch + 1}, 正确率为{correct_count / len(val_dataloader.dataset)}")
                break


    # 可视化训练结果

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(first_step_train_loss_num, alpha=0.6, linewidth=0.5)
    plt.title("Stage 1 — Train Loss (per batch)")
    plt.xlabel("Batch")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 2)
    plt.plot(first_step_val_loss_num, alpha=0.6, linewidth=0.5)
    plt.title("Stage 1 — Val Loss (per batch)")
    plt.xlabel("Batch")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 3)
    plt.plot(second_step_train_loss_num, alpha=0.6, linewidth=0.5)
    plt.title("Stage 2 — Train Loss (per batch)")
    plt.xlabel("Batch")
    plt.ylabel("Loss")

    plt.subplot(2, 2, 4)
    plt.plot(second_step_val_loss_num, alpha=0.6, linewidth=0.5)
    plt.title("Stage 2 — Val Loss (per batch)")
    plt.xlabel("Batch")
    plt.ylabel("Loss")

    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/loss_curves.png", dpi=150)
    plt.show()
    print("loss 曲线已保存至 models/loss_curves.png")

print("END!")


