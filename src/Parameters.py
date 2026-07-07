

# Parameters 文件

# 导入包
import torch


# DataLoader Para
train_batch_size = 64
train_num_workers = 8
train_prefetch_factor = 4

val_batch_size = 64
val_num_workers = 2
val_prefetch_factor = 2

test_batch_size = 64
test_num_workers = 2
test_prefetch_factor = 2



# Data Para

num_class = 50


# Device Para

gpu = "cuda" if torch.cuda.is_available() else "cpu"


# Train Para


first_step_epochs = 20
first_step_learn_rate = 1e-3
first_step_patience = 3

second_step_epochs = 50
second_step_learn_rate = 1e-4
second_step_patience = 5

weight_decay = 1e-4


