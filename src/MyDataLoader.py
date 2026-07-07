

# DataLoader 文件

# 导入包

from torch.utils.data import DataLoader
from MyDataset import  train_dataset, val_dataset, test_dataset
from Parameters import train_batch_size, train_num_workers, train_prefetch_factor, val_batch_size, val_num_workers, val_prefetch_factor, test_batch_size, test_num_workers, test_prefetch_factor




train_dataloader = DataLoader(
    dataset = train_dataset, 
    batch_size = train_batch_size, 
    shuffle = True, 
    num_workers = train_num_workers, 
    pin_memory = True, 
    persistent_workers = True, 
    prefetch_factor = train_prefetch_factor
)


val_dataloader = DataLoader(
    dataset = val_dataset, 
    batch_size = val_batch_size, 
    shuffle = False, 
    num_workers = val_num_workers, 
    pin_memory = True, 
    persistent_workers = True, 
    prefetch_factor = val_prefetch_factor
)

test_dataloader = DataLoader(
    dataset = test_dataset, 
    batch_size = test_batch_size, 
    shuffle = False, 
    num_workers = test_num_workers, 
    pin_memory = True, 
    persistent_workers = True, 
    prefetch_factor = test_prefetch_factor

)

