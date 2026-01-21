# =============================================================================
# NEU Plant Seedling Classification Task 2 - Deep Learning Pipeline
# Model: EfficientNet-B4 (Pretrained)
# Framework: PyTorch
# =============================================================================

import os
import cv2
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler # 混合精度训练，加速
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import timm # PyTorch Image Models (必须安装)
import albumentations as A # 强大的数据增强库
from albumentations.pytorch import ToTensorV2
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore')

# ===========================
# 1. 配置参数
# ===========================
CONFIG = {
    'seed': 2025,
    'img_size': 380,   # EfficientNetB4 建议 380, B3 建议 300
    'batch_size': 16,  # 显存不够就改小
    'epochs': 15,      # 训练轮数
    'lr': 1e-4,        # 学习率
    'model_name': 'tf_efficientnet_b4_ns', # 使用 Noisy Student 预训练权重的 B4
    'num_classes': 12,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # 你的数据路径 (Task 2 的数据集路径可能变了，请检查！)
    'base_dir': "C:\Users\shmil\Desktop\dataset-for-task2"
}

# 锁定随机种子
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['seed'])

# ===========================
# 2. 路径处理
# ===========================
# 自动寻找 train/test 文件夹
def find_folder(base, name_options):
    if not os.path.exists(base):
        # 容错：如果 base 不对，尝试上一级
        base = os.path.dirname(base)
    
    available = os.listdir(base)
    for opt in name_options:
        if opt in available:
            return os.path.join(base, opt)
    # 如果还没找到，可能是直接在 dataset 目录下
    if 'train' in name_options: # 简单的回退策略
        return base 
    return base

TRAIN_DIR = find_folder(CONFIG['base_dir'], ['train', 'Train'])
TEST_DIR = find_folder(CONFIG['base_dir'], ['test', 'Test'])

print(f"Train Dir: {TRAIN_DIR}")
print(f"Test Dir: {TEST_DIR}")

# ===========================
# 3. 数据集定义 (Dataset)
# ===========================
class PlantDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, is_test=False):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 拼接图片路径
        if self.is_test:
            # 测试集通常是直接在文件夹里
            img_path = os.path.join(self.root_dir, row['file'])
        else:
            # 训练集通常是 类别/图片名
            img_path = os.path.join(self.root_dir, row['species'], row['file'])
            
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(img_path)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        if self.is_test:
            return image
        else:
            label = row['label_idx']
            return image, torch.tensor(label, dtype=torch.long)

# ===========================
# 4. 数据增强 (Augmentation)
# ===========================
def get_transforms(data='train'):
    if data == 'train':
        return A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.7), # 植物是旋转不变的，可以大力旋转
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2),
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.2), # 模拟遮挡
            A.Normalize(
                mean=[0.485, 0.456, 0.406], # ImageNet 标准均值
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

# ===========================
# 5. 准备数据
# ===========================
train_data = []
for label in os.listdir(TRAIN_DIR):
    class_dir = os.path.join(TRAIN_DIR, label)
    if not os.path.isdir(class_dir): continue
    for f in os.listdir(class_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            train_data.append([f, label])

df_train = pd.DataFrame(train_data, columns=['file', 'species'])

# 编码标签
le = LabelEncoder()
df_train['label_idx'] = le.fit_transform(df_train['species'])
CONFIG['num_classes'] = len(le.classes_)

# 划分训练/验证集 (9:1)
train_df, valid_df = train_test_split(df_train, test_size=0.1, stratify=df_train['label_idx'], random_state=CONFIG['seed'])

# DataLoader
train_dataset = PlantDataset(train_df, TRAIN_DIR, transform=get_transforms('train'))
valid_dataset = PlantDataset(valid_df, TRAIN_DIR, transform=get_transforms('valid'))

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

# ===========================
# 6. 定义模型 (EfficientNet)
# ===========================
class PlantModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        # 加载预训练模型
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        # 修改最后的全连接层
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

model = PlantModel(CONFIG['model_name'], CONFIG['num_classes']).to(CONFIG['device'])

# ===========================
# 7. 训练循环 (Training Loop)
# ===========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
# 余弦退火调度器
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
scaler = GradScaler() # 混合精度

best_acc = 0.0

print(f"开始训练... Device: {CONFIG['device']}")

for epoch in range(CONFIG['epochs']):
    model.train()
    train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
    
    for images, labels in pbar:
        images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
        
        optimizer.zero_grad()
        
        with autocast(): # 混合精度前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    scheduler.step()
    
    # 验证
    model.eval()
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"  >>> 模型已保存 (Best Acc: {best_acc:.4f})")

# ===========================
# 8. 预测与 TTA (测试集)
# ===========================
print("开始预测测试集...")
# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

test_files = sorted(os.listdir(TEST_DIR))
test_df = pd.DataFrame({'file': test_files})
test_dataset = PlantDataset(test_df, TEST_DIR, transform=get_transforms('valid'), is_test=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

# 普通预测 (无 TTA)
# 如果想要更高分，可以实现类似 Task 1 的 5-View TTA，这里为了代码简洁先做单次预测
final_preds = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Predicting"):
        images = images.to(CONFIG['device'])
        outputs = model(images)
        # 如果需要 TTA，可以在这里对 images 做 flip 再预测取平均
        preds = torch.argmax(outputs, dim=1)
        final_preds.extend(preds.cpu().numpy())

# 生成提交
test_df['species'] = le.inverse_transform(final_preds)
test_df.to_csv('submission_dl_task2.csv', index=False)
print("任务完成！提交文件: submission_dl_task2.csv")