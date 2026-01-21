# =============================================================================
# NEU FER - Commercial Standard (VGGFace2 + 5-Fold + Checkpoint)
# Dataset: fer_data_aligned
# Fixes: Updated Albumentations API, Auto-Resume
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
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from facenet_pytorch import InceptionResnetV1
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm 

# Windows Fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ===========================
# 1. 配置参数
# ===========================
CONFIG = {
    'seed': 2025,
    'img_size': 224,       
    'batch_size': 16,      
    'epochs': 15,          
    'lr': 1e-4,            
    'num_classes': 6,
    'n_folds': 5,          
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'num_workers': 0,
    
    # 数据集路径
    'base_dir': r"C:\Users\shmil\Desktop\fer_data_aligned",
    
    # 断点保存路径
    'ckpt_path': r"C:\Users\shmil\Desktop\vggface_5fold_checkpoint.pth"
}

# ===========================
# 2. 类别映射
# ===========================
LABEL_MAP = {
    'Anger': 0, 'Fear': 1, 'Happy': 2, 'Sad': 3, 'Surprise': 4, 'Neutral': 5
}

# ===========================
# 3. 基础组件
# ===========================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_dirs(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    if not os.path.exists(train_dir): train_dir = os.path.join(base_dir, 'Train')
    test_dir = os.path.join(base_dir, 'test')
    if not os.path.exists(test_dir): test_dir = os.path.join(base_dir, 'Test')
    return train_dir, test_dir

class EmotionDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, is_test=False):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        if self.is_test:
            img_path = os.path.join(self.root_dir, row['ID'])
        else:
            img_path = os.path.join(self.root_dir, row['label_name'], row['ID'])
            
        image = cv2.imread(img_path)
        if image is None: image = np.zeros((CONFIG['img_size'], CONFIG['img_size'], 3), dtype=np.uint8)
        else: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        if self.is_test:
            return image
        else:
            return image, torch.tensor(row['label_idx'], dtype=torch.long)

def get_transforms(data='train'):
    if data == 'train':
        return A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            
            # --- 修复部分：使用新版 API 参数 ---
            A.CoarseDropout(
                num_holes_range=(1, 4), 
                hole_height_range=(10, 20), 
                hole_width_range=(10, 20), 
                p=0.2
            ),
            # --------------------------------
            
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])

class FaceModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 第一次运行时，这里会自动下载权重
        self.model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

# ===========================
# 4. 主程序
# ===========================
def main():
    print(f"🚀 启动 VGGFace2 (5-Fold + Checkpoint) | 设备: {CONFIG['device']}")
    seed_everything(CONFIG['seed'])
    
    try:
        TRAIN_DIR, TEST_DIR = get_dirs(CONFIG['base_dir'])
    except Exception as e:
        print(e); return

    # 扫描数据
    print(">>> 扫描 Aligned 数据...")
    train_data = []
    valid_folders = list(LABEL_MAP.keys())
    try: actual_folders = os.listdir(TRAIN_DIR)
    except: return

    for folder in actual_folders:
        key = None
        if folder in valid_folders: key = folder
        elif folder.capitalize() in valid_folders: key = folder.capitalize()
            
        if key:
            class_dir = os.path.join(TRAIN_DIR, folder)
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    train_data.append({'ID': f, 'label_name': folder, 'label_idx': LABEL_MAP[key]})

    df_all = pd.DataFrame(train_data)
    print(f"✅ 总样本: {len(df_all)}")

    # 5-Fold
    skf = StratifiedKFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed'])
    
    # 状态变量初始化
    start_fold = 0
    start_epoch = 0
    fold_paths = [] # 存储已经跑完的 fold 的最佳模型路径
    
    # --- 尝试加载断点 ---
    if os.path.exists(CONFIG['ckpt_path']):
        print(f"🔄 发现中断点: {CONFIG['ckpt_path']}")
        try:
            ckpt = torch.load(CONFIG['ckpt_path'])
            start_fold = ckpt['fold']
            start_epoch = ckpt['epoch'] + 1  # 从下一轮开始
            fold_paths = ckpt.get('fold_paths', [])
            
            # 如果上一折已经跑满了所有 epoch，则进入下一折
            if start_epoch >= CONFIG['epochs']:
                start_fold += 1
                start_epoch = 0
                
            print(f"✅ 成功恢复! 将从 Fold {start_fold+1}, Epoch {start_epoch+1} 继续")
        except Exception as e:
            print(f"⚠️ 断点文件损坏，将从头开始: {e}")
            start_fold = 0
            start_epoch = 0
    else:
        print(">>> 从头开始训练")

    # ==========================
    # 训练循环
    # ==========================
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_all, df_all['label_idx'])):
        
        # 跳过已经跑完的 Fold
        if fold < start_fold:
            print(f"⏩ 跳过 Fold {fold+1} (已完成)")
            continue
            
        print(f"\n============================")
        print(f"🔄 Fold {fold+1}/{CONFIG['n_folds']}")
        print(f"============================")
        
        # 准备数据
        train_df = df_all.iloc[train_idx].reset_index(drop=True)
        valid_df = df_all.iloc[val_idx].reset_index(drop=True)
        
        train_ds = EmotionDataset(train_df, TRAIN_DIR, transform=get_transforms('train'))
        valid_ds = EmotionDataset(valid_df, TRAIN_DIR, transform=get_transforms('valid'))
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
        valid_loader = DataLoader(valid_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
        
        # 初始化模型
        model = FaceModel(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)
        scaler = GradScaler()
        
        best_acc = 0.0
        save_path = f"best_fold{fold}.pth"
        
        # 如果是当前中断的 Fold，加载之前的状态
        if fold == start_fold and os.path.exists(CONFIG['ckpt_path']):
            try:
                ckpt = torch.load(CONFIG['ckpt_path'])
                if ckpt['fold'] == fold:
                    model.load_state_dict(ckpt['model_state'])
                    optimizer.load_state_dict(ckpt['optimizer_state'])
                    best_acc = ckpt.get('best_acc', 0.0)
                    print(f"   -> 恢复模型参数，当前 Fold 最佳 Acc: {best_acc:.4f}")
            except:
                pass # 如果加载失败，就重新开始这一折
        else:
            # 新的 Fold，start_epoch 归零
            start_epoch = 0 

        # Epoch 循环
        for epoch in range(start_epoch, CONFIG['epochs']):
            model.train()
            loop = tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1}", leave=False)
            
            for images, labels in loop:
                images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                
                optimizer.zero_grad()
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loop.set_postfix(loss=loss.item())
            
            scheduler.step()
            
            # 验证
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
                    outputs = model(images)
                    val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            acc = accuracy_score(val_labels, val_preds)
            print(f"  > Ep {epoch+1} Val Acc: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)
                print(f"  >>> ⭐ Fold {fold+1} Best: {best_acc:.4f}")
            
            # 🔴 保存断点
            checkpoint = {
                'fold': fold,
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_acc': best_acc,
                'fold_paths': fold_paths + [save_path]
            }
            torch.save(checkpoint, CONFIG['ckpt_path'])
        
        # Fold 结束
        fold_paths.append(save_path)
        checkpoint['fold_paths'] = fold_paths
        torch.save(checkpoint, CONFIG['ckpt_path'])
        
        print(f"✅ Fold {fold+1} 完成.")
        start_epoch = 0
        del model, optimizer, scaler
        torch.cuda.empty_cache()

    # ==========================
    # 集成预测
    # ==========================
    print("\n🚀 开始 5-Model 集成预测...")
    test_files = sorted(os.listdir(TEST_DIR))
    test_df = pd.DataFrame({'ID': test_files})
    
    test_ds = EmotionDataset(test_df, TEST_DIR, transform=get_transforms('valid'), is_test=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    ensemble_probs = np.zeros((len(test_df), CONFIG['num_classes']), dtype=np.float32)
    
    # 读取模型路径
    final_paths = fold_paths
    if len(final_paths) == 0 and os.path.exists(CONFIG['ckpt_path']):
        ckpt = torch.load(CONFIG['ckpt_path'])
        final_paths = ckpt.get('fold_paths', [])
    
    final_paths = sorted(list(set([p for p in final_paths if os.path.exists(p)])))
    print(f"可用模型文件: {final_paths}")
    
    if len(final_paths) == 0:
        print("❌ 没有找到训练好的模型文件！")
        return

    for path in final_paths:
        print(f"Loading: {path}")
        model = FaceModel(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
        model.load_state_dict(torch.load(path))
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for images in tqdm(test_loader, desc="Predicting"):
                images = images.to(CONFIG['device'])
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                fold_preds.append(probs)
        
        ensemble_probs += np.concatenate(fold_preds, axis=0)
        del model
        torch.cuda.empty_cache()
    
    # 取平均
    ensemble_probs /= len(final_paths)
    final_labels = np.argmax(ensemble_probs, axis=1)

    # 提交
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'Emotion': final_labels
    })
    output_path = os.path.join(r"C:\Users\shmil\Desktop", 'submission_fer_vggface_5fold.csv')
    submission.to_csv(output_path, index=False)
    print(f"\n✅ 任务完成！文件: {output_path}")

if __name__ == '__main__':
    main()