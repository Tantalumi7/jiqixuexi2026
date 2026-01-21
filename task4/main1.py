import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.model_selection import KFold
import segmentation_models_pytorch as smp

# ==========================================
# 1. 配置参数 (CONFIG) - 最终冲刺版
# ==========================================
CONFIG = {
    'seed': 2024,
    'img_size': 512,           # 分辨率 640 (高精度关键)
    'batch_size': 4,           # 你的最新设定
    'learning_rate': 1e-4,
    'epochs': 60,              # 训练 60 轮
    'n_fold': 5,               # 5折交叉验证
    'sigma': 15,               # 热力图高斯半径
    'backbone': 'efficientnet-b5', # 强大的 B5 骨干
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"🚀 [方案一·最终版] {CONFIG['backbone']} + 640px + 断点续训 | Device={CONFIG['device']}")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CONFIG['seed'])

# ==========================================
# 2. 核心算法函数
# ==========================================
def generate_heatmap(width, height, x_pos, y_pos, sigma):
    # 生成高斯热力图
    X, Y = np.meshgrid(np.arange(0, width, 1), np.arange(0, height, 1))
    heatmap = np.exp(-((X - x_pos)**2 + (Y - y_pos)**2) / (2 * sigma**2))
    return heatmap.astype(np.float32)

def get_coords_subpixel(heatmap):
    # 【亚像素重心法】 - 突破像素精度的关键
    idx = np.argmax(heatmap)
    y_center, x_center = np.unravel_index(idx, heatmap.shape)
    
    # 取最亮点周围 7x7 区域计算重心
    win = 3 
    y_min = max(0, y_center - win)
    y_max = min(heatmap.shape[0], y_center + win + 1)
    x_min = max(0, x_center - win)
    x_max = min(heatmap.shape[1], x_center + win + 1)
    
    patch = heatmap[y_min:y_max, x_min:x_max]
    grid_y, grid_x = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing='ij')
    
    sum_val = np.sum(patch)
    if sum_val > 0:
        center_y = np.sum(patch * grid_y) / sum_val
        center_x = np.sum(patch * grid_x) / sum_val
    else:
        center_x, center_y = x_center, y_center
        
    return center_x, center_y

# ==========================================
# 3. 数据增强 (保留 CLAHE 以追求极致精度)
# ==========================================
def get_train_transforms():
    return A.Compose([
        A.Resize(height=CONFIG['img_size'], width=CONFIG['img_size']),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(p=0.2),         # 保留这个虽然慢一点，但对血管细节很有帮助
        A.GaussNoise(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

def get_valid_transforms():
    return A.Compose([
        A.Resize(height=CONFIG['img_size'], width=CONFIG['img_size']),
        A.Normalize(),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# ==========================================
# 4. 数据集定义
# ==========================================
class RetinalHeatmapDataset(Dataset):
    def __init__(self, img_dir, csv_file=None, mode='train', transform=None):
        self.img_dir = img_dir
        self.mode = mode
        self.transform = transform
        
        if self.mode == 'train':
            df = pd.read_csv(csv_file)
            df.columns = [c.strip() for c in df.columns]
            df = df.dropna()
            df['data'] = df['data'].astype(str)
            
            valid_data = []
            files_in_dir = os.listdir(img_dir)
            file_map = {f.lower(): f for f in files_in_dir}
            
            for idx, row in df.iterrows():
                raw_id = row['data'].strip()
                candidates = [raw_id, raw_id + '.jpg', raw_id.zfill(4) + '.jpg', 
                              raw_id + '.png', raw_id.zfill(4) + '.png']
                found_name = None
                for c in candidates:
                    if c.lower() in file_map:
                        found_name = file_map[c.lower()]
                        break
                if found_name:
                    row['data'] = found_name
                    valid_data.append(row)
            
            self.data = pd.DataFrame(valid_data).reset_index(drop=True)
            print(f"[{mode}] 数据量: {len(self.data)}")
            
        else:
            self.file_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        else:
            return len(self.file_names)

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_name = self.data.iloc[idx]['data']
            x_original = float(self.data.iloc[idx]['Fovea_X'])
            y_original = float(self.data.iloc[idx]['Fovea_Y'])
            
            if os.path.exists(os.path.join(self.img_dir, img_name)):
                img_path = os.path.join(self.img_dir, img_name)
            else:
                img_path = os.path.join('./test', img_name)
                
            image = cv2.imread(img_path)
            if image is None: raise FileNotFoundError(f"Error {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            transformed = self.transform(image=image, keypoints=[[x_original, y_original]])
            image = transformed['image']
            
            kp = transformed['keypoints']
            if len(kp) == 0: tx, ty = 0, 0
            else: tx, ty = kp[0]
            
            heatmap = generate_heatmap(CONFIG['img_size'], CONFIG['img_size'], tx, ty, CONFIG['sigma'])
            heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)
            return image, heatmap
        else:
            img_name = self.file_names[idx]
            img_path = os.path.join(self.img_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            transformed = self.transform(image=image, keypoints=[])
            image = transformed['image']
            return image, img_name, w, h

# ==========================================
# 5. 模型定义 (U-Net B5)
# ==========================================
def get_model():
    model = smp.Unet(
        encoder_name=CONFIG['backbone'],
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model

# ==========================================
# 6. 单折训练逻辑 (断点续训 + FP32)
# ==========================================
def train_one_fold(fold_idx, train_loader, val_loader, device):
    save_path = f'unet_b5_fold_{fold_idx}.pth'
    
    # 【断点续训逻辑】
    if os.path.exists(save_path):
        print(f"\n✅ 检测到模型 {save_path} 已存在，跳过该折训练...")
        return save_path

    print(f"\n>>> Fold {fold_idx + 1}/{CONFIG['n_fold']} 开始训练 (BatchSize={CONFIG['batch_size']})...")
    
    model = get_model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1)
    
    best_loss = float('inf')
    best_weights = None
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        scheduler.step()
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        
        if (epoch+1) % 5 == 0:
            print(f"Fold {fold_idx+1} | Ep {epoch+1} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict()
    
    # 确保保存了模型
    if best_weights is None: best_weights = model.state_dict()
    torch.save(best_weights, save_path)
    print(f"Fold {fold_idx+1} 完成. Best Val Loss: {best_loss:.6f}")
    return save_path

# ==========================================
# 7. 主流程
# ==========================================
if __name__ == '__main__':
    TRAIN_DIR = './train'
    TEST_DIR = './test'
    TRAIN_CSV = './fovea_localization_train_GT.csv'
    
    # 1. 训练阶段 (5折)
    full_dataset = RetinalHeatmapDataset(TRAIN_DIR, TRAIN_CSV, mode='train')
    kfold = KFold(n_splits=CONFIG['n_fold'], shuffle=True, random_state=CONFIG['seed'])
    indices = np.arange(len(full_dataset))
    
    model_paths = []
    device = torch.device(CONFIG['device'])
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        train_ds = Subset(RetinalHeatmapDataset(TRAIN_DIR, TRAIN_CSV, mode='train', transform=get_train_transforms()), train_idx)
        val_ds = Subset(RetinalHeatmapDataset(TRAIN_DIR, TRAIN_CSV, mode='train', transform=get_valid_transforms()), val_idx)
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
        
        path = train_one_fold(fold_idx, train_loader, val_loader, device)
        model_paths.append(path)

    # 2. 预测阶段 (融合 + TTA)
    if os.path.exists(TEST_DIR):
        print("\n开始 B5 融合预测 (TTA + 亚像素)...")
        models_list = []
        for path in model_paths:
            m = get_model().to(device)
            m.load_state_dict(torch.load(path))
            m.eval()
            models_list.append(m)
            
        test_dataset = RetinalHeatmapDataset(TEST_DIR, mode='test', transform=get_valid_transforms())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        results = []
        
        with torch.no_grad():
            for images, img_names, w_orig, h_orig in tqdm(test_loader):
                images = images.to(device)
                accumulated_heatmap = None
                
                for model in models_list:
                    # 正向预测
                    out = model(images)
                    # 翻转预测 (TTA)
                    img_flip = torch.flip(images, [3])
                    out_flip = model(img_flip)
                    out_flip = torch.flip(out_flip, [3])
                    avg_out = (out + out_flip) / 2.0
                    
                    if accumulated_heatmap is None: accumulated_heatmap = avg_out
                    else: accumulated_heatmap += avg_out
                
                # 取平均
                final_heatmap = accumulated_heatmap / len(models_list)
                final_heatmap = final_heatmap.cpu().numpy()[0, 0]
                
                # 亚像素坐标解析
                pred_x, pred_y = get_coords_subpixel(final_heatmap)
                
                # 坐标还原
                scale_x = w_orig.item() / CONFIG['img_size']
                scale_y = h_orig.item() / CONFIG['img_size']
                real_x = pred_x * scale_x
                real_y = pred_y * scale_y
                
                # ID处理
                full_name = img_names[0]
                fid = os.path.splitext(full_name)[0]
                try: fid = str(int(fid))
                except: pass
                
                results.append({'ImageID': f"{fid}_Fovea_X", 'value': real_x})
                results.append({'ImageID': f"{fid}_Fovea_Y", 'value': real_y})
                
        df = pd.DataFrame(results)[['ImageID', 'value']]
        df.to_csv('submission.csv', index=False)
        print("预测完成！结果已保存。")
    else:
        print("未找到测试集。")