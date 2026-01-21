import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as pd_nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
CONFIG = {
    "DATA_DIR": "./",          # 根目录
    "IMG_SIZE": 768,           # [升级] 提高分辨率以保留血管细节 (显存不够可改回512)
    "BATCH_SIZE": 2,           # [调整] 分辨率大了，BatchSize改小防止爆显存
    "LR": 3e-4,                
    "EPOCHS": 50,              
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "SEED": 2025,
    "RESUME_PATH": None,       # 若要断点续训，改为 "last_checkpoint.pth"
    "SAVE_NAME": "last_checkpoint.pth"
}

print(f"Running on device: {CONFIG['DEVICE']}")

# ==========================================
# 2. 图像预处理工具
# ==========================================
def preprocess_fundus(img_rgb):
    # 提取绿色通道
    green = img_rgb[:, :, 1]
    # CLAHE 增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    # 转回三通道适配模型输入
    img_merged = cv2.merge([enhanced, enhanced, enhanced])
    return img_merged

# ==========================================
# 3. 数据集定义 (Dataset Class)
# ==========================================
class VesselDataset(Dataset):
    def __init__(self, df, transforms=None, mode='train', img_dir=None, mask_dir=None):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.img_dir = img_dir    
        self.mask_dir = mask_dir 
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_name = self.df.iloc[idx]['Id']
        
        # 1. 读取图片
        img_path = os.path.join(CONFIG['DATA_DIR'], self.img_dir, file_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"❌ 无法读取图片: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = preprocess_fundus(image)
        
        if self.mode == 'train':
            # 2. 读取 Mask (尝试匹配多种后缀)
            mask_path = os.path.join(CONFIG['DATA_DIR'], self.mask_dir, file_name)
            
            # 如果找不到同名文件，尝试其他后缀
            if not os.path.exists(mask_path):
                filename_no_ext = os.path.splitext(file_name)[0]
                for ext in ['.png', '.gif', '.tif', '.jpg', '.bmp']:
                    temp_path = os.path.join(CONFIG['DATA_DIR'], self.mask_dir, filename_no_ext + ext)
                    if os.path.exists(temp_path):
                        mask_path = temp_path
                        break
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            mask = mask / 255.0
            
            if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            mask = mask.unsqueeze(0)
            return image, mask.float()
        
        else:
            # Test 模式
            if self.transforms:
                augmented = self.transforms(image=image)
                image = augmented['image']
            return image, file_name

# ==========================================
# 4. 数据增强
# ==========================================
def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([A.GridDistortion(p=0.5), A.ElasticTransform(p=0.5)], p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# ==========================================
# 5. RLE 编码 (关键修复)
# ==========================================
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    # [关键修复] 必须先转置 (.T)，因为 Kaggle 是列优先 (Column-major)
    # 如果不加 .T，分数会非常低（约 0.1 左右）
    pixels = img.T.flatten() 
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# ==========================================
# 6. 主流程 (run)
# ==========================================
def run():
    # --- A. 准备数据 ---
    train_img_dir = os.path.join('train', 'image')
    train_mask_dir = os.path.join('train', 'label')
    
    abs_train_dir = os.path.join(CONFIG['DATA_DIR'], train_img_dir)
    if not os.path.exists(abs_train_dir):
        print(f"❌ 错误: 找不到路径 {abs_train_dir}")
        return

    all_files = os.listdir(abs_train_dir)
    train_files = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.tif', '.bmp', '.jpeg'))]
    
    print(f"✅ 找到 {len(train_files)} 张训练图片。")
    if len(train_files) == 0: return

    df_train = pd.DataFrame({'Id': train_files})
    train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=CONFIG['SEED'])
    
    train_dataset = VesselDataset(train_df, transforms=get_transforms('train'), mode='train', img_dir=train_img_dir, mask_dir=train_mask_dir)
    val_dataset = VesselDataset(val_df, transforms=get_transforms('valid'), mode='train', img_dir=train_img_dir, mask_dir=train_mask_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=2, pin_memory=True)
    
    # --- B. 定义模型 ---
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=3,                  
        classes=1,                      
        activation=None                 
    )
    model.to(CONFIG['DEVICE'])
    
    criterion_dice = smp.losses.DiceLoss(mode='binary')
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['LR'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'], eta_min=1e-6)

    # --- C. 断点续训 ---
    start_epoch = 0
    best_score = 0.0
    if CONFIG['RESUME_PATH'] is not None and os.path.exists(CONFIG['RESUME_PATH']):
        checkpoint = torch.load(CONFIG['RESUME_PATH'], map_location=CONFIG['DEVICE'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        print(f"Resuming from Epoch {start_epoch}, Best Score: {best_score:.4f}")

    # --- D. 训练循环 ---
    print("Start Training...")
    for epoch in range(start_epoch, CONFIG['EPOCHS']):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}")
        for images, masks in loop:
            images = images.to(CONFIG['DEVICE'])
            masks = masks.to(CONFIG['DEVICE'])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = 0.5 * criterion_bce(outputs, masks) + 0.5 * criterion_dice(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        scheduler.step()
        
        # 验证
        model.eval()
        val_score = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(CONFIG['DEVICE'])
                masks = masks.to(CONFIG['DEVICE'])
                outputs = model(images)
                preds = (outputs.sigmoid() > 0.5).float()
                
                tp = (preds * masks).sum().to(torch.float32)
                fp = (preds * (1 - masks)).sum().to(torch.float32)
                fn = ((1 - preds) * masks).sum().to(torch.float32)
                dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                val_score += dice.item()
        
        val_score /= len(val_loader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Dice: {val_score:.4f}")
        
        # 保存
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_score': best_score
        }
        torch.save(checkpoint, CONFIG['SAVE_NAME'])
        
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), "best_model.pth")
            print(f">>> Best Model Saved (Dice: {best_score:.4f})")

    # ==========================================
    # 7. 预测与生成提交 (包含 TTA 和 ID 修复)
    # ==========================================
    print("Start Inference with TTA...")
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=CONFIG['DEVICE']))
    model.eval()
    
    test_img_dir = os.path.join('test', 'image')
    abs_test_dir = os.path.join(CONFIG['DATA_DIR'], test_img_dir)
    
    if not os.path.exists(abs_test_dir):
        print("❌ Test directory not found.")
        return

    test_files = [f for f in os.listdir(abs_test_dir) if f.lower().endswith(('.jpg', '.png', '.tif'))]
    test_df = pd.DataFrame({'Id': test_files})
    
    test_dataset = VesselDataset(test_df, transforms=get_transforms('valid'), mode='test', img_dir=test_img_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    results = []
    
    with torch.no_grad():
        for image, file_name in tqdm(test_loader, desc="Testing"):
            image = image.to(CONFIG['DEVICE'])
            file_name = file_name[0]
            
            # --- TTA (Test Time Augmentation) ---
            # 1. 正常预测
            out1 = model(image).sigmoid().cpu().numpy()[0, 0]
            
            # 2. 水平翻转预测 (Horizontal Flip)
            img_h = torch.flip(image, [3])
            out_h = model(img_h).sigmoid().cpu().numpy()[0, 0]
            out2 = np.fliplr(out_h)
            
            # 3. 垂直翻转预测 (Vertical Flip)
            img_v = torch.flip(image, [2])
            out_v = model(img_v).sigmoid().cpu().numpy()[0, 0]
            out3 = np.flipud(out_v)
            
            # 平均结果
            pred = (out1 + out2 + out3) / 3.0
            
            # --- 还原尺寸 ---
            original_path = os.path.join(abs_test_dir, file_name)
            original_img = cv2.imread(original_path)
            h, w = original_img.shape[:2]
            
            pred_resized = cv2.resize(pred, (w, h))
            pred_binary = (pred_resized > 0.5).astype(np.uint8)
            
            rle_str = rle_encode(pred_binary)
            
            # --- [关键] ID 格式修复 ---
            # 将 "1.jpg" 转换为整数 1
            try:
                # 假设文件名是纯数字+后缀 (如 1.jpg)
                file_id = int(file_name.split('.')[0])
            except ValueError:
                # 如果文件名包含非数字字符，保留原样 (防止报错)
                file_id = file_name
            
            results.append({'Id': file_id, 'Predicted': rle_str})
            
    # 按 ID 排序 (可选，看起来更整齐)
    results.sort(key=lambda x: x['Id'] if isinstance(x['Id'], int) else 0)
            
    submission = pd.DataFrame(results)
    submission.to_csv('submission.csv', index=False)
    print("Done! submission.csv saved (ID fixed & Score bug fixed).")

if __name__ == '__main__':
    run()