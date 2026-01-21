import os
import cv2
import torch
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ================= 配置 =================
CONFIG = {
    "DATA_DIR": "./",
    "TEST_DIR": "test/image",  # 测试集原图位置
    "SAVE_DIR": "predictions", # 预测结果保存位置
    "IMG_SIZE": 768,           # 和你训练时一致
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}
# =======================================

def preprocess_fundus(img_rgb):
    green = img_rgb[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green)
    img_merged = cv2.merge([enhanced, enhanced, enhanced])
    return img_merged

def get_transforms():
    return A.Compose([
        A.Resize(CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def main():
    # 1. 准备目录
    if not os.path.exists(CONFIG['SAVE_DIR']):
        os.makedirs(CONFIG['SAVE_DIR'])
        
    # 2. 加载模型
    print("加载模型...")
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=CONFIG['DEVICE']))
    except:
        print("⚠️ 找不到 best_model.pth，请先训练！")
        return
    model.to(CONFIG['DEVICE'])
    model.eval()
    
    # 3. 读取测试文件
    test_abs_path = os.path.join(CONFIG['DATA_DIR'], CONFIG['TEST_DIR'])
    test_files = [f for f in os.listdir(test_abs_path) if f.lower().endswith(('.jpg', '.png'))]
    
    transforms = get_transforms()
    
    print(f"开始生成预测图片，共 {len(test_files)} 张...")
    
    with torch.no_grad():
        for file_name in tqdm(test_files):
            # 读取
            img_path = os.path.join(test_abs_path, file_name)
            original_img = cv2.imread(img_path)
            h, w = original_img.shape[:2]
            
            # 预处理
            image = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            image_proc = preprocess_fundus(image)
            
            # 增强/Tensor化
            augmented = transforms(image=image_proc)
            img_tensor = augmented['image'].unsqueeze(0).to(CONFIG['DEVICE'])
            
            # 预测
            output = model(img_tensor)
            pred = output.sigmoid().cpu().numpy()[0, 0]
            
            # 还原尺寸
            pred_resized = cv2.resize(pred, (w, h))
            
            # 二值化 (0 或 255)
            # 注意：保存为图片通常需要 0-255，而不是 0-1
            pred_binary = (pred_resized > 0.5).astype(np.uint8) * 255
            
            # 保存
            # 假设官方脚本需要同样的文件名，但可能是 png 格式
            # 这里我们保持原文件名，或者统一存为 png
            save_name = os.path.splitext(file_name)[0] + ".png"
            save_path = os.path.join(CONFIG['SAVE_DIR'], save_name)
            
            cv2.imwrite(save_path, pred_binary)
            
    print(f"✅ 所有预测图片已保存到 {CONFIG['SAVE_DIR']} 文件夹。")
    print("下一步：请运行官方提供的 segmentation_to_csv.py")

if __name__ == '__main__':
    main()