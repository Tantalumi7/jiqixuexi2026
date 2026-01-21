# =============================================================================
# NEU FER - Face Alignment & Cropping Tool (Commercial Standard)
# Method: MTCNN (Multi-task Cascaded Convolutional Networks)
# Input: fer_data -> Output: fer_data_aligned
# =============================================================================

import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

# ===========================
# 配置参数
# ===========================
# 源数据路径
SOURCE_DIR = r"C:\Users\shmil\Desktop\fer_data"
# 新数据保存路径 (会自动创建)
TARGET_DIR = r"C:\Users\shmil\Desktop\fer_data_aligned"

IMG_SIZE = 224  # VGGFace2 标准尺寸
BATCH_SIZE = 1  # 预处理建议单张处理，方便容错

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 正在使用设备: {device} 进行 MTCNN 处理")

# ===========================
# 初始化 MTCNN
# ===========================
# keep_all=False: 只保留一张脸
# select_largest=True:如果有好几张脸，只取最大的那张（主角）
# margin=20: 裁剪时多留一点边缘，不要切得太紧，防止把下巴切掉
mtcnn = MTCNN(
    image_size=IMG_SIZE, 
    margin=20, 
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], 
    factor=0.709, 
    post_process=True,
    keep_all=False,
    select_largest=True,
    device=device
)

def process_directory(source_root, target_root):
    # 统计数据
    total_imgs = 0
    face_detected = 0
    no_face = 0
    
    # 获取所有图片列表
    all_files = []
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                source_path = os.path.join(root, file)
                # 计算相对路径，以便在目标文件夹重建结构
                # 例如: train\Anger\001.jpg
                rel_path = os.path.relpath(source_path, source_root)
                target_path = os.path.join(target_root, rel_path)
                all_files.append((source_path, target_path))

    print(f"📂 扫描到 {len(all_files)} 张图片，开始处理...")

    for source_path, target_path in tqdm(all_files, desc="Aligning"):
        # 创建目标文件夹
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        try:
            # 1. 读取图片 (MTCNN 需要 PIL 格式)
            img = Image.open(source_path).convert('RGB')
            
            # 2. 尝试用 MTCNN 检测并保存
            # mtcnn(img, save_path) 会自动完成 检测->对齐->裁剪->保存
            # 如果检测成功，返回 tensor；如果失败（无人脸），返回 None
            ret = mtcnn(img, save_path=target_path)
            
            if ret is not None:
                face_detected += 1
            else:
                # 3. 兜底策略：如果没检测到人脸，直接 Resize 原图并保存
                # 这种通常是图片太黑、太模糊，或者根本不是人脸
                # 我们不能丢弃它，因为测试集还需要预测
                img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
                img_resized.save(target_path)
                no_face += 1
                
        except Exception as e:
            print(f"\n❌ 处理出错: {source_path} | Error: {e}")
            # 出错了也尝试硬存一张原图，防止缺文件
            try:
                img = Image.open(source_path).convert('RGB')
                img.resize((IMG_SIZE, IMG_SIZE)).save(target_path)
            except:
                pass

    print("\n========================================")
    print("✅ 数据清洗完成！")
    print(f"📍 原数据: {source_root}")
    print(f"📍 新数据: {target_root}")
    print("----------------------------------------")
    print(f"😊 成功检测并对齐人脸: {face_detected} 张")
    print(f"⚠️ 未检测到人脸(使用原图): {no_face} 张")
    print("========================================")

if __name__ == '__main__':
    # 处理 Train 和 Test
    # 假设 fer_data 下面直接是 train 和 test 文件夹
    # 脚本会递归处理所有子文件夹
    process_directory(SOURCE_DIR, TARGET_DIR)