import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

# ================= 配置 =================
# 你那个分数为 0.11 的 csv 文件
INPUT_CSV = 'submission.csv' 
# 输出的新文件名
OUTPUT_CSV = 'submission_row_major.csv'
# 测试集图片文件夹 (必须存在，用于获取宽高)
TEST_IMG_DIR = os.path.join('test', 'image') 
# =======================================

def rle_decode_column_major(mask_rle, shape):
    '''
    解码你当前的提交 (你是按列优先 .T 编码的)
    '''
    if pd.isna(mask_rle) or str(mask_rle) == 'nan':
        return np.zeros(shape, dtype=np.uint8)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    # 还原为你生成时的逻辑 (.T)
    return img.reshape((shape[1], shape[0])).T

def rle_encode_row_major(img):
    '''
    【核心修改】
    不使用 .T (转置)，改用默认的行优先 (Row-major)
    '''
    # 注意：这里去掉了 .T
    pixels = img.flatten() 
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def main():
    print(f"正在读取 {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    
    new_rles = []
    
    print("正在反转 RLE 方向 (尝试 Row-major 格式)...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        file_id = row['Id']
        old_rle = row['Predicted']
        
        # 1. 找原图读尺寸
        filename = f"{file_id}.jpg"
        img_path = os.path.join(TEST_IMG_DIR, filename)
        if not os.path.exists(img_path):
             filename = f"{file_id}.png"
             img_path = os.path.join(TEST_IMG_DIR, filename)
             
        if not os.path.exists(img_path):
            print(f"❌ 警告: 找不到 ID {file_id} 的原图")
            new_rles.append(old_rle)
            continue
            
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # 2. 先还原出正确的图像 (你之前的可视化证明了这一步能得到正确的图)
        mask_matrix = rle_decode_column_major(old_rle, (h, w))
        
        # 3. 【关键】用另一种方式编码回去 (去掉 .T)
        new_rle = rle_encode_row_major(mask_matrix)
        new_rles.append(new_rle)
        
    # 保存
    df['Predicted'] = new_rles
    # 确保 ID 是整数排序 (防止 sample_submission 的顺序影响)
    df.sort_values(by='Id', key=lambda col: col.astype(int), inplace=True)
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 生成完毕: {OUTPUT_CSV}")
    print(">>> 请提交这个新文件，它去掉了 .T 转置。")

if __name__ == '__main__':
    main()