import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# ================= 配置 =================
# 你的 submission.csv 路径
SUBMISSION_FILE = 'submission.csv'
# 测试集图片文件夹 (用于读取尺寸和底图对比)
TEST_IMG_DIR = os.path.join('test', 'image') 
# =======================================

def rle_decode(mask_rle, shape):
    '''
    解码函数：将 Kaggle 的 RLE 字符串还原为二维图像
    '''
    if pd.isna(mask_rle) or str(mask_rle) == 'nan':
        return np.zeros(shape, dtype=np.uint8)
    
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    
    # 1. 还原出一维像素流
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
        
    # 2. 【关键】Kaggle 是列优先 (Column-major)，所以我们要 reshape 成 (W, H) 然后转置
    # 如果这里还原出来的图是正常的，说明你的编码逻辑是对的
    return img.reshape((shape[1], shape[0])).T

def check_submission():
    if not os.path.exists(SUBMISSION_FILE):
        print("❌ 找不到 submission.csv，请先运行 main.py 生成结果。")
        return

    df = pd.read_csv(SUBMISSION_FILE)
    print(f"📄 读取提交文件，共 {len(df)} 行")
    print(f"📝 ID 示例: {df.iloc[0]['Id']} (应为纯数字)")

    # 随机抽取 3 张图进行检查
    sample_indices = [0, 5, 10] if len(df) > 10 else range(len(df))
    
    plt.figure(figsize=(15, 5*len(sample_indices)))
    
    for i, idx in enumerate(sample_indices):
        row = df.iloc[idx]
        file_id = row['Id']
        rle = row['Predicted']
        
        # 尝试寻找原图
        filename = f"{file_id}.jpg"
        img_path = os.path.join(TEST_IMG_DIR, filename)
        if not os.path.exists(img_path):
             filename = f"{file_id}.png"
             img_path = os.path.join(TEST_IMG_DIR, filename)
        
        if os.path.exists(img_path):
            original_img = cv2.imread(img_path)
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            h, w = original_img.shape[:2]
            
            # 解码 Mask
            try:
                mask = rle_decode(rle, (h, w))
                
                # 画图
                plt.subplot(len(sample_indices), 3, i*3 + 1)
                plt.imshow(original_img)
                plt.title(f"Original ID: {file_id}")
                plt.axis('off')
                
                plt.subplot(len(sample_indices), 3, i*3 + 2)
                plt.imshow(mask, cmap='gray')
                plt.title("Decoded Prediction")
                plt.axis('off')
                
                plt.subplot(len(sample_indices), 3, i*3 + 3)
                plt.imshow(original_img)
                plt.imshow(mask, alpha=0.4, cmap='Reds') # 叠加显示
                plt.title("Overlay")
                plt.axis('off')
                
            except Exception as e:
                print(f"❌ ID {file_id} 解码失败: {e}")
        else:
            print(f"⚠️ 找不到原图 {filename}，跳过可视化")

    plt.tight_layout()
    plt.show()
    print("✅ 可视化完成。请检查图片：")
    print("1. 血管是否清晰？(如果全黑，说明阈值太高)")
    print("2. 血管位置是否和原图重合？(如果错位或旋转，说明 RLE 编码方向反了)")

if __name__ == '__main__':
    check_submission()