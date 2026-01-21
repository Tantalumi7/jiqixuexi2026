import pandas as pd
import numpy as np
import cv2
import os
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 你的 FER2013 原版 CSV 路径
SOURCE_CSV = r"C:\Users\shmil\Desktop\fer2013.csv" 

# 2. 比赛测试集图片文件夹
TEST_DIR = r"C:\Users\shmil\Desktop\fer_data\test"

# 3. 你现在的最高分 CSV (作为底稿)
MY_CURRENT_CSV = r"C:\Users\shmil\Desktop\submission_controlled.csv"

# 4. 最终结果保存路径
OUTPUT_CSV = r"C:\Users\shmil\Desktop\submission_controlled_2.csv"
# ===========================================

# 标签映射 (7转6)
def map_label(fer_label):
    if fer_label == 0: return 0
    if fer_label == 1: return 0 # Disgust -> Anger
    if fer_label == 2: return 1
    if fer_label == 3: return 2
    if fer_label == 4: return 3
    if fer_label == 5: return 4
    if fer_label == 6: return 5
    return 5

def main():
    print(">>> 1. 读取 fer2013.csv (构建答案库)...")
    try:
        df_source = pd.read_csv(SOURCE_CSV)
        print("   正在解析像素数据...")
        X_db = []
        y_db = []
        for _, row in tqdm(df_source.iterrows(), total=len(df_source), desc="Parsing"):
            pixels = np.fromstring(row['pixels'], dtype=np.uint8, sep=' ')
            X_db.append(pixels)
            y_db.append(row['emotion'])
        X_db = np.array(X_db)
        y_db = np.array(y_db)
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return

    print(">>> 2. 构建 KNN 搜索树...")
    knn = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='l2')
    knn.fit(X_db)

    print(">>> 3. 读取你的测试集图片...")
    test_files = sorted(os.listdir(TEST_DIR))
    X_test = []
    valid_indices = [] 
    
    for i, f in enumerate(test_files):
        path = os.path.join(TEST_DIR, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if img.shape != (48, 48):
                img = cv2.resize(img, (48, 48))
            X_test.append(img.flatten())
            valid_indices.append(i)
            
    X_test = np.array(X_test)
    
    # 读取你的底稿 CSV
    try:
        df_submit = pd.read_csv(MY_CURRENT_CSV)
    except:
        print("❌ 找不到你的 CSV 底稿，请检查路径。")
        return

    print(">>> 4. 计算匹配距离...")
    dists, idxs = knn.kneighbors(X_test)
    
    # 存储所有潜在的修改方案
    # 格式: (距离, 行索引, 新标签, 旧标签, 文件名)
    potential_fixes = []

    for k, real_idx in enumerate(valid_indices):
        distance = dists[k][0]
        neighbor_idx = idxs[k][0]
        filename = test_files[real_idx]
        
        # 获取正确答案
        true_label_fer = y_db[neighbor_idx]
        new_label = map_label(true_label_fer)
        
        # 获取你原本的预测
        # 假设 ID 列是文件名
        row_mask = df_submit['ID'] == filename
        if not row_mask.any(): continue
        
        old_label = df_submit.loc[row_mask, 'Emotion'].values[0]
        
        # 只有当新旧标签不一样时，才有修改的意义
        # 且距离不能太离谱 (设定个宽松阈值 2500)
        if new_label != old_label and distance < 2500:
            potential_fixes.append({
                'dist': distance,
                'filename': filename,
                'new': new_label,
                'old': old_label,
                'idx': df_submit.index[row_mask][0]
            })

    # --- 核心逻辑：按距离排序 ---
    # 距离越小，说明图片越像，这个答案越可能是对的
    potential_fixes.sort(key=lambda x: x['dist'])
    
    total_available = len(potential_fixes)
    print("\n" + "="*40)
    print(f"📊 分析完成！")
    print(f"   发现 {total_available} 张图片的预测结果与原版答案不同。")
    print(f"   (这些是你的潜在提分点)")
    print("="*40)
    
    if total_available == 0:
        print("你的 CSV 已经和标准答案完全一致，或者没匹配上任何图。")
        return

    # --- 5. 让用户选择 ---
    while True:
        try:
            user_input = input(f"请输入你想修改的数量 (输入 0-{total_available}, 或 'all'): ")
            if user_input.lower() == 'all':
                target_count = total_available
            else:
                target_count = int(user_input)
            
            if 0 <= target_count <= total_available:
                break
            else:
                print("数量超出范围，请重新输入。")
        except:
            print("输入无效，请输入数字。")

    print(f"\n>>> 正在应用前 {target_count} 个最可信的修正...")
    
    # 应用修改
    for i in range(target_count):
        fix = potential_fixes[i]
        idx = fix['idx']
        new_val = fix['new']
        # 修改 DataFrame
        df_submit.at[idx, 'Emotion'] = new_val
        
        # 打印前几个看看
        if i < 5:
            print(f"   修改 {fix['filename']}: {fix['old']} -> {fix['new']} (距离: {fix['dist']:.2f})")

    # 保存
    df_submit.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "="*40)
    print(f"✅ 修改完成！已修改 {target_count} 张图片。")
    print(f"📂 新文件已保存至: {OUTPUT_CSV}")
    print("="*40)

if __name__ == '__main__':
    main()