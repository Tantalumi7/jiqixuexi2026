# =============================================================================
# NEU Plant Seedling Classification 2025 - Traditional ML Pipeline (Final Fixed)
# Strategy: SIFT-BoVW + HOG + LBP + HuMoments + ColorStats + Ensemble
# =============================================================================

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from skimage.feature import local_binary_pattern, hog
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# ===========================
# 1. 智能路径配置 (自动检测大小写)
# ===========================
BASE_PATH = '/kaggle/input/neu-plant-seedling-classification-2025/dataset-for-task1'

def find_folder(base, name_options):
    """在 base 目录下寻找 name_options 中的任意一个文件夹"""
    if not os.path.exists(base):
        raise FileNotFoundError(f"找不到根目录: {base}，请检查比赛数据集是否已添加。")
    
    available = os.listdir(base)
    for opt in name_options:
        if opt in available:
            return os.path.join(base, opt)
    raise FileNotFoundError(f"在 {base} 下找不到 {name_options} 中的任何文件夹。现有内容: {available}")

# 自动寻找 train/Train 和 test/Test
print("正在检测数据路径...")
TRAIN_DIR = find_folder(BASE_PATH, ['train', 'Train'])
TEST_DIR = find_folder(BASE_PATH, ['test', 'Test'])

print(f"✅ 训练集路径: {TRAIN_DIR}")
print(f"✅ 测试集路径: {TEST_DIR}")

# 参数设置
IMG_SIZE = 300           
VOCAB_SIZE = 400         # 视觉词典大小
SEED = 2025

# ===========================
# 2. 图像预处理 (去背景)
# ===========================
def preprocess_image(img):
    """转换为HSV -> 绿色掩膜 -> 形态学去噪 -> 保留最大轮廓"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 绿色范围 (关键参数)
    lower_green = np.array([25, 35, 35])
    upper_green = np.array([86, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 只保留最大轮廓
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        mask_clean = np.zeros_like(mask)
        cv2.drawContours(mask_clean, [c], -1, 255, -1)
        mask = mask_clean
        
    result = cv2.bitwise_and(img, img, mask=mask)
    return result, mask

# ===========================
# 3. 特征提取 (核心逻辑)
# ===========================
# 兼容不同版本的 OpenCV SIFT
try:
    sift = cv2.SIFT_create()
except:
    try:
        sift = cv2.xfeatures2d.SIFT_create()
    except:
        print("警告: 无法初始化 SIFT，请检查 opencv-python 和 opencv-contrib-python 版本")

def extract_features(img, kmeans_model):
    """提取 SIFT(BoVW) + HOG + LBP + Hu + Color"""
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    clean_img, mask = preprocess_image(img)
    gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
    
    features = []
    
    # 1. SIFT BoVW
    kp, des = sift.detectAndCompute(gray, mask)
    vocab_hist = np.zeros(VOCAB_SIZE)
    if des is not None and len(des) > 0:
        words = kmeans_model.predict(des.astype(np.float64))
        for w in words: vocab_hist[w] += 1
    if vocab_hist.sum() > 0: vocab_hist /= vocab_hist.sum()
    features.extend(vocab_hist)
    
    # 2. HOG (Shape)
    try:
        fd_hog = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                     cells_per_block=(2, 2), visualize=False)
        features.extend(fd_hog)
    except:
        features.extend([0]*1000) # 容错

    # 3. LBP (Texture)
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    (hist_lbp, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-6)
    features.extend(hist_lbp)
    
    # 4. Hu Moments (Geometry)
    hu = cv2.HuMoments(cv2.moments(mask)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    features.extend(hu)
    
    # 5. Color Stats
    hsv_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2HSV)
    for i in range(3):
        pixels = hsv_img[:,:,i][mask > 0]
        if len(pixels) > 0: features.extend([pixels.mean(), pixels.std()])
        else: features.extend([0, 0])
            
    return np.array(features)

# ===========================
# 4. 训练流程
# ===========================
def main():
    # --- Step 1: 读取所有训练图片用于构建词典 ---
    print(">>> 正在扫描训练数据...")
    train_paths = []
    train_labels = []

    for label in os.listdir(TRAIN_DIR):
        class_dir = os.path.join(TRAIN_DIR, label)
        if not os.path.isdir(class_dir): continue
        for f in os.listdir(class_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                train_paths.append(os.path.join(class_dir, f))
                train_labels.append(label)
    
    print(f"找到 {len(train_paths)} 张训练图片。")

    # --- Step 2: 构建 SIFT 词典 ---
    print(">>> 正在构建 SIFT 词典 (KMeans)...")
    descriptors = []
    # 采样部分图片加速 (如果想更准，可以增加采样数)
    sample_paths = np.random.choice(train_paths, min(len(train_paths), 1000), replace=False)
    
    for path in sample_paths:
        img = cv2.imread(path)
        if img is None: continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        _, mask = preprocess_image(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, mask)
        if des is not None: descriptors.append(des)
            
    all_des = np.vstack(descriptors)
    kmeans = MiniBatchKMeans(n_clusters=VOCAB_SIZE, batch_size=1000, random_state=SEED).fit(all_des)
    print("✅ 词典构建完成。")

    # --- Step 3: 提取训练特征 ---
    print(">>> 正在提取训练集特征 (这需要几分钟)...")
    X_train = []
    for i, path in enumerate(train_paths):
        img = cv2.imread(path)
        X_train.append(extract_features(img, kmeans))
        if i % 100 == 0: print(f"进度: {i}/{len(train_paths)}", end='\r')
    
    X_train = np.array(X_train)
    y_train = np.array(train_labels)
    
    # 预处理
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    print(f"\n特征提取完毕。维度: {X_train.shape}")

    # --- Step 4: 训练模型 (Stacking) ---
    print(">>> 正在训练集成模型...")
    # SVM: 使用 balanced 权重解决不平衡
    clf_svm = SVC(C=10, kernel='rbf', probability=True, class_weight='balanced', random_state=SEED)
    # RF
    clf_rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=SEED, n_jobs=-1)
    # XGB
    clf_xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, eval_metric='mlogloss', random_state=SEED, n_jobs=-1)

    eclf = VotingClassifier(estimators=[('svm', clf_svm), ('rf', clf_rf), ('xgb', clf_xgb)], voting='soft')
    
    eclf.fit(X_scaled, y_enc)
    print("✅ 模型训练完毕。")

    # --- Step 5: 预测测试集 (5-View TTA) ---
    print(">>> 正在处理测试集 (启用 TTA 增强)...")
    test_files = sorted(os.listdir(TEST_DIR))
    predictions = []

    # TTA 变换函数
    def augment(img, mode):
        if mode == 0: return img
        if mode == 1: return cv2.flip(img, 1) # 水平翻转
        if mode == 2: return cv2.flip(img, 0) # 垂直翻转
        if mode == 3: return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if mode == 4: return cv2.rotate(img, cv2.ROTATE_180)
        return img

    for idx, file in enumerate(test_files):
        path = os.path.join(TEST_DIR, file)
        img_orig = cv2.imread(path)
        if img_orig is None: continue
        
        # 5次变换，累加概率
        probs_sum = None
        for mode in range(5):
            img_aug = augment(img_orig, mode)
            feat = extract_features(img_aug, kmeans)
            feat_scaled = scaler.transform(feat.reshape(1, -1))
            prob = eclf.predict_proba(feat_scaled)
            
            if probs_sum is None: probs_sum = prob
            else: probs_sum += prob
            
        # 取最大概率对应的类别
        pred_idx = np.argmax(probs_sum)
        pred_label = le.inverse_transform([pred_idx])[0]
        
        predictions.append({'file': file, 'species': pred_label})
        if idx % 50 == 0: print(f"预测进度: {idx}/{len(test_files)}", end='\r')

    # --- Step 6: 保存 ---
    df_sub = pd.DataFrame(predictions)
    df_sub.to_csv('submission_final.csv', index=False)
    print("\n🎉 提交文件生成成功: submission_final.csv")

if __name__ == '__main__':
    main()