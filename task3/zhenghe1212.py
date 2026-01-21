import pandas as pd
import numpy as np
from collections import Counter

# 1. 把你所有的 csv 文件名填在这里
csv_files = [
    r"C:\Users\shmil\Desktop\submission_fer_070.csv",   # 假设这是你分最高的
    r"C:\Users\shmil\Desktop\submission_emotion_correct.csv",   # 之前的
    r"C:\Users\shmil\Desktop\submission_fer_final.csv",
    r"C:\Users\shmil\Desktop\submission_fer_swin_tencrop.csv",
    # 如果还有别的，继续加...
]

print("正在读取 CSV 文件...")
dfs = [pd.read_csv(f) for f in csv_files]

# 2. 确保 ID 顺序一致
ids = dfs[0]['ID'].values
final_preds = []

# 3. 逐行投票
for i in range(len(ids)):
    # 拿到这一行，所有 csv 给出的预测结果
    preds = [df.iloc[i]['Emotion'] for df in dfs]
    
    # 找到出现次数最多的数字 (众数)
    # 如果平票，就选第一个 csv (通常放分最高的那个) 的结果
    most_common = Counter(preds).most_common(1)[0][0]
    final_preds.append(most_common)

# 4. 保存
submission = pd.DataFrame({'ID': ids, 'Emotion': final_preds})
submission.to_csv(r"C:\Users\shmil\Desktop\submission_vote.csv", index=False)
print("✅ 融合完成！生成文件: submission_vote.csv")