import pandas as pd

# 1. 读取生成的 csv
df = pd.read_csv('submission.csv')

# 2. 定义转换函数：把 "1.jpg" 变成 1
def fix_id(filename):
    # 强制转为字符串，防止已经读取为数字
    filename = str(filename)
    # 如果包含 .jpg 或 .png，去掉后缀
    if '.' in filename:
        return int(filename.split('.')[0])
    return filename

# 3. 应用转换
df['Id'] = df['Id'].apply(fix_id)

# 4. 确保按 ID 排序 (虽然不是必须，但是个好习惯)
df = df.sort_values(by='Id')

# 5. 保存新的文件
output_file = 'submission_fixed.csv'
df.to_csv(output_file, index=False)

print(f"✅ 修复完成！请提交 {output_file} 到 Kaggle。")
print("前5行预览：")
print(df.head())