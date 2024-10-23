import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
data = pd.read_csv('E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks.csv')

# 设置拆分比例
train_ratio = 0.7  # 训练集比例
test_ratio = 0.2   # 测试集比例
valid_ratio = 0.1  # 验证集比例

# 获取所有segment的唯一标识
segments = data['segment'].unique()

# 按照 segment 拆分
train_segments, temp_segments = train_test_split(segments, test_size=(1 - train_ratio), random_state=42)
test_segments, valid_segments = train_test_split(temp_segments, test_size=(valid_ratio / (test_ratio + valid_ratio)), random_state=42)

# 根据拆分后的 segment 获取相应的数据
train_data = data[data['segment'].isin(train_segments)]
test_data = data[data['segment'].isin(test_segments)]
valid_data = data[data['segment'].isin(valid_segments)]

# 保存拆分后的数据到文件
train_data.to_csv('E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_train.csv', index=False)
test_data.to_csv('E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_test.csv', index=False)
valid_data.to_csv('E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_valid.csv', index=False)

print("数据集已按segment分组并拆分保存")
