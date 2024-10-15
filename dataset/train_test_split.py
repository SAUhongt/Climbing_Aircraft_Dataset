import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
data = pd.read_csv('E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks.csv')

# 设置拆分比例
train_ratio = 0.7  # 训练集比例
test_ratio = 0.2   # 测试集比例
valid_ratio = 0.1  # 验证集比例

# 先拆分出训练集和临时集（包括测试集和验证集）
train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42)

# 再从临时集中拆分出测试集和验证集
test_data, valid_data = train_test_split(temp_data, test_size=(valid_ratio / (test_ratio + valid_ratio)), random_state=42)

# 保存拆分后的数据到文件
train_data.to_csv('E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_train.csv', index=False)
test_data.to_csv('E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_test.csv', index=False)
valid_data.to_csv('E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_valid.csv', index=False)

print("数据集已拆分并保存")
