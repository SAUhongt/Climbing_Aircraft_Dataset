import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 加载CSV数据
file_path = 'E:\\climbing-aircraft-dataset\\A320_train.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 2. 计算相关系数矩阵
correlation_matrix = df.corr()

# 3. 使用分层聚类热图
clustered_heatmap = sns.clustermap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=.5, figsize=(24, 20))

# 4. 获取并保存图像
plt.title('Clustered Feature Correlation Matrix')  # 需要单独调用来设置标题
clustered_heatmap.fig.suptitle('Clustered Feature Correlation Matrix')  # 通过 `fig` 对象设置标题
clustered_heatmap.savefig('clustered_correlation_matrix.png')  # 保存图像

plt.show()  # 显示图像

