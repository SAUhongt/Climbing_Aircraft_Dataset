import pandas as pd
import os

# 指定要统计的特征
features_to_analyze = ['segment', 'timestep', 'lat', 'lon', 'baroaltitudekalman',
                       'velocity', 'vertratecorr', 'taskalman', 'heading',
                       'ukalman', 'vkalman', 'tempkalman']


def analyze_data(file_path):
    # 加载数据
    data = pd.read_csv(file_path)

    # 统计特征情况
    stats = {}
    for feature in features_to_analyze:
        feature_info = {
            '缺失值数量': data[feature].isnull().sum(),
            '唯一值数量': data[feature].nunique(),
            '最小值': data[feature].min() if data[feature].dtype in ['float64', 'int64'] else None,
            '最大值': data[feature].max() if data[feature].dtype in ['float64', 'int64'] else None,
            '平均值': data[feature].mean() if data[feature].dtype in ['float64', 'int64'] else None,
            '标准差': data[feature].std() if data[feature].dtype in ['float64', 'int64'] else None
        }
        stats[feature] = feature_info

    return stats


# 主程序
base_dir = 'E:\\climbing-aircraft-dataset\\filter_and_save_data'  # 数据所在目录
output_stats = []

# 处理每个文件
for file in os.listdir(base_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(base_dir, file)
        print(f"正在分析文件: {file_path}")

        file_stats = analyze_data(file_path)
        output_stats.append({'文件名': file})
        for feature, info in file_stats.items():
            output_stats[-1].update({f"{feature} - {key}": value for key, value in info.items()})

# 转换为DataFrame
output_df = pd.DataFrame(output_stats)

# 输出到控制台
print(output_df)

# 保存到CSV文件
output_file_path = os.path.join(base_dir, 'features_analysis.csv')
output_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
print(f"分析结果已保存到: {output_file_path}")

