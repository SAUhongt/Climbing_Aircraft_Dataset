import pandas as pd
import os
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor

# 指定要保留的特征
desired_columns = ['segment', 'timestep', 'lat', 'lon', 'baroaltitudekalman',
                   'velocity', 'vertratecorr', 'taskalman', 'heading',
                   'ukalman', 'vkalman', 'tempkalman']

def get_transformer(lat, lon):
    # 根据经纬度动态选择适当的 UTM 区
    utm_zone = int((lon + 180) // 6) + 1
    crs_code = f"epsg:326{utm_zone:02d}" if lat >= 0 else f"epsg:327{utm_zone:02d}"
    return Transformer.from_crs("epsg:4326", crs_code, always_xy=True)

def calculate_lat_lon_diff_and_convert(grouped):
    # 计算纬度和经度的差分
    grouped['lat_diff'] = grouped['lat'].diff()
    grouped['lon_diff'] = grouped['lon'].diff()

    # 处理缺失值（差分会导致首行缺失）
    grouped.fillna(0, inplace=True)

    # 使用第一个点为原点的转换
    origin_lat = grouped['lat'].iloc[0]
    origin_lon = grouped['lon'].iloc[0]

    # 获取转换器，动态选择UTM投影
    transformer = get_transformer(origin_lat, origin_lon)

    # 将经纬度转换为直角坐标系
    grouped['x'], grouped['y'] = transformer.transform(grouped['lon'].values, grouped['lat'].values)

    # 以第一个点为原点进行平移
    origin_x, origin_y = transformer.transform(origin_lon, origin_lat)
    grouped['x'] -= origin_x
    grouped['y'] -= origin_y

    return grouped

def process_segment(segment_data):
    return calculate_lat_lon_diff_and_convert(segment_data)

def add_lat_lon_diffs(file_path, output_path, max_workers=4):
    # 加载数据
    data = pd.read_csv(file_path)

    # 只保留所需的特征
    filtered_data = data[desired_columns]

    # 按照'segment'和'timestep'排序
    filtered_data.sort_values(by=['segment', 'timestep'], inplace=True)

    # 按照segment分组
    grouped_segments = filtered_data.groupby('segment')

    # 使用ThreadPoolExecutor并行处理每个分组
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_segment, [group for _, group in grouped_segments]))

    # 将结果合并为一个DataFrame
    grouped = pd.concat(results, ignore_index=True)

    # 保留'segment', 'timestep'和差分及直角坐标数据
    final_columns = ['segment', 'timestep', 'lat', 'lon', 'lat_diff', 'lon_diff', 'x', 'y',
                     'baroaltitudekalman', 'velocity', 'vertratecorr', 'taskalman',
                     'heading', 'ukalman', 'vkalman', 'tempkalman']
    grouped = grouped[final_columns]

    # 保存到新文件
    grouped.reset_index(drop=True, inplace=True)  # 重置索引
    grouped.to_csv(output_path, index=False)

# 主程序
base_dir = 'E:\\climbing-aircraft-dataset\\filter_and_save_data'  # 数据所在目录
output_dir = 'E:\\climbing-aircraft-dataset\\lat_lon_diff_data'  # 输出目录
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

# 处理每个文件
for file in os.listdir(base_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(base_dir, file)
        output_path = os.path.join(output_dir, f'diff_{file}')

        # 输出处理日志
        print(f"正在处理文件: {file_path}")
        add_lat_lon_diffs(file_path, output_path, max_workers=4)  # 设置最大工作线程数
        print(f"已保存: {output_path}")

print("经纬度差分和直角坐标转换处理完成，已保存。")
