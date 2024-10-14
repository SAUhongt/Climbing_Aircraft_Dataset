import pandas as pd
import os
from pyproj import Proj, Transformer
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# 指定要保留的特征
desired_columns = ['segment', 'timestep', 'lat', 'lon', 'baroaltitudekalman',
                   'velocity', 'vertratecorr', 'taskalman', 'heading',
                   'ukalman', 'vkalman', 'tempkalman']

# 创建WGS84投影和对应的直角坐标系
wgs84 = Proj('epsg:4326')  # WGS84
utm = Proj('epsg:3857')    # Web Mercator
transformer = Transformer.from_proj(wgs84, utm)

def lat_lon_to_cartesian(lat, lon):
    """
    将经纬度转换为平面直角坐标（米），使用pyproj
    """
    try:
        x, y = transformer.transform(lon, lat)  # 注意顺序为 (lon, lat)
        return x, y
    except Exception as e:
        print(f"经纬度转换失败 (lat: {lat}, lon: {lon})，错误信息: {e}")
        return np.nan, np.nan

def process_group(group):
    """
    处理每个分组，将经纬度转换为直角坐标
    """
    group[['x', 'y']] = group.apply(
        lambda row: pd.Series(lat_lon_to_cartesian(row['lat'], row['lon'])),
        axis=1
    )
    return group

def filter_and_save_data(file_path, output_path):
    # 加载数据
    data = pd.read_csv(file_path)

    # 只保留所需的特征
    filtered_data = data[desired_columns]

    # 按照segment分组，然后按timestep排序
    grouped = filtered_data.groupby('segment', group_keys=False).apply(lambda x: x.sort_values('timestep'))

    # 使用并行处理转换经纬度，限制最大工作进程数
    max_workers = min(multiprocessing.cpu_count() - 1, 4)  # 限制为 CPU 核心数 - 1 或 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        grouped = pd.concat(list(executor.map(process_group, [group for _, group in grouped.groupby('segment')])))

    # 以每条轨迹的第一个点作为坐标原点
    grouped['x'] -= grouped.groupby('segment')['x'].transform('first')
    grouped['y'] -= grouped.groupby('segment')['y'].transform('first')

    # 保留'segment', 'timestep', 'lat', 'lon'和直角坐标
    final_columns = ['segment', 'timestep', 'lat', 'lon', 'x', 'y'] + desired_columns[3:]  # 保留其他特征
    grouped = grouped[final_columns]

    # 保存到新文件
    grouped.reset_index(drop=True, inplace=True)  # 重置索引
    grouped.to_csv(output_path, index=False)

if __name__ == '__main__':
    # 主程序
    base_dir = 'E:\\climbing-aircraft-dataset\\filter_and_save_data'  # 数据所在目录
    output_dir = 'E:\\climbing-aircraft-dataset\\cartesian_data'  # 输出目录
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

    # 处理每个文件
    for file in os.listdir(base_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(base_dir, file)
            output_path = os.path.join(output_dir, f'cartesian_{file}')

            # 输出处理日志
            print(f"正在处理文件: {file_path}")
            filter_and_save_data(file_path, output_path)
            print(f"已保存: {output_path}")

    print("数据处理完成，已保存直角坐标数据。")
