import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('E:\\climbing-aircraft-dataset\\B77W_valid.csv')

# 按照 segment 和 time 排序
df = df.sort_values(by=['segment', 'time'])

# 定义时间间隔的阈值（以秒为单位，例如 1 小时 = 3600 秒）
time_threshold = 3600

# 计算相邻时间点之间的差值
df['time_diff'] = df.groupby('segment')['time'].diff().fillna(0)

# 如果时间差大于阈值，标记为新的班次
df['new_flight'] = (df['time_diff'] > time_threshold).cumsum()

# 按照 segment 和 new_flight 分组
grouped = df.groupby(['segment', 'new_flight'])

# 获取所有分组的名称
all_flight_segments = list(grouped.groups.keys())

# 随机选择要绘制的航班数量（例如随机选择20个航班）
num_flights_to_plot = 20
random_flight_segments = pd.Series(all_flight_segments).sample(n=num_flights_to_plot).tolist()

# 遍历每个随机选取的航班编号并绘制轨迹
for flight_id in random_flight_segments:
    flight_data = grouped.get_group(flight_id)
    print(len(flight_data))

    # 创建一个新的绘图窗口
    plt.figure(figsize=(10, 8))

    # 绘制航班轨迹
    plt.plot(flight_data['lon'], flight_data['lat'], marker='o', linestyle='-', color='b', label=f'Segment {flight_id}')

    # 计算箭头的起点和方向
    x = flight_data['lon'].values
    y = flight_data['lat'].values
    u = np.diff(x, prepend=x[0])  # x方向的增量
    v = np.diff(y, prepend=y[0])  # y方向的增量

    # 绘制方向箭头
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color='r', width=0.002)

    # 添加图例和标签
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Trajectory of Segment {flight_id}')
    plt.legend()
    plt.grid(True)

    # 显示当前的图
    plt.show()
