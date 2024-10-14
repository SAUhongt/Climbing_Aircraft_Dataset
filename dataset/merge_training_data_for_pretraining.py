import pandas as pd
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def merge_training_data_for_pretraining(input_dir, pretrain_output_file):
    """
    合并所有机型的训练集数据，生成一个预训练数据集文件。
    """
    try:
        combined_data = pd.DataFrame()
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('train.csv'):
                    file_path = os.path.join(root, file)
                    logging.info(f"加载训练集文件: {file_path}")
                    data = pd.read_csv(file_path)
                    combined_data = pd.concat([combined_data, data], ignore_index=True)

        # 保存合并后的预训练数据集
        combined_data.to_csv(pretrain_output_file, index=False)
        logging.info(f"预训练数据集已保存: {pretrain_output_file}")
    except Exception as e:
        logging.error(f"合并预训练数据时发生错误: {e}")


def copy_downstream_data(input_dir, downstream_output_dir):
    """
    复制每个机型的测试集和验证集数据到下游任务目录中。
    """
    try:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.csv') and ('test.csv' in file or 'valid.csv' in file):
                    file_path = os.path.join(root, file)

                    # 获取机型名称和输出路径
                    aircraft_type = os.path.basename(root)
                    target_dir = os.path.join(downstream_output_dir, aircraft_type)
                    os.makedirs(target_dir, exist_ok=True)

                    output_path = os.path.join(target_dir, file)

                    # 复制文件
                    df = pd.read_csv(file_path)
                    df.to_csv(output_path, index=False)
                    logging.info(f"已保存 {file} 到 {output_path}")
    except Exception as e:
        logging.error(f"复制下游任务数据时发生错误: {e}")


def main():
    input_dir = 'E:\\climbing-aircraft-dataset\\normalized_data'  # 归一化后的数据目录
    pretrain_output_file = 'E:\\climbing-aircraft-dataset\\pretraining_data.csv'  # 预训练数据输出文件
    downstream_output_dir = 'E:\\climbing-aircraft-dataset\\downstream_data'  # 下游任务数据目录

    # 合并训练集数据生成预训练数据集
    merge_training_data_for_pretraining(input_dir, pretrain_output_file)

    # 复制下游任务数据（测试集和验证集）
    copy_downstream_data(input_dir, downstream_output_dir)

    logging.info("数据集处理完成。")


if __name__ == '__main__':
    main()
