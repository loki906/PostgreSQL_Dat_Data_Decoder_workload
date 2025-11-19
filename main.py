from operator import index
import os
from tkinter import Variable
from xml.dom.minidom import Element
# from iotdb.Session import Session
# from iotdb.utils.IoTDBConstants import TSDataType, TSEncoding, Compressor
from numpy import var
from Dat_Data_Decoder import CAE_Decoder
from PostgreSQL_Interface import create_postgres_connection, insert_cae_simulation_data, create_tables, close_postgres_connection
import psycopg2
from psycopg2 import OperationalError
import Zone
import pandas as pd
import numpy as np
from Zone import Zone_3D
from Dat_Data_Decoder import print_array
from pathlib import Path
from collections import defaultdict


def extract_ship_info_from_path(file_path):
    """
    从文件路径中提取船舶类型和尺度信息
    路径示例: data/gpudb/downloaded/JBC_615k/Postprocessing/200.dat
    """
    path_parts = Path(file_path).parts
    
    # 定义已知的船舶类型
    known_ship_types = ['JBC', 'Kvlcc2', 'Suboff']
    
    for part in path_parts:
        if '_' in part:
            # 可能的格式: JBC_615k, Kvlcc2_351k, Suboff_528k
            for ship_type in known_ship_types:
                if ship_type in part:
                    try:
                        _, scale_part = part.split('_')
                        # 验证尺度格式 (数字+k)
                        if scale_part[:-1].isdigit() and scale_part.endswith('k'):
                            return ship_type, scale_part
                    except ValueError:
                        continue
    # 如果无法解析，返回默认值
    print(f"警告: 无法从路径解析船舶信息: {file_path}")
    return "Unknown", "000k"


def find_dat_files(base_path):
    """
    递归查找所有.dat文件，并提取船舶信息
    """
    dat_files_with_info = []
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.dat'):
                file_path = os.path.join(root, file)
                
                # 从路径中提取船舶类型和尺度
                ship_type, scale = extract_ship_info_from_path(file_path)
                
                # 从文件名提取时间步 (去掉.dat后缀)
                timestep_str = os.path.splitext(file)[0]
                try:
                    timestep = int(timestep_str)
                except ValueError:
                    timestep = 0
                    print(f"警告: 无法从文件名解析时间步: {file}")
                
                dat_files_with_info.append({
                    'file_path': file_path,
                    'ship_type': ship_type,
                    'scale': scale,
                    'timestep': timestep
                })
    
    return dat_files_with_info


def verify_data_insertion(connection, ship_type, timestep):
    """
    验证数据插入情况
    """
    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT variable, is_element, array_length(data, 1) as data_length, 
                   max_range IS NULL as max_range_null
            FROM cae_simulation_data 
            WHERE ship_type = %s AND timestep = %s
            ORDER BY variable, is_element
        """, (ship_type, int(timestep)))
        
        results = cursor.fetchall()
        print(f"验证数据插入 - ShipType: {ship_type}, Timestep: {timestep}:")
        for row in results:
            max_range_status = "待导入" if row[3] else f"已导入({row[2]})"
            print(f"  Variable: {row[0]}, IsElement: {row[1]}, DataLength: {row[2]}, MaxRange: {max_range_status}")
            
        cursor.close()
    except Exception as e:
        print(f"验证数据时出错: {e}")

def validate_dataframe(df, file_path):
    """
    验证DataFrame中的数据是否正确
    """
    print(f"验证数据: {file_path}")
    
    # 检查variable字段的取值
    valid_variables = ['X', 'Y', 'Z', 'u', 'v', 'w', 'p', 'k', 'e']
    actual_variables = df['variable'].unique()
    
    print(f"有效变量: {valid_variables}")
    print(f"实际变量: {list(actual_variables)}")
    
    # 检查是否有无效的变量名
    invalid_vars = [var for var in actual_variables if var not in valid_variables]
    if invalid_vars:
        print(f"警告: 发现无效变量名: {invalid_vars}")
        return False
    
    # 检查数据长度一致性
    for _, row in df.iterrows():
        if not isinstance(row['data'], list) or len(row['data']) == 0:
            print(f"错误: 无效的数据数组 - {row['variable']}, is_element={row['is_element']}")
            return False
    
    print("数据验证通过")
    return True

def main(input_path=None):
    # 配置
    CONFIG = {
        "txt_output_dir": "Decoded_Data/",
        "postgres": {
            "HOST": "localhost",
            "PORT": "5432",
            "USER": "postgres",
            "PASSWORD": "123456",
            "DATABASE": "cae_data2"
        }
    }

    # 检查dat目录
    if input_path is not None:
        base_path = input_path
    else:
        base_path = '/data/gpudb/downloaded/'  # 根据你的结构调整
    
    print(f"开始在基础路径下搜索.dat文件: {base_path}")

    if not os.path.exists(base_path):
        print(f"错误: 路径不存在: {base_path}")
        return

    # 查找所有dat文件
    dat_files_info = find_dat_files(base_path)
    
    if not dat_files_info:
        print(f"在 {base_path} 下未找到.dat文件")
        return

    print(f"找到 {len(dat_files_info)} 个.dat文件")
    
    # 按船舶类型和尺度分组显示
    ship_groups = defaultdict(list)
    for info in dat_files_info:
        key = f"{info['ship_type']}_{info['scale']}"
        ship_groups[key].append(info['timestep'])
    
    for ship_key, timesteps in ship_groups.items():
        print(f"  {ship_key}: {len(timesteps)}个时间步, 时间步范围: {min(timesteps)}-{max(timesteps)}")

    # 创建数据库连接
    connection = create_postgres_connection(
        db_name=CONFIG["postgres"]["DATABASE"],
        db_user=CONFIG["postgres"]["USER"],
        db_password=CONFIG["postgres"]["PASSWORD"],
        db_host=CONFIG["postgres"]["HOST"],
        db_port=CONFIG["postgres"]["PORT"]
    )

    if not connection:
        print("无法连接到数据库")
        return

    try:
        # 创建表结构
        create_tables(connection)
        
        processed_count = 0
        for file_info in dat_files_info:
            file_path = file_info['file_path']
            ship_type = file_info['ship_type']
            scale = file_info['scale']
            timestep = file_info['timestep']
            
            print(f"\n处理文件: {file_path}")
            print(f"  船舶类型: {ship_type}, 尺度: {scale}, 时间步: {timestep}")

            try:
                # 解析dat文件
                data = CAE_Decoder(3)
                data.Decode_dat_file(file_path)

                if not data.Zones:
                    print(f"  警告: 文件 {file_path} 没有解析到区域数据")
                    continue

                fluid_zone = data.Zones[0]

                # 使用从路径提取的信息创建DataFrame
                dataframes = fluid_zone.to_dataframes(
                    ship_type=ship_type, 
                    scale=scale
                )

                # 插入数据到数据库
                if "cae_simulation_data" in dataframes:
                    df = dataframes["cae_simulation_data"]

                    if not validate_dataframe(df, file_path):
                        print(f"数据验证失败，跳过文件：{file_path}")
                        continue

                    print(f"  准备插入 {len(df)} 条记录")
                    
                    insert_cae_simulation_data(connection, df, "cae_simulation_data")
                    
                    # 验证数据插入
                    verify_data_insertion(connection, ship_type, timestep)
                    
                    processed_count += 1

            except Exception as e:
                print(f"  处理文件 {file_path} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n处理完成! 成功处理 {processed_count}/{len(dat_files_info)} 个文件")

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        close_postgres_connection(connection)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, 
                       default='/data/gpudb/downloaded/',
                       help='Input base path for .dat files')
    args = parser.parse_args()
    main(args.input_path)