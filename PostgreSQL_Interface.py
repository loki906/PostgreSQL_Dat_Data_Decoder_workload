import psycopg2
from psycopg2 import OperationalError
import numpy as np
import pandas as pd


def create_postgres_connection(db_name="cae_data2", db_user="postgres", db_password="123456", db_host="localhost",
                               db_port="5432"):
    """
    Create a connection to the PostgreSQL database.
    """
    connection = None
    try:
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection


def create_tables(connection):
    """
    创建表
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS cae_simulation_data (
        ship_type VARCHAR(50) NOT NULL,
        scale VARCHAR(50) NOT NULL,
        timestep INTEGER NOT NULL,
        variable VARCHAR(50) NOT NULL,
        is_element BOOLEAN NOT NULL,
        max_range INTEGER,
        data DOUBLE PRECISION[] NOT NULL,
        PRIMARY KEY (ship_type, scale, timestep, variable, is_element)
    );
    
    CREATE INDEX IF NOT EXISTS idx_ship_timestep ON cae_simulation_data(ship_type, timestep);
    CREATE INDEX IF NOT EXISTS idx_variable_search ON cae_simulation_data(variable, is_element);
    """
    
    try:
        cursor = connection.cursor()
        cursor.execute(create_table_sql)
        connection.commit()
        print("新表结构创建成功")
    except Exception as e:
        print(f"创建表结构时出错: {e}")
        connection.rollback()
    finally:
        if cursor:
            cursor.close()


def insert_cae_simulation_data(connection, df, table_name="cae_simulation_data"):
    """
    插入CAE数据
    """
    if not connection:
        print("No valid database connection")
        return

    columns = ["ship_type", "scale", "timestep", "variable", "is_element", "data"]
    
    # 确保列顺序正确
    try:
        df_reordered = df[columns]
    except Exception as e:
        print(f"DataFrame列不匹配: {e}")
        print(f"DataFrame实际列: {list(df.columns)}")
        return

    cols_str = ', '.join(columns)
    placeholders = ', '.join(['%s'] * len(columns))
    
    sql = f"""
    INSERT INTO {table_name} ({cols_str}) 
    VALUES ({placeholders}) 
    ON CONFLICT (ship_type, scale, timestep, variable, is_element) 
    DO UPDATE SET 
        data = EXCLUDED.data;
    """

    try:
        cursor = connection.cursor()
        
        data = []
        for row in df_reordered.values:
            row_converted = []
            for val in row:
                if isinstance(val, (list, np.ndarray)):
                    if isinstance(val, np.ndarray):
                        row_converted.append([float(x) for x in val])
                    else:
                        row_converted.append([float(x) for x in val])
                elif pd.isna(val) or val is None:
                    row_converted.append(None)
                elif isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    row_converted.append(int(val))
                elif isinstance(val, (np.floating, np.float64, np.float32, np.float16)):
                    row_converted.append(float(val))
                elif isinstance(val, (int, float, str, bool)):
                    row_converted.append(val)
                else:
                    print(f"警告: 遇到未知数据类型 {type(val)}，值：{val}")
                    row_converted.append(str(val))
            data.append(tuple(row_converted))

        # 调试信息：显示第一条数据
        if data:
            print(f"第一条插入数据预览:")
            for i, col in enumerate(columns):
                if col == "data":
                    print(f"  {col}: 数组长度={len(data[0][i])}, 前5个值={data[0][i][:5]}")
                else:
                    print(f"  {col}: {data[0][i]}")

        cursor.executemany(sql, data)
        connection.commit()
        print(f"成功插入/更新 {cursor.rowcount} 行数据到 {table_name}")
        
    except Exception as e:
        print(f"插入数据时出错: {e}")
        connection.rollback()
        import traceback
        traceback.print_exc()
    finally:
        if cursor:
            cursor.close()


# def update_max_range(connection, ship_type, scale, timestep, variable, is_element, max_range_value):
#     """
#     后续更新max_range值的函数
#     """
#     update_sql = """
#     UPDATE cae_simulation_data 
#     SET max_range = %s
#     WHERE ship_type = %s 
#       AND scale = %s 
#       AND timestep = %s 
#       AND variable = %s 
#       AND is_element = %s;
#     """
    
#     try:
#         cursor = connection.cursor()
#         cursor.execute(update_sql, (max_range_value, ship_type, scale, timestep, variable, is_element))
#         connection.commit()
#         print(f"成功更新 {variable} 的max_range为 {max_range_value}")
#     except Exception as e:
#         print(f"更新max_range时出错: {e}")
#         connection.rollback()
#     finally:
#         if cursor:
#             cursor.close()


# def batch_update_max_range(connection, max_range_data):
#     """
#     批量更新max_range值
#     max_range_data格式: [(ship_type, scale, timestep, variable, is_element, max_range_value), ...]
#     """
#     update_sql = """
#     UPDATE cae_simulation_data 
#     SET max_range = %s
#     WHERE ship_type = %s 
#       AND scale = %s 
#       AND timestep = %s 
#       AND variable = %s 
#       AND is_element = %s;
#     """
    
#     try:
#         cursor = connection.cursor()
#         cursor.executemany(update_sql, max_range_data)
#         connection.commit()
#         print(f"成功批量更新 {cursor.rowcount} 条记录的max_range")
#     except Exception as e:
#         print(f"批量更新max_range时出错: {e}")
#         connection.rollback()
#     finally:
#         if cursor:
#             cursor.close()


def close_postgres_connection(connection):
    """
    Close the PostgreSQL database connection.
    """
    if connection:
        connection.close()
        print("PostgreSQL connection closed")