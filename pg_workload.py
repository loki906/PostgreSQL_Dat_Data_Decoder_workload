import psycopg2
from psycopg2 import OperationalError
import random
import numpy as np
import pandas as pd
import time
import vtk_workload as VTKAPI
import vtk
import matplotlib.pyplot as plt


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


class Workload_Solver():
    def __init__(self, geo_vtk_file, db_name="cae_data2", db_user="postgres", db_password="123456", db_host="localhost",
                 db_port="5432", scale="None"):
        # 建立PostgreSQL连接
        self.conn = create_postgres_connection(db_name, db_user, db_password, db_host, db_port)
        self.cursor = self.conn.cursor() if self.conn else None

        file_name = geo_vtk_file.split('/')[-1]
        parts = file_name.split('_')
        self.scale = parts[1] if len(parts) >= 2 else "None"
        print("scale: " + self.scale)

        # 加载网格几何结构
        print("Loading geometry mesh from VTK file...")
        print("Loading file: " + geo_vtk_file)
        print("--------------------------------------------")
        print("The info of the mesh is as below:")
        self.geo_mesh = VTKAPI.load_unstructured_grid_from_vtk_file(geo_vtk_file)
        self.SHIP_TYPE = geo_vtk_file.split('_GEO')[0].split('/')[-1].split('_')[0]
        print("SHIP TYPE: " + self.SHIP_TYPE)
        self.TIMESTEPS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.VARIABLES = ['u', 'v', 'w', 'p', 'k', 'e']
        print("--------------------------------------------")

    def point_query(self, element_index, variable_name, time_step):
        """查询单个元素的变量值"""
        # is_element=True表示查询元素数据
        sql = """
            SELECT data[%s] FROM cae_simulation_data 
            WHERE ship_type = %s 
              AND scale = %s 
              AND timestep = %s 
              AND variable = %s 
              AND is_element = 't';
        """
        # PostgreSQL数组索引从1开始，因此element_index+1
        params = (element_index + 1, self.SHIP_TYPE, self.scale, time_step, variable_name)
        print(f"SQL: {sql} with params {params}")
        self.cursor.execute(sql, params)
        result = self.cursor.fetchone()
        # 构造DataFrame（包含'value'列）
        return pd.DataFrame([result[0]], columns=['value']) if result else pd.DataFrame(columns=['value'])

    def points_query(self, element_indices, variable_name, time_step):
        """查询多个元素的变量值"""
        # 将索引转换为PostgreSQL数组索引（+1）
        pg_indices = [i + 1 for i in element_indices]
        # 使用ANY函数查询数组中多个索引的值
        sql = """
            SELECT unnest(ARRAY[%s]) AS element_index, 
                   data[unnest(ARRAY[%s])] AS value 
            FROM cae_simulation_data 
            WHERE ship_type = %s 
              AND scale = %s 
              AND timestep = %s 
              AND variable = %s 
              AND is_element = 't';
        """
        params = (element_indices, pg_indices, self.SHIP_TYPE, self.scale, time_step, variable_name)
        print(f"SQL: {sql} with params {params}")
        self.cursor.execute(sql, params)
        # 转换结果为DataFrame，包含元素索引和值
        columns = ['element_id', 'value']
        return pd.DataFrame(self.cursor.fetchall(), columns=columns)

    def range_query_var(self, variable_name, lower_bound, upper_bound, time_step):
        """根据变量值范围查询元素ID"""
        # 先获取符合条件的数组，再通过索引定位符合条件的元素
        sql = """
            SELECT idx AS element_id 
            FROM cae_simulation_data,
                 generate_subscripts(data, 1) AS idx 
            WHERE ship_type = %s 
              AND scale = %s 
              AND timestep = %s 
              AND variable = %s 
              AND is_element = TRUE 
              AND data[idx] BETWEEN %s AND %s;
        """
        params = (self.SHIP_TYPE, self.scale, time_step, variable_name, lower_bound, upper_bound)
        print(f"SQL: {sql} with params {params}")
        self.cursor.execute(sql, params)
        # PostgreSQL数组索引从1开始，转换为原代码的0基索引
        return [row[0] - 1 for row in self.cursor.fetchall()]

    def range_query_coor(self, lower_bound: list, upper_bound: list):
        """根据坐标范围查询元素ID"""
        X_lower, Y_lower, Z_lower = lower_bound
        X_upper, Y_upper, Z_upper = upper_bound
        sql = """
            SELECT idx AS element_id 
            FROM (
                SELECT data AS X_data FROM cae_simulation_data 
                WHERE ship_type = %s AND scale = %s AND variable = 'X' AND is_element = TRUE
            ) x,
            (
                SELECT data AS Y_data FROM cae_simulation_data 
                WHERE ship_type = %s AND scale = %s AND variable = 'Y' AND is_element = TRUE
            ) y,
            (
                SELECT data AS Z_data FROM cae_simulation_data 
                WHERE ship_type = %s AND scale = %s AND variable = 'Z' AND is_element = TRUE
            ) z,
            generate_subscripts(x.X_data, 1) AS idx 
            WHERE x.X_data[idx] BETWEEN %s AND %s
              AND y.Y_data[idx] BETWEEN %s AND %s
              AND z.Z_data[idx] BETWEEN %s AND %s;
        """
        params = (
            self.SHIP_TYPE, self.scale,
            self.SHIP_TYPE, self.scale,
            self.SHIP_TYPE, self.scale,
            X_lower, X_upper, Y_lower, Y_upper, Z_lower, Z_upper
        )
        print(f"SQL: {sql} with params {params}")
        self.cursor.execute(sql, params)
        # 转换为0基索引
        return [row[0] - 1 for row in self.cursor.fetchall()]

    # 聚合函数保持不变（因为DataFrame格式兼容）
    def aggregation_sum(self, query_result: pd.DataFrame):
        query_result["value"] = pd.to_numeric(query_result["value"], errors='coerce', downcast='float')
        return query_result["value"].sum()

    def aggregation_avg(self, query_result: pd.DataFrame):
        query_result["value"] = pd.to_numeric(query_result["value"], errors='coerce', downcast='float')
        return query_result["value"].mean()

    def aggregation_max(self, query_result: pd.DataFrame):
        query_result["value"] = pd.to_numeric(query_result["value"], errors='coerce', downcast='float')
        return query_result["value"].max()

    def aggregation_min(self, query_result: pd.DataFrame):
        query_result["value"] = pd.to_numeric(query_result["value"], errors='coerce', downcast='float')
        return query_result["value"].min()

    def solve_1(self, line_start: list, line_end: list, variable: str, time_step: int):
        mesh = self.geo_mesh
        line_start = [0.01, 0.02, -0.1]
        line_end = np.random.random(3)
        _, intersected_cells_line, intersected_time = VTKAPI.workload_line_interesction(mesh, line_start, line_end, 0.001)
        
        query_start = time.time()
        query_result = self.points_query(intersected_cells_line, variable, time_step)
        query_end = time.time()
        query_time = query_end - query_start
        
        aggregation_start = time.time()
        result_sum = self.aggregation_sum(query_result)
        result_max = self.aggregation_max(query_result)
        result_min = self.aggregation_min(query_result)
        result_avg = self.aggregation_avg(query_result)
        aggregation_end = time.time()
        aggregation_time = aggregation_end - aggregation_start
        
        return intersected_time, query_time, aggregation_time

    def solve_4(self, init_coor, time_step_interval):
        mesh = self.geo_mesh
        result_pathline = []
        current_point = np.array(init_coor)
        result_pathline.append(init_coor)
        
        for step in self.TIMESTEPS:
            cell_id = VTKAPI.workload_locate_cells_by_point(mesh, current_point)
            u = self.point_query(cell_id, "u", step)['value'][0]
            v = self.point_query(cell_id, "v", step)['value'][0]
            w = self.point_query(cell_id, "w", step)['value'][0]
            
            velocity = np.array([float(u), float(v), float(w)])
            next_point = current_point + velocity * time_step_interval
            result_pathline.append(next_point)
            current_point = next_point
        
        return np.array(result_pathline)

    def solve_5(self, init_coor, time_step_interval, time_step):
        mesh = self.geo_mesh
        Streamline = []
        current_point = np.array(init_coor)
        Streamline.append(init_coor)
        
        while True:
            cell_id = VTKAPI.workload_locate_cells_by_point(mesh, current_point)
            u = self.point_query(cell_id, "u", time_step)['value'][0]
            v = self.point_query(cell_id, "v", time_step)['value'][0]
            w = self.point_query(cell_id, "w", time_step)['value'][0]
            
            velocity = np.array([float(u), float(v), float(w)])
            next_point = current_point + velocity * time_step_interval
            current_point = next_point
            
            if VTKAPI.workload_locate_cells_by_point(mesh, current_point) == -1:
                break
            else:
                Streamline.append(current_point)
        
        return np.array(Streamline)

    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("PostgreSQL connection closed")