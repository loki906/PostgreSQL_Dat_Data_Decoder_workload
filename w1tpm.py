import time
import random
import numpy as np
from pg_workload import Workload_Solver  
import vtk_workload as VTKAPI
import vtk

# 初始化
solver = Workload_Solver("/data/vtk_files/JBC_615k_GEO.vtk")
mesh = solver.geo_mesh

# 随机参数生成函数
def random_line_params():
    # 生成随机直线起点和终点（基于网格边界）
    bounds = mesh.GetBounds()
    start = [random.uniform(bounds[0], bounds[1]), random.uniform(bounds[2], bounds[3]), random.uniform(bounds[4], bounds[5])]
    end = [random.uniform(bounds[0], bounds[1]), random.uniform(bounds[2], bounds[3]), random.uniform(bounds[4], bounds[5])]
    return start, end

def random_plane_params():
    # 生成随机平面原点和法向量
    bounds = mesh.GetBounds()
    origin = [random.uniform(bounds[0], bounds[1]), random.uniform(bounds[2], bounds[3]), random.uniform(bounds[4], bounds[5])]
    normal = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
    return origin, normal

def random_variable():
    # 随机选择变量
    variables = solver.VARIABLES
    return random.choice(variables)

def random_time_step():
    time_step = solver.TIMESTEPS
    return random.choice(time_step)

# 单个事务函数
def execute_transaction():
    # 随机选择操作
    if random.choice([True, False]):  # 50% 直线或平面
        # 直线相交
        start, end = random_line_params()
        _, intersected_cells = VTKAPI.workload_line_interesction(mesh, start, end, tolerance=1e-6)
    else:
        # 平面相交
        origin, normal = random_plane_params()
        intersected_cells = VTKAPI.workload_plane_intersection(mesh, origin, normal)

    # 随机变量查询和聚合
    variable = random_variable()
    time_step = random_time_step()

     # 随机选择聚合类型
    aggregation_types = ['sum', 'avg', 'max', 'min']
    agg_type = random.choice(aggregation_types)

    if intersected_cells:
        # 查询变量值（返回DataFrame）
        data = solver.points_query(intersected_cells, variable, time_step)
        print(f"查询到 {len(data)} 个单元格的 {variable} 值（时间步 {time_step}）")
        # 聚合
        agg_funcs = {
            'sum': solver.aggregation_sum,
            'avg': solver.aggregation_avg,
            'max': solver.aggregation_max,
            'min': solver.aggregation_min
        }

        aggregate = agg_funcs[agg_type](data)
        print(f"聚合类型：{agg_type}，结果：{aggregate:.6f}\n")
        return aggregate
    return 0

# TPM测量
def measure_tpm(duration_seconds=60):
    start_time = time.time()
    transaction_count = 0
    while time.time() - start_time < duration_seconds:
        execute_transaction()
        transaction_count += 1
    tpm = transaction_count  # 直接返回事务数量（每分钟）
    return tpm

# 运行并打印结果
tpm = measure_tpm()  
print(f"TPM: {tpm:.2f} transactions per minute")