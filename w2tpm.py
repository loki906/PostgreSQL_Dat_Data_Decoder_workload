import time
import random
import numpy as np
from pg_workload import Workload_Solver
import vtk_workload as VTKAPI
import vtk

# 初始化Workload_Solver（加载VTK网格和数据库连接）
solver = Workload_Solver("/data/vtk_files/JBC_615k_GEO.vtk")
mesh = solver.geo_mesh

# 生成随机ROI（基于网格边界的空间范围）
def random_roi():
    bounds = mesh.GetBounds()  # 获取网格x/y/z轴的边界范围
    # 随机生成ROI的下界和上界（确保下界<上界）
    x_lower = random.uniform(bounds[0], bounds[1] - 0.1)
    x_upper = random.uniform(x_lower + 0.1, bounds[1])
    y_lower = random.uniform(bounds[2], bounds[3] - 0.1)
    y_upper = random.uniform(y_lower + 0.1, bounds[3])
    z_lower = random.uniform(bounds[4], bounds[5] - 0.1)
    z_upper = random.uniform(z_lower + 0.1, bounds[5])
    return [x_lower, y_lower, z_lower], [x_upper, y_upper, z_upper]

# 随机选择变量（与solver中定义的VARIABLES一致）
def random_variable():
    return random.choice(solver.VARIABLES)

# 执行单个Workload2事务
def execute_workload2():
    # 步骤1：生成随机ROI和变量
    roi_lower, roi_upper = random_roi()
    variable = random_variable()
    timesteps = solver.TIMESTEPS  # 所有待遍历的时间步（[2,4,...,20]）
    
    try:
        # 步骤2：范围查询ROI内的单元格ID
        print(f"ROI范围：x[{roi_lower[0]:.2f},{roi_upper[0]:.2f}], y[{roi_lower[1]:.2f},{roi_upper[1]:.2f}], z[{roi_lower[2]:.2f},{roi_upper[2]:.2f}]")
        roi_cell_ids = solver.range_query_coor(roi_lower, roi_upper)
        
        if not roi_cell_ids:
            print("ROI内无匹配单元格，跳过本次事务")
            return 0
        
        # 步骤3：遍历所有时间步，查询并收集数据
        result_set = []
        for t in timesteps:
            # 查询当前时间步下ROI内所有单元格的变量值
            cell_data = solver.points_query(roi_cell_ids, variable, t)
            if not cell_data.empty:
                # 转换为数值类型并收集有效数据
                cell_data["value"] = pd.to_numeric(cell_data["value"], errors='coerce')
                valid_values = cell_data["value"].dropna().tolist()
                result_set.extend(valid_values)
        
        # 步骤4：时间聚合计算（平均、最大、最小）
        if result_set:
            avg_val = np.mean(result_set)
            max_val = np.max(result_set)
            min_val = np.min(result_set)
            print(f"变量{variable}时间聚合结果：平均={avg_val:.6f}, 最大={max_val:.6f}, 最小={min_val:.6f}")
            print(f"参与聚合的数据总量：{len(result_set)}个\n")
        else:
            print("所有时间步均无有效数据，聚合结果为空\n")
        
        return 1  # 事务执行成功计数
    except Exception as e:
        print(f"Workload2执行出错：{e}")
        return 0

# 测量Workload2的TPM（每分钟事务数）
def measure_workload2_tpm(duration_seconds=60):
    start_time = time.time()
    transaction_count = 0
    
    while time.time() - start_time < duration_seconds:
        success = execute_workload2()
        transaction_count += success  # 仅计数成功执行的事务
    
    tpm = transaction_count  # 1分钟内的事务数
    return tpm

# 运行TPM测量并输出结果
if __name__ == "__main__":
    print("开始测量Workload2的TPM（时间聚合事务/分钟）...")
    workload2_tpm = measure_workload2_tpm()
    print(f"\nWorkload2 TPM结果：{workload2_tpm} transactions per minute")