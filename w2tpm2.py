import time
import random
import numpy as np
from PostgreSQL_Interface import create_postgres_connection
import vtk_workload as VTKAPI
import vtk
import psycopg2
from psycopg2.extras import RealDictCursor
from tqdm import tqdm
import threading
import signal
import sys
import os

# 调试开关
DEBUG_MODE = False

def debug_print(message, transaction_type="", step=""):
    """调试信息输出函数"""
    if DEBUG_MODE:
        prefix = f"[DEBUG {transaction_type}] " if transaction_type else "[DEBUG] "
        step_prefix = f"[{step}] " if step else ""
        print(f"{prefix}{step_prefix}{message}")

class TimeoutException(Exception):
    """自定义超时异常"""
    pass

def timeout_handler(signum, frame):
    """超时处理函数"""
    raise TimeoutException("查询超时")

class Workload_Solver:
    def __init__(self, vtk_file_path):
        self.geo_mesh = VTKAPI.load_unstructured_grid_from_vtk_file(vtk_file_path)
        self.connection = create_postgres_connection()
        self.VARIABLES = ['u', 'v', 'w', 'p', 'k', 'e']  # 支持的物理变量
        self.TIMESTEPS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # 固定时间步集合
        self.ship_type = "JBC"  # 船舶类型（可根据需求扩展）
        self.scale = "615k"     # 船舶尺度（可根据需求扩展）
        debug_print(f"初始化完成: {len(self.VARIABLES)}个变量, {len(self.TIMESTEPS)}个时间步")
        debug_print(f"船舶类型: {self.ship_type}, 尺度: {self.scale}")

    def get_roi_cells_by_range(self, lower_bound, upper_bound):
        debug_print(f"空间范围查询: 下界={lower_bound}, 上界={upper_bound}", "时间聚合", "ROI查询")
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            sql = """
                SELECT idx AS element_id 
                FROM (
                    SELECT data AS X_data FROM cae_simulation_data 
                    WHERE ship_type = %s AND scale = %s AND variable = 'X' AND is_element = true
                ) x,
                (
                    SELECT data AS Y_data FROM cae_simulation_data 
                    WHERE ship_type = %s AND scale = %s AND variable = 'Y' AND is_element = true
                ) y,
                (
                    SELECT data AS Z_data FROM cae_simulation_data 
                    WHERE ship_type = %s AND scale = %s AND variable = 'Z' AND is_element = true
                ) z,
                generate_subscripts(x.X_data, 1) AS idx 
                WHERE x.X_data[idx] BETWEEN %s AND %s
                  AND y.Y_data[idx] BETWEEN %s AND %s
                  AND z.Z_data[idx] BETWEEN %s AND %s;
            """
            params = (
                self.ship_type, self.scale,
                self.ship_type, self.scale,
                self.ship_type, self.scale,
                lower_bound[0], upper_bound[0],
                lower_bound[1], upper_bound[1],
                lower_bound[2], upper_bound[2]
            )
            cursor.execute(sql, params)
            results = cursor.fetchall()
            cell_ids = [row['element_id'] - 1 for row in results]  # 转0基索引
            cursor.close()
            debug_print(f"ROI查询结果: {len(cell_ids)}个单元格", "时间聚合", "ROI查询")
            return cell_ids
        except Exception as e:
            print(f"ROI查询错误: {e}")
            return []

    def query_variable_by_cells_timesteps(self, cell_ids, variable, timesteps):
        """查询指定单元格在多个时间步的变量值"""
        debug_print(f"多时间步查询: {len(cell_ids)}个单元格, 变量={variable}, 时间步={timesteps}", "时间聚合", "多步查询")
        if not cell_ids or not timesteps:
            return []
        
        all_values = []
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            for timestep in timesteps:
                sql = """
                    SELECT data
                    FROM cae_simulation_data 
                    WHERE ship_type = %s AND scale = %s AND timestep = %s 
                      AND variable = %s AND is_element = true
                """
                cursor.execute(sql, (self.ship_type, self.scale, timestep, variable))
                row = cursor.fetchone()
                if row and row['data']:
                    # 提取目标单元格的值
                    cell_values = [row['data'][cid] for cid in cell_ids if cid < len(row['data'])]
                    all_values.extend(cell_values)
            cursor.close()
            debug_print(f"多时间步查询结果: 共{len(all_values)}个数值", "时间聚合", "多步查询")
            return all_values
        except Exception as e:
            print(f"多时间步查询错误: {e}")
            return []

    # 聚合函数
    def aggregation_sum(self, data):
        debug_print(f"执行SUM聚合: 数据量={len(data)}", "时间聚合", "聚合运算")
        return sum(data) if data else 0

    def aggregation_avg(self, data):
        debug_print(f"执行AVG聚合: 数据量={len(data)}", "时间聚合", "聚合运算")
        return sum(data) / len(data) if data else 0

    def aggregation_max(self, data):
        debug_print(f"执行MAX聚合: 数据量={len(data)}", "时间聚合", "聚合运算")
        return max(data) if data else 0

    def aggregation_min(self, data):
        debug_print(f"执行MIN聚合: 数据量={len(data)}", "时间聚合", "聚合运算")
        return min(data) if data else 0

def random_roi_range(mesh, range_scale=0.1):
    """生成随机ROI空间范围（基于网格边界的子范围）"""
    bounds = mesh.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    debug_print(f"网格边界: {bounds}", "时间聚合", "ROI生成") 
    # 计算边界长度，按比例生成子范围
    x_len = bounds[1] - bounds[0]
    y_len = bounds[3] - bounds[2]
    z_len = bounds[5] - bounds[4]
    
    # 随机生成范围起点（确保不超出边界）
    x_start = random.uniform(bounds[0], bounds[1] - x_len * range_scale)
    y_start = random.uniform(bounds[2], bounds[3] - y_len * range_scale)
    z_start = random.uniform(bounds[4], bounds[5] - z_len * range_scale)
    
    # 生成范围终点
    lower_bound = [x_start, y_start, z_start]
    upper_bound = [
        x_start + x_len * range_scale,
        y_start + y_len * range_scale,
        z_start + z_len * range_scale
    ]
    debug_print(f"生成随机ROI: 下界={lower_bound}, 上界={upper_bound}", "时间聚合", "ROI生成")
    return lower_bound, upper_bound

def execute_temporal_agg_transaction(solver):
    """执行时间聚合事务（W2查询）"""
    # 1. 生成随机ROI
    lower_bound, upper_bound = random_roi_range(solver.geo_mesh)
    
    # 2. 查询ROI内的单元格
    cell_ids = solver.get_roi_cells_by_range(lower_bound, upper_bound)
    if not cell_ids:
        debug_print("ROI内无单元格，事务失败", "时间聚合", "事务结束")
        return False
    
    # 3. 随机选择目标变量和时间步子集（避免全量时间步耗时过长）
    variable = random.choice(solver.VARIABLES)
    timesteps = random.sample(solver.TIMESTEPS, k=min(5, len(solver.TIMESTEPS)))  # 随机选5个时间步
    
    # 4. 多时间步变量查询
    data = solver.query_variable_by_cells_timesteps(cell_ids, variable, timesteps)
    if not data:
        debug_print("未查询到有效数据，事务失败", "时间聚合", "事务结束")
        return False
    
    # 5. 执行时间聚合（随机选择一种聚合类型）
    agg_type = random.choice(['sum', 'avg', 'max', 'min'])
    agg_funcs = {
        'sum': solver.aggregation_sum,
        'avg': solver.aggregation_avg,
        'max': solver.aggregation_max,
        'min': solver.aggregation_min
    }
    result = agg_funcs[agg_type](data)
    debug_print(f"时间聚合事务成功: 变量={variable}, 时间步={timesteps}, 聚合类型={agg_type}, 结果={result}", "时间聚合", "事务结束")
    return True

def measure_tpm(solver, duration_seconds=60, transaction_type="时间聚合"):
    """测量时间聚合事务的TPM（修复进度条：按真实时间推进）"""
    print(f"\n开始测量{transaction_type}的TPM...")
    start_time = time.time()
    transaction_count = 0
    successful_count = 0
    timeout_count = 0
    is_running = False

    # 创建进度条（总时长为duration_seconds秒）
    pbar = tqdm(total=duration_seconds, desc=f"{transaction_type}进度", unit="秒")

    # 定义总时间监控线程：到达指定时间后强制终止程序
    def time_monitor():
        nonlocal is_running
        # 等待指定时长
        time.sleep(duration_seconds)
        # 时间到，检查是否有事务在运行
        print(f"\n{duration_seconds}秒时间已到，检测到{'有事务正在运行' if is_running else '无事务运行'}")
        # 无论是否有事务，强制输出结果并终止程序
        # finalize_and_exit()
        elapsed_time = time.time() - start_time
        tpm = (successful_count / elapsed_time) * 60 if elapsed_time > 0 else 0
        print(f"\n{transaction_type} TPM测试结果:")
        print(f"  总事务数: {transaction_count}")
        print(f"  成功事务数: {successful_count}")
        print(f"  超时事务数: {timeout_count}")
        print(f"  测试时间: {elapsed_time:.2f}秒")
        print(f"  TPM: {tpm:.2f} transactions per minute")
        print(f"  成功率: {successful_count / transaction_count * 100:.1f}%" if transaction_count > 0 else "  成功率: 0%")
        
        os._exit(0)  # 强制退出程序

        # 关闭数据库连接
        if solver.connection:
            solver.connection.close()
            print("数据库连接已关闭")
        
        print("时间到，强制终止程序")
        sys.exit(0)  # 强制退出程序

    # 启动总时间监控线程
    monitor_thread = threading.Thread(target=time_monitor)
    monitor_thread.daemon = False  # 设为非守护线程，确保能执行终止逻辑

    # 定义进度条自动更新线程（每秒更新1格）
    def update_progress_bar():
        while time.time() - start_time < duration_seconds:
            time.sleep(1)  # 每秒更新一次
            elapsed = time.time() - start_time
            # 计算需要更新的秒数（避免超进度条总长度）
            update_steps = min(int(elapsed) - pbar.n, duration_seconds - pbar.n)
            if update_steps > 0:
                pbar.update(update_steps)
        # 测试结束后，将进度条拉满
        pbar.update(duration_seconds - pbar.n)

    # 启动进度条更新线程（守护线程，主程序结束后自动退出）
    progress_thread = threading.Thread(target=update_progress_bar, daemon=True)
    progress_thread.start()
    monitor_thread = threading.Thread(target=time_monitor)
    monitor_thread.start()

    # 执行事务循环（核心逻辑不变，只改进度条更新方式）
            # # 关闭数据库连接（关键资源释放）
            # if solver.connection:
            #     solver.connection.close()
            #     print("数据库连接已关闭")
            
            # # 终止整个程序
            # print("测试时间结束，程序退出")
            # sys.exit(0)  # 强制退出程序
    while time.time() - start_time < duration_seconds:
        is_running = True
        try:
            print(f"\n正在执行第{transaction_count+1}个事务...")
            success = execute_temporal_agg_transaction(solver)

            if success:
                successful_count += 1
            transaction_count += 1

            # 实时更新进度条的“后缀信息”（事务数、成功率）
            pbar.set_postfix({
                '事务数': transaction_count,
                '成功率': f'{successful_count / transaction_count * 100:.1f}%',
                '超时数': timeout_count
            })

        except TimeoutException:
            timeout_count += 1
            transaction_count += 1
            pbar.set_postfix({
                '事务数': transaction_count,
                '成功率': f'{successful_count / transaction_count * 100:.1f}%',
                '超时数': timeout_count
            })
            print(f"第{transaction_count}个事务超时，跳过")

        except Exception as e:
            print(f"\n{transaction_type}事务执行错误: {e}")
            transaction_count += 1
            pbar.set_postfix({
                '事务数': transaction_count,
                '成功率': f'{successful_count / transaction_count * 100:.1f}%',
                '超时数': timeout_count
            })
        finally:
            is_running = False  # 标记事务结束

    # 正常结束时的清理
    finalize_and_exit()
    return tpm

def check_database_connection(solver):
    """检查数据库连接和数据"""
    try:
        cursor = solver.connection.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT ship_type, scale, timestep, variable, array_length(data, 1) as data_length
            FROM cae_simulation_data 
            WHERE ship_type = %s AND scale = %s
            LIMIT 3
        """, (solver.ship_type, solver.scale))
        rows = cursor.fetchall()
        print("数据库数据预览:")
        for row in rows:
            print(f"  {row}")
        cursor.close()
    except Exception as e:
        print(f"数据库检查错误: {e}")

def main():
    signal.signal(signal.SIGALRM, timeout_handler)
    print("=" * 50)
    print("W2 - 时间聚合查询 TPM 性能测试")
    print("=" * 50)

    # 初始化求解器（需替换为实际VTK文件路径）
    vtk_file_path = "/data/vtk_files/JBC_615k_GEO.vtk"
    solver = Workload_Solver(vtk_file_path)

    # 检查数据库连接
    print("\n检查数据库连接和数据...")
    check_database_connection(solver)

    # 开始TPM测试（默认60秒）
    print("\n" + "=" * 50)
    print(f"开始{60}秒性能测试...")
    print("=" * 50)
    try:
        tpm = measure_tpm(solver, duration_seconds=60)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        return
    except Exception as e:
        print(f"\n\n测试错误: {e}")
        return

    # 输出最终结果
    print("\n" + "=" * 50)
    print("TPM测试最终结果")
    print("=" * 50)
    print(f"时间聚合查询 TPM: {tpm:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    main()