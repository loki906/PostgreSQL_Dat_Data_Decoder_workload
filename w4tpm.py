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

class W4_Workload_Solver:
    def __init__(self, vtk_file_path):
        self.geo_mesh = VTKAPI.load_unstructured_grid_from_vtk_file(vtk_file_path)
        self.connection = create_postgres_connection()
        self.TIMESTEPS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        self.VARIABLES = ['u', 'v', 'w']
        
        file_name = vtk_file_path.split('/')[-1]
        parts = file_name.split('_')
        self.scale = parts[1] if len(parts) >= 2 else "None"
        self.SHIP_TYPE = vtk_file_path.split('_GEO')[0].split('/')[-1].split('_')[0]
        
        debug_print(f"W4求解器初始化完成: {len(self.VARIABLES)}个速度变量, {len(self.TIMESTEPS)}个时间步")
        debug_print(f"船舶类型: {self.SHIP_TYPE}, 尺度: {self.scale}")

    def point_query(self, cell_id, variable_name, time_step):
        """查询单个单元的变量值"""
        debug_print(f"点查询: cell_id={cell_id}, variable={variable_name}, timestep={time_step}", "W4", "数据库查询")
        
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            # PostgreSQL数组索引从1开始
            sql = """
            SELECT data[%s] as value 
            FROM cae_simulation_data 
            WHERE ship_type = %s 
              AND scale = %s 
              AND timestep = %s 
              AND variable = %s 
              AND is_element = true;
            """
            
            params = (cell_id + 1, self.SHIP_TYPE, self.scale, time_step, variable_name)
            cursor.execute(sql, params)
            result = cursor.fetchone()
            
            cursor.close()
            
            if result and result['value'] is not None:
                debug_print(f"查询成功: {result['value']}", "W4", "数据库查询")
                return float(result['value'])
            else:
                debug_print(f"查询失败: 未找到数据", "W4", "数据库查询")
                return 0.0
                
        except Exception as e:
            print(f"数据库查询错误: {e}")
            return 0.0

    def solve_pathline(self, init_coor, time_step_interval=0.1):
        """执行Pathline分析"""
        mesh = self.geo_mesh
        result_pathline = []
        current_point = np.array(init_coor)
        
        debug_print(f"开始Pathline分析: 初始位置={init_coor}, 时间步长={time_step_interval}", "W4", "开始")
        
        result_pathline.append(init_coor)
        
        for step in self.TIMESTEPS:
            # 定位当前点所在的单元
            cell_id = VTKAPI.workload_locate_cells_by_point(mesh, current_point)
            debug_print(f"时间步{step}: 当前位置={current_point}, 所在单元={cell_id}", "W4", "几何定位")
            
            if cell_id == -1:
                debug_print(f"时间步{step}: 粒子离开网格，停止追踪", "W4", "边界检查")
                break
            
            # 查询速度分量
            u = self.point_query(cell_id, "u", step)
            v = self.point_query(cell_id, "v", step)
            w = self.point_query(cell_id, "w", step)
            
            debug_print(f"时间步{step}: 速度分量 u={u:.6f}, v={v:.6f}, w={w:.6f}", "W4", "速度查询")
            
            velocity = np.array([u, v, w])
            
            # 更新位置
            next_point = current_point + velocity * time_step_interval
            debug_print(f"时间步{step}: 下一位置={next_point}", "W4", "位置更新")
            
            # 检查下一位置是否在网格内
            next_cell_id = VTKAPI.workload_locate_cells_by_point(mesh, next_point)
            if next_cell_id == -1:
                debug_print(f"时间步{step}: 下一位置离开网格，停止追踪", "W4", "边界检查")
                break
            
            result_pathline.append(next_point)
            current_point = next_point
        
        pathline_array = np.array(result_pathline)
        debug_print(f"Pathline分析完成: 生成{len(pathline_array)}个轨迹点", "W4", "完成")
        return pathline_array

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            debug_print("数据库连接已关闭", "W4", "清理")

def random_point_in_mesh(mesh):
    """在网格边界内生成随机点"""
    bounds = mesh.GetBounds()
    point = [
        random.uniform(bounds[0], bounds[1]),
        random.uniform(bounds[2], bounds[3]),
        random.uniform(bounds[4], bounds[5])
    ]
    debug_print(f"生成随机点: {point}", "W4", "参数生成")
    return point

def execute_w4_transaction(solver):
    """执行单个W4事务"""
    try:
        # 生成随机初始位置
        init_point = random_point_in_mesh(solver.geo_mesh)
        
        # 随机时间步长 (0.05到0.2之间)
        time_step_interval = random.uniform(0.05, 0.2)
        
        debug_print(f"执行W4事务: 初始点={init_point}, 步长={time_step_interval:.3f}", "W4", "事务开始")
        
        # 执行Pathline分析
        pathline = solver.solve_pathline(init_point, time_step_interval)
        
        # 返回轨迹长度作为结果
        trajectory_length = len(pathline)
        debug_print(f"W4事务完成: 轨迹长度={trajectory_length}", "W4", "事务完成")
        
        return trajectory_length > 1  # 成功条件：至少生成2个点（起点+至少一个移动点）
        
    except Exception as e:
        print(f"W4事务执行错误: {e}")
        return False

def measure_w4_tpm(solver, duration_seconds=60):
    """测量W4的TPM"""
    print(f"\n开始测量W4 Pathline Analysis的TPM...")
    
    start_time = time.time()
    transaction_count = 0
    successful_count = 0
    
    # 创建进度条
    with tqdm(total=duration_seconds, desc="W4进度", unit="秒") as pbar:
        while time.time() - start_time < duration_seconds:
            try:
                success = execute_w4_transaction(solver)
                
                if success:
                    successful_count += 1
                transaction_count += 1
                
                # 更新进度条
                current_time = time.time()
                elapsed = current_time - start_time
                pbar.update(int(elapsed - pbar.n))
                pbar.set_postfix({
                    '事务数': transaction_count,
                    '成功率': f'{successful_count/transaction_count*100:.1f}%' if transaction_count > 0 else '0%'
                })
                
            except Exception as e:
                print(f"\nW4事务执行错误: {e}")
                transaction_count += 1
                continue
    
    elapsed_time = time.time() - start_time
    tpm = (successful_count / elapsed_time) * 60
    
    print(f"\nW4 Pathline Analysis TPM测试结果:")
    print(f"  总事务数: {transaction_count}")
    print(f"  成功事务数: {successful_count}")
    print(f"  测试时间: {elapsed_time:.2f}秒")
    print(f"  TPM: {tpm:.2f} transactions per minute")
    print(f"  成功率: {successful_count/transaction_count*100:.1f}%" if transaction_count > 0 else "  成功率: 0%")
    
    return tpm

def check_database_connectivity(solver):
    """检查数据库连接和数据可用性"""
    try:
        cursor = solver.connection.cursor(cursor_factory=RealDictCursor)
        
        # 检查是否有W4所需的速度数据
        cursor.execute("""
        SELECT variable, timestep, array_length(data, 1) as data_length
        FROM cae_simulation_data 
        WHERE ship_type = %s AND scale = %s AND variable IN ('u', 'v', 'w') AND is_element = true
        LIMIT 3
        """, (solver.SHIP_TYPE, solver.scale))
        
        rows = cursor.fetchall()
        print("W4所需速度数据检查:")
        if rows:
            for row in rows:
                print(f"  变量: {row['variable']}, 时间步: {row['timestep']}, 数据长度: {row['data_length']}")
        else:
            print("  警告: 未找到W4所需的速度数据!")
        
        cursor.close()
        return len(rows) > 0
        
    except Exception as e:
        print(f"数据库连接检查错误: {e}")
        return False

def main():
    # 设置信号处理
    signal.signal(signal.SIGALRM, timeout_handler)
    
    print("开始W4 Pathline Analysis TPM性能测试")
    print("=" * 50)
    
    # 初始化W4求解器
    vtk_file_path = "/data/vtk_files/JBC_615k_GEO.vtk"  # 根据实际情况调整路径
    solver = W4_Workload_Solver(vtk_file_path)
    
    # 检查数据库连接和数据
    print("检查数据库连接和数据...")
    data_available = check_database_connectivity(solver)
    
    if not data_available:
        print("错误: 数据库中没有W4所需的速度数据，无法进行测试!")
        return
    
    print("\n" + "=" * 50)
    print("开始W4性能测试 (60秒)")
    print("=" * 50)
    
    try:
        # 测量W4 TPM
        w4_tpm = measure_w4_tpm(solver, 60)
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        return
    except Exception as e:
        print(f"\n\n测试过程中发生错误: {e}")
        return
    finally:
        # 清理资源
        solver.close()
    
    # 输出结果
    print("\n" + "=" * 50)
    print("W4 TPM性能测试汇总")
    print("=" * 50)
    print(f" Pathline Analysis TPM: {w4_tpm:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--vtk_file", type=str, default="/data/vtk_files/JBC_615k_GEO.vtk", 
                       help="VTK几何文件路径")
    args = parser.parse_args()
    
    # 设置调试模式
    DEBUG_MODE = args.debug
    if DEBUG_MODE:
        print("调试模式已启用")
    
    main()