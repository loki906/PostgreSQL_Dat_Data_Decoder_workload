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
        self.VARIABLES = ['u', 'v', 'w', 'p', 'k', 'e']
        self.TIMESTEPS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        
    def points_query(self, cell_ids, variable, timestep):
        """查询指定单元在特定时间步的变量值"""
        if not cell_ids:
            return []
            
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT data
            FROM cae_simulation_data 
            WHERE ship_type = %s AND scale = %s AND timestep = %s AND variable = %s AND is_element = true
            AND array_length(data, 1) >= %s
            """
            
            max_cell_id = max(cell_ids)
            required_length = max_cell_id + 1
            
            cursor.execute(query, ('JBC', '615k', timestep, variable, required_length))
            row = cursor.fetchone()
            
            results = []
            if row and row['data']:
                data_array = row['data']
                for cell_id in cell_ids:
                    if cell_id < len(data_array):
                        results.append(data_array[cell_id])
            
            cursor.close()
            return results
            
        except Exception as e:
            print(f"查询错误: {e}")
            return []
    
    def aggregation_sum(self, data):
        result = sum(data) if data else 0
        return result
    
    def aggregation_avg(self, data):
        result = sum(data) / len(data) if data else 0
        return result
    
    def aggregation_max(self, data):
        result = max(data) if data else 0
        return result
    
    def aggregation_min(self, data):
        result = min(data) if data else 0
        return result

def random_line_params(mesh):
    """生成随机直线参数"""
    bounds = mesh.GetBounds()
    start = [
        random.uniform(bounds[0], bounds[1]),
        random.uniform(bounds[2], bounds[3]), 
        random.uniform(bounds[4], bounds[5])
    ]
    end = [
        random.uniform(bounds[0], bounds[1]),
        random.uniform(bounds[2], bounds[3]),
        random.uniform(bounds[4], bounds[5])
    ]
    return start, end

def random_plane_params(mesh):
    """生成随机平面参数"""
    bounds = mesh.GetBounds()
    origin = [
        random.uniform(bounds[0], bounds[1]),
        random.uniform(bounds[2], bounds[3]),
        random.uniform(bounds[4], bounds[5])
    ]
    normal = [random.uniform(-1, 1) for _ in range(3)]
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = [n/norm for n in normal]
    return origin, normal

def random_point_params(mesh):
    """生成随机点参数"""
    bounds = mesh.GetBounds()
    point = [
        random.uniform(bounds[0], bounds[1]),
        random.uniform(bounds[2], bounds[3]),
        random.uniform(bounds[4], bounds[5])
    ]
    return point

def execute_point_transaction(solver):
    """执行点相交事务"""
    point = random_point_params(solver.geo_mesh)
    cell_id = VTKAPI.workload_locate_cells_by_point(solver.geo_mesh, point)
    
    if cell_id != -1:
        variable = random.choice(solver.VARIABLES)
        timestep = random.choice(solver.TIMESTEPS)
        data = solver.points_query([cell_id], variable, timestep)
        
        if data:
            agg_type = random.choice(['sum', 'avg', 'max', 'min'])
            agg_funcs = {
                'sum': solver.aggregation_sum,
                'avg': solver.aggregation_avg, 
                'max': solver.aggregation_max,
                'min': solver.aggregation_min
            }
            result = agg_funcs[agg_type](data)
            return True
    return False

def execute_line_transaction(solver):
    """执行线相交事务"""
    start, end = random_line_params(solver.geo_mesh)
    _, intersected_cells = VTKAPI.workload_line_interesction(solver.geo_mesh, start, end, tolerance=1e-6)

    if intersected_cells:
        variable = random.choice(solver.VARIABLES)
        timestep = random.choice(solver.TIMESTEPS)
        data = solver.points_query(intersected_cells, variable, timestep)
        
        if data:
            agg_type = random.choice(['sum', 'avg', 'max', 'min'])
            agg_funcs = {
                'sum': solver.aggregation_sum,
                'avg': solver.aggregation_avg,
                'max': solver.aggregation_max, 
                'min': solver.aggregation_min
            }
            result = agg_funcs[agg_type](data)
            return True
    return False

def execute_plane_transaction_with_timeout(solver, timeout_seconds=10):
    """执行面相交事务"""
    def plane_worker():
        try:
            origin, normal = random_plane_params(solver.geo_mesh)
            intersected_cells = VTKAPI.workload_plane_intersection(solver.geo_mesh, origin, normal)
            
            if intersected_cells:
                variable = random.choice(solver.VARIABLES)
                timestep = random.choice(solver.TIMESTEPS)
                data = solver.points_query(intersected_cells, variable, timestep)
                
                if data:
                    agg_type = random.choice(['sum', 'avg', 'max', 'min'])
                    agg_funcs = {
                        'sum': solver.aggregation_sum,
                        'avg': solver.aggregation_avg,
                        'max': solver.aggregation_max,
                        'min': solver.aggregation_min
                    }
                    result = agg_funcs[agg_type](data)
                    return True
            return False
        except Exception as e:
            print(f"面相交事务内部错误: {e}")
            return False
    
    # 使用线程执行面相交事务
    result = [None]  # 使用列表来传递结果
    
    def worker_wrapper():
        result[0] = plane_worker()
    
    thread = threading.Thread(target=worker_wrapper)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        print(f"⚠️  面相交查询超时 ({timeout_seconds}秒)，终止当前事务")
        return False
    
    return result[0] if result[0] is not None else False

def measure_tpm_by_type(solver, transaction_func, duration_seconds=60, transaction_type="", use_timeout=False, timeout_seconds=10):
    """测量特定类型事务的TPM，带进度条显示"""
    print(f"\n开始测量{transaction_type}的TPM...")
    
    start_time = time.time()
    transaction_count = 0
    successful_count = 0
    timeout_count = 0
    
    # 创建进度条
    with tqdm(total=duration_seconds, desc=f"{transaction_type}进度", unit="秒") as pbar:
        last_update_time = start_time
        
        while time.time() - start_time < duration_seconds:
            try:
                if use_timeout:
                    success = transaction_func(solver, timeout_seconds)
                else:
                    success = transaction_func(solver)
                
                if success:
                    successful_count += 1
                transaction_count += 1
                
                # 更新进度条
                current_time = time.time()
                elapsed = current_time - start_time
                pbar.update(int(elapsed - pbar.n))
                pbar.set_postfix({
                    '事务数': transaction_count,
                    '成功率': f'{successful_count/transaction_count*100:.1f}%',
                    '超时数': timeout_count
                })
                
            except TimeoutException:
                timeout_count += 1
                transaction_count += 1
                pbar.set_postfix({
                    '事务数': transaction_count,
                    '成功率': f'{successful_count/transaction_count*100:.1f}%',
                    '超时数': timeout_count
                })
                continue
            except Exception as e:
                print(f"\n{transaction_type}事务执行错误: {e}")
                transaction_count += 1
                continue
    
    elapsed_time = time.time() - start_time
    tpm = (successful_count / elapsed_time) * 60
    
    print(f"\n{transaction_type} TPM测试结果:")
    print(f"  总事务数: {transaction_count}")
    print(f"  成功事务数: {successful_count}")
    print(f"  超时事务数: {timeout_count}")
    print(f"  测试时间: {elapsed_time:.2f}秒")
    print(f"  TPM: {tpm:.2f} transactions per minute")
    print(f"  成功率: {successful_count/transaction_count*100:.1f}%")
    
    return tpm

def check_database_data(solver):
    """检查数据库中是否有可用的数据"""
    try:
        cursor = solver.connection.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
        SELECT ship_type, scale, timestep, variable, is_element, array_length(data, 1) as data_length
        FROM cae_simulation_data 
        WHERE ship_type = 'JBC' AND scale = '615k'
        LIMIT 5
        """)
        
        rows = cursor.fetchall()
        print("数据库中的数据样本:")
        for row in rows:
            print(f"  {row}")
        
        cursor.close()
        
    except Exception as e:
        print(f"检查数据库错误: {e}")

def main():
    # 设置信号处理（用于超时控制）
    signal.signal(signal.SIGALRM, timeout_handler)
    
    print("开始TPM性能测试")
    print("=" * 50)
    
    # 初始化求解器
    solver = Workload_Solver("/data/vtk_files/JBC_615k_GEO.vtk")
    
    # 检查数据库数据
    print("检查数据库连接和数据...")
    check_database_data(solver)
    
    print("\n" + "=" * 50)
    print("开始性能测试 (60秒每种类型)")
    print("=" * 50)
    
    # 分别测量三种相交类型的TPM
    try:
        point_tpm = measure_tpm_by_type(solver, execute_point_transaction, 60, "点相交")
        line_tpm = measure_tpm_by_type(solver, execute_line_transaction, 60, "线相交")
        
        # 面相交使用超时控制，设置10秒超时
        plane_tpm = measure_tpm_by_type(
            solver, 
            execute_plane_transaction_with_timeout, 
            60, 
            "面相交", 
            use_timeout=True, 
            timeout_seconds=10
        )
        
    except KeyboardInterrupt:
        print("\n\n 测试被用户中断")
        return
    except Exception as e:
        print(f"\n\n 测试过程中发生错误: {e}")
        return
    
    # 输出汇总结果
    print("\n" + "=" * 50)
    print("TPM性能测试汇总")
    print("=" * 50)
    print(f" 点相交 TPM: {point_tpm:.2f}")
    print(f" 线相交 TPM: {line_tpm:.2f}")
    print(f" 面相交 TPM: {plane_tpm:.2f}")
    print(f" 平均 TPM: {(point_tpm + line_tpm + plane_tpm) / 3:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    main()