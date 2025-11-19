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

class W5_Workload_Solver:
    def __init__(self, vtk_file_path):
        self.geo_mesh = VTKAPI.load_unstructured_grid_from_vtk_file(vtk_file_path)
        self.connection = create_postgres_connection()
        
        # 从文件名解析船舶类型和尺度
        file_name = vtk_file_path.split('/')[-1]
        parts = file_name.split('_')
        self.scale = parts[1] if len(parts) >= 2 else "615k"
        self.SHIP_TYPE = file_name.split('_GEO')[0].split('_')[0]
        
        debug_print(f"W5求解器初始化完成", "W5", "初始化")
        debug_print(f"船舶类型: {self.SHIP_TYPE}, 尺度: {self.scale}")

    def query_velocity_at_position(self, position, timestep):
        """查询指定位置的流速 - 基于W5图片中的Q_p函数"""
        debug_print(f"查询位置流速: position={position}, timestep={timestep}", "W5", "流速查询")
        
        # 首先定位位置所在的网格单元
        cell_id = VTKAPI.workload_locate_cells_by_point(self.geo_mesh, position)
        
        if cell_id == -1:
            debug_print(f"位置不在网格内: {position}", "W5", "网格检查")
            return None
        
        # 查询该单元的速度分量
        velocity = self._query_cell_velocity(cell_id, timestep)
        
        if velocity is not None:
            debug_print(f"查询到流速: {velocity}", "W5", "流速查询")
        else:
            debug_print(f"未查询到流速数据", "W5", "流速查询")
            
        return velocity

    def _query_cell_velocity(self, cell_id, timestep):
        """查询指定单元的速度向量"""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            # 查询三个速度分量
            velocity_components = []
            for var in ['u', 'v', 'w']:
                sql = """
                SELECT data[%s] as value 
                FROM cae_simulation_data 
                WHERE ship_type = %s 
                  AND scale = %s 
                  AND timestep = %s 
                  AND variable = %s 
                  AND is_element = true;
                """
                
                params = (cell_id + 1, self.SHIP_TYPE, self.scale, timestep, var)
                cursor.execute(sql, params)
                result = cursor.fetchone()
                
                if result and result['value'] is not None:
                    velocity_components.append(float(result['value']))
                else:
                    velocity_components.append(0.0)
            
            cursor.close()
            
            if len(velocity_components) == 3:
                return np.array(velocity_components)
            else:
                return None
                
        except Exception as e:
            print(f"速度查询错误: {e}")
            return None

    def check_point_in_mesh(self, point):
        """检查点是否在网格内 - 基于W5图片中的I(p, M)函数"""
        cell_id = VTKAPI.workload_locate_cells_by_point(self.geo_mesh, point)
        return cell_id != -1

    def solve_streamline(self, init_position, timestep, delta_t=0.1):
        """Streamline Analysis - 严格按照W5图片中的算法实现"""
        debug_print(f"开始Streamline分析", "W5", "开始")
        debug_print(f"初始位置: {init_position}, 时间步: {timestep}, 步长: {delta_t}", "W5", "参数")
        
        # S ← {p₀} - 初始化起始位置
        streamline = [np.array(init_position)]
        current_position = np.array(init_position)
        
        step_count = 0
        max_iterations = 500  # 防止无限循环
        
        while step_count < max_iterations:
            step_count += 1
            debug_print(f"迭代步骤 {step_count}", "W5", f"步骤{step_count}")
            
            # v ← Q_p(p_{i-1}, vel, F) - 点查询：获取位置处的速度
            velocity = self.query_velocity_at_position(current_position, timestep)
            
            if velocity is None:
                debug_print(f"步骤{step_count}: 无法获取流速，停止追踪", "W5", "流速检查")
                break
            
            # 检查速度是否接近0（停滞）
            velocity_magnitude = np.linalg.norm(velocity)
            if velocity_magnitude < 1e-6:
                debug_print(f"步骤{step_count}: 流速接近0，停止追踪", "W5", "流速检查")
                break
            
            debug_print(f"步骤{step_count}: 当前流速大小: {velocity_magnitude:.6f}", "W5", "流速大小")
            
            # p_i ← p_{i-1} + v · Δt - 更新位置
            next_position = current_position + velocity * delta_t
            debug_print(f"步骤{step_count}: 当前位置: {current_position}, 下一位置: {next_position}", "W5", "位置更新")
            
            # if I(p_i, M) = ∅ then break - 如果点超出网格则退出
            if not self.check_point_in_mesh(next_position):
                debug_print(f"步骤{step_count}: 下一位置超出网格，停止追踪", "W5", "边界检查")
                break
            
            # S ← S ∪ p_i - 添加到流线
            streamline.append(next_position)
            current_position = next_position
            
            # 检查是否形成循环（位置变化很小）
            if len(streamline) >= 2:
                last_movement = np.linalg.norm(streamline[-1] - streamline[-2])
                if last_movement < 1e-6:
                    debug_print(f"步骤{step_count}: 位置变化很小，可能形成循环，停止追踪", "W5", "循环检查")
                    break
        
        streamline_array = np.array(streamline)
        debug_print(f"Streamline分析完成: 生成{len(streamline_array)}个轨迹点", "W5", "完成")
        
        return streamline_array

    def execute_streamline_transaction(self):
        """执行单个Streamline事务 - 基于w1tpm.py的事务模式"""
        try:
            # 生成随机初始位置（在网格边界内）
            bounds = self.geo_mesh.GetBounds()
            init_position = [
                random.uniform(bounds[0], bounds[1]),
                random.uniform(bounds[2], bounds[3]),
                random.uniform(bounds[4], bounds[5])
            ]
            
            # 随机选择时间步（从可用时间步中选择）
            available_timesteps = self._get_available_timesteps()
            if not available_timesteps:
                debug_print("没有可用的时间步数据", "W5", "数据检查")
                return False
                
            timestep = random.choice(available_timesteps)
            
            # 随机时间步长
            delta_t = random.uniform(0.05, 0.3)
            
            debug_print(f"执行W5事务", "W5", "事务开始")
            debug_print(f"初始位置: {init_position}", "W5", "参数")
            debug_print(f"时间步: {timestep}, 步长: {delta_t:.3f}", "W5", "参数")
            
            # 执行Streamline分析
            streamline = self.solve_streamline(init_position, timestep, delta_t)
            
            # 成功条件：至少生成2个轨迹点
            success = len(streamline) > 1
            debug_print(f"W5事务完成: 成功={success}, 轨迹点数={len(streamline)}", "W5", "事务完成")
            
            return success
            
        except Exception as e:
            print(f"W5事务执行错误: {e}")
            return False

    def _get_available_timesteps(self):
        """获取数据库中可用的时间步"""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
            SELECT DISTINCT timestep
            FROM cae_simulation_data 
            WHERE ship_type = %s AND scale = %s AND variable = 'u' AND is_element = true
            ORDER BY timestep
            """, (self.SHIP_TYPE, self.scale))
            
            results = cursor.fetchall()
            timesteps = [row['timestep'] for row in results]
            
            cursor.close()
            return timesteps
            
        except Exception as e:
            print(f"获取时间步错误: {e}")
            return []

    def measure_tpm(self, duration_seconds=60):
        """测量TPM - 基于w1tpm.py的测量模式"""
        print(f"\n开始测量W5 Streamline Analysis的TPM...")
        
        start_time = time.time()
        transaction_count = 0
        successful_count = 0
        
        # 创建进度条
        with tqdm(total=duration_seconds, desc="W5 TPM进度", unit="秒") as pbar:
            while time.time() - start_time < duration_seconds:
                try:
                    success = self.execute_streamline_transaction()
                    
                    if success:
                        successful_count += 1
                    transaction_count += 1
                    
                    # 更新进度条
                    current_time = time.time()
                    elapsed = current_time - start_time
                    pbar.update(int(elapsed - pbar.n))
                    pbar.set_postfix({
                        '事务数': transaction_count,
                        '成功率': f'{successful_count/transaction_count*100:.1f}%' if transaction_count > 0 else '0%',
                        'TPM预估': f'{(successful_count/elapsed)*60:.1f}' if elapsed > 0 else '0'
                    })
                    
                except Exception as e:
                    print(f"\nW5事务执行错误: {e}")
                    transaction_count += 1
                    continue
        
        elapsed_time = time.time() - start_time
        tpm = (successful_count / elapsed_time) * 60
        
        print(f"\nW5 Streamline Analysis TPM测试结果:")
        print(f"  总事务数: {transaction_count}")
        print(f"  成功事务数: {successful_count}")
        print(f"  测试时间: {elapsed_time:.2f}秒")
        print(f"  TPM: {tpm:.2f} transactions per minute")
        print(f"  成功率: {successful_count/transaction_count*100:.1f}%" if transaction_count > 0 else "  成功率: 0%")
        
        return tpm

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            debug_print("数据库连接已关闭", "W5", "清理")

def check_database_connectivity(solver):
    """检查数据库连接和数据可用性"""
    try:
        cursor = solver.connection.cursor(cursor_factory=RealDictCursor)
        
        # 检查速度数据可用性
        cursor.execute("""
        SELECT variable, COUNT(*) as count, 
               MIN(timestep) as min_timestep, MAX(timestep) as max_timestep,
               AVG(array_length(data, 1)) as avg_data_length
        FROM cae_simulation_data 
        WHERE ship_type = %s AND scale = %s AND variable IN ('u', 'v', 'w') AND is_element = true
        GROUP BY variable
        """, (solver.SHIP_TYPE, solver.scale))
        
        rows = cursor.fetchall()
        print("W5数据库连接检查:")
        if rows:
            print(f"  找到 {len(rows)} 个速度变量:")
            for row in rows:
                print(f"    {row['variable']}: {row['count']}个时间步, 数据长度: {row['avg_data_length']:.0f}")
            
            available_timesteps = solver._get_available_timesteps()
            if available_timesteps:
                print(f"  可用时间步: {available_timesteps}")
            else:
                print("  警告: 没有可用的时间步数据!")
                
        else:
            print("  警告: 未找到W5所需的速度数据!")
        
        cursor.close()
        return len(rows) > 0
        
    except Exception as e:
        print(f"数据库连接检查错误: {e}")
        return False

def main():
    # 设置信号处理
    signal.signal(signal.SIGALRM, timeout_handler)
    
    print("开始W5 Streamline Analysis TPM性能测试")
    print("=" * 50)
    
    # 初始化W5求解器
    vtk_file_path = "/data/vtk_files/JBC_615k_GEO.vtk"
    solver = W5_Workload_Solver(vtk_file_path)
    
    # 检查数据库连接和数据
    print("检查数据库连接和数据...")
    data_available = check_database_connectivity(solver)
    
    if not data_available:
        print("错误: 数据库中没有W5所需的速度数据，无法进行测试!")
        return
    
    print("\n" + "=" * 50)
    print("开始W5 TPM性能测试 (60秒)")
    print("=" * 50)
    
    try:
        # 测量W5 TPM
        w5_tpm = solver.measure_tpm(60)
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        return
    except Exception as e:
        print(f"\n\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        # 清理资源
        solver.close()
    
    # 输出结果
    print("\n" + "=" * 50)
    print("W5 TPM性能测试汇总")
    print("=" * 50)
    print(f" Streamline Analysis TPM: {w5_tpm:.2f}")
    print("=" * 50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--vtk_file", type=str, default="/data/vtk_files/JBC_615k_GEO.vtk", 
                       help="VTK几何文件路径")
    parser.add_argument("--duration", type=int, default=60, 
                       help="测试持续时间（秒）")
    args = parser.parse_args()
    
    # 设置调试模式
    DEBUG_MODE = args.debug
    if DEBUG_MODE:
        print("调试模式已启用")
    
    main()