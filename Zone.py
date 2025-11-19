import re
from tkinter import OFF
from xml.sax.handler import DTDHandler
from xmlrpc.client import boolean
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
# Class Zone_3D:
# A data structure used for storing a zone for a 3D model
# The main components are as follows
# Elements: #Elements (#: Number of)
# Faces: #Faces
# Nodes: #Nodes
# ZoneType: Self-explainatory
# Parameters:
#   X, Y, Z: float[Nodes] arrays
#   X, Y, Z,U, V, W, P, K, E: float[Elements] arrays
"""


class Zone_3D:

    def generateMesh(self):
        return

    def __init__(self, raw_content, var_count, dim, variables, zone_local_id, solution_time, db_zone_id=None):
        self.Zone_name = ''
        self.Zone_type = ''
        self.Element_count = 0
        self.Face_count = 0
        self.Node_count = 0
        self.Variables = variables
        self.Node_Coordinates = []
        self.Element_Variables = []
        self.Element_Coordinates = []
        self.NCPF = []
        self.FN = []
        self.LE = []
        self.RE = []
        self.EN = []
        self.EF = []

        self.DIMENSION = dim
        self.zone_local_id = zone_local_id
        self.db_zone_id = db_zone_id
        self.solution_time = float(solution_time) if solution_time is not None else 0.0

        # 调试信息
        print(f"Zone初始化, zone_local_id = {zone_local_id}, solution_time = {self.solution_time}")
        print(f" 变量数量: {var_count}, 变量列表: {variables}")

        self.Variables = variables
        # Construting the line of DT=(..).
        splitter_DT = 'DT=('
        for i in range(0, var_count):
            splitter_DT += 'DOUBLE'
            if i < var_count - 1:
                splitter_DT += ' '
        splitter_DT += ')'

        sections = raw_content.split(splitter_DT)
        _vars = sections[1]

        # Extracting header
        _header = sections[0]
        # Extracting Zone name
        lines = _header.split('\n')
        self.Zone_name = lines[0].replace('"', '')
        # Extracting Node, Element, Face counts & Zonetype
        for line in lines:
            if line.strip().startswith('Nodes'):
                line = line.replace(' ', '')
                parts = line.split(',')
                for part in parts:
                    pairs = part.split('=')
                    if pairs[0] == 'Nodes':
                        self.Node_count = int(pairs[1])
                    elif pairs[0] == 'Elements':
                        self.Element_count = int(pairs[1])
                    elif pairs[0] == 'Faces':
                        self.Face_count = int(pairs[1])
                    else:
                        self.Zone_type = pairs[1]

        # Extracting parameter values

        vals_components = _vars.split("#")

        DT = vals_components[0]

        for i in range(1, len(vals_components)):
            component = vals_components[i]
            if component.startswith(' node count per face'):
                node_count_per_face = component
            elif component.startswith(' face nodes'):
                face_nodes = component
            elif component.startswith(' left elements'):
                left_elements = component
            elif component.startswith(' right elements'):
                right_elements = component

        '''
        Decoding DT:
        '''
        # Processing DT
        print("Decoding DT:")
        DT_array = DT.replace("\n", "").replace("   ", "  ").replace("  ", " ").strip().split(" ")

        N = self.Node_count
        E = self.Element_count

        if len(DT_array) != 3 * N + (var_count - 3) * E:
            print("WARINING, The length of DT could be wrong.")

        N = self.Node_count
        E = self.Element_count
        F = self.Face_count

        visited_element = 0
        for i in tqdm(range(0, var_count)):
            if i < 3:
                tmp_var_txt = DT_array[visited_element: visited_element + N]
                visited_element += N
                tmp_var_double = np.zeros(N, dtype=np.float64)
                for i in range(0, N):
                    tmp_var_double[i] = np.float64(tmp_var_txt[i])
                self.Node_Coordinates.append(tmp_var_double)
            else:
                tmp_var_txt = DT_array[visited_element: visited_element + E]
                visited_element += E
                tmp_var_double = np.zeros(E, dtype=np.float64)
                for i in range(0, E):
                    tmp_var_double[i] = np.float64(tmp_var_txt[i])
                self.Element_Variables.append(tmp_var_double)

        '''
        Decoding node_count_per_face:
        '''
        print("Decoding NCPF:")
        self.NCPF = np.array(self.decode_regular_part(node_count_per_face, self.Face_count, 0))
        '''
        Decoding face_nodes:
        '''
        print("Decoding FN:")
        face_nodes_array = np.array(self.decode_regular_part(face_nodes, -1, -1))
        # self.FN = self.decode_face_nodes(face_nodes)
        self.FN = self.decode_face_node_array(face_nodes_array, self.Face_count)
        '''
        Decoding left_elements:
        '''
        print("Decoding LE:")
        self.LE = np.array(self.decode_regular_part(left_elements, self.Face_count, -1))

        '''
        Decoding right_elements:
        '''
        print("Decoding RE:")
        self.RE = np.array(self.decode_regular_part(right_elements, self.Face_count, -1))

        '''
        Constructing element_nodes:
        '''
        print("Constructing Element_Nodes")
        [self.EN, self.EF] = self.construct_element_face_and_nodes()

        '''
        Computing element centroids:
        '''
        self.decode_element_centroids_using_element_nodes()

        REQUIRES_CHECKING = False
        if REQUIRES_CHECKING:
            ''' Checking ...'''
            # # Create histogram
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

            # Plot histograms
            ax1.hist(self.Element_Coordinates[0], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax2.hist(self.Element_Coordinates[1], bins=30, color='salmon', edgecolor='black', alpha=0.7)
            ax3.hist(self.Element_Coordinates[2], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)

            # Customize each subplot
            ax1.set_title('Normal Distribution', fontsize=14)
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, linestyle='--', alpha=0.6)

            ax2.set_title('Exponential Distribution', fontsize=14)
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, linestyle='--', alpha=0.6)

            ax3.set_title('Uniform Distribution', fontsize=14)
            ax3.set_xlabel('Value')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, linestyle='--', alpha=0.6)

            # Add overall title and adjust layout
            plt.suptitle('Distribution Comparison of Three Arrays', fontsize=16, y=1.02)
            plt.tight_layout()
            plt.show(block=True)

            # Checking EN & EF
            good_match_count = 0
            bad_match_count = 0
            for e in tqdm(range(0, self.Element_count)):
                nodes_by_EN = set(self.EN[e])
                nodes_by_EF = set()
                for f in self.EF[e]:
                    for n in self.FN[f]:
                        nodes_by_EF.add(n)
                if nodes_by_EF == nodes_by_EN:
                    # print(f"GOOD: EN and EF of Element {e} matches...")
                    good_match_count += 1
                else:
                    print(f"ERROR: EN and EF of Element {e} does not match...")
                    bad_match_count += 1

            print(f"\nGood match count = {good_match_count}.")
            print(f"\nBad match count = {bad_match_count}.")

    def decode_regular_part(self, array_in_text, N, offset):
        lines = array_in_text.split("\n")[1:]
        values_in_txt = []
        for line in tqdm(lines):
            if len(line) == 0:
                continue
            tokens = line.strip().replace("   ", "  ").replace("  ", " ").split(" ")
            for token in tokens:
                values_in_txt.append(token)
        if N != -1:
            if len(values_in_txt) != N:
                print("Error, decoding regular array wrong.")

        result = np.zeros(len(values_in_txt), dtype=np.int64)
        for i in range(0, len(values_in_txt)):
            result[i] = np.int64(values_in_txt[i]) + offset

        return result

    def decode_face_nodes(self, array_in_text):
        lines = array_in_text.split("\n")[1:]
        result = []
        for line in tqdm(lines):
            if len(line) == 0:
                continue
            processed_line = []
            elements = line.strip().split(' ')
            for element in elements:
                processed_line.append(np.int64(element))
            result.append(processed_line)

        return result

    def decode_face_node_array(self, face_node_array, N):
        # N: face count
        result = []
        offset = 0
        for i in tqdm(range(0, N)):
            tmp_face_node_count = self.NCPF[i]
            result.append(face_node_array[offset: offset + tmp_face_node_count])
            offset += tmp_face_node_count

        return result

    def to_dataframes(self, ship_type="JBC", scale="615k"):
        """
        转换为新的表格结构DataFrame
        """
        solution_time = int(self.solution_time) if self.solution_time is not None else 0
        
        print(f"创建DataFrame - Ship: {ship_type}, Scale: {scale}, Timestep: {solution_time}")
        print(f"可用变量列表: {self.Variables}")
        print(f"节点坐标数量：{len(self.Node_Coordinates)}")
        print(f"单元坐标数量：{len(self.Element_Coordinates)}")
        print(f"单元变量数量：{len(self.Element_Variables)}")

        data_records = []

        coordinate_vars = ['X', 'Y', 'Z']
        physics_vars = ['u', 'v', 'w', 'p', 'k', 'e']

        # 处理节点坐标数据 (is_element = False)
        # coordinate_vars = self.Variables[:3] if len(self.Variables) >=3 else ['X', 'Y', 'Z']
        # for i, var_name in enumerate(coordinate_vars):
        #     if i < len(self.Node_Coordinates):
        #         node_data = self.Node_Coordinates[i].tolist() if hasattr(self.Node_Coordinates[i], 'tolist') else self.Node_Coordinates[i]
        #         record = {
        #             "ship_type": ship_type,
        #             "scale": scale, 
        #             "timestep": solution_time,
        #             "variable": var_name,
        #             "is_element": False,
        #             "data": self.Node_Coordinates[i].tolist()
        #         }
        #         data_records.append(record)
        for i in range(min(3, len(self.Node_Coordinates))):
            var_name = coordinate_vars[i] if i < len(coordinate_vars) else f"Coord_{i}"
            record = {
                "ship_type": ship_type,
                "scale": scale,
                "timestep": solution_time,
                "variable": var_name,
                "is_element": False,
                "data": self.Node_Coordinates[i].tolist() if hasattr(self.Node_Coordinates[i], 'tolist') else list(self.Node_Coordinates[i])
            }
            data_records.append(record)
            print(f"  添加节点数据: {var_name}, 长度: {len(record['data'])}")

        # 处理单元坐标数据 (is_element = True)
        # for i, var_name in enumerate(coordinate_vars):
        #     if i < len(self.Element_Coordinates):
        #         element_data = self.Element_Coordinates[i].tolist() if hasattr(self.Element_Coordinates[i], 'tolist') else list(self.Element_Coordinates[i])
        #         record = {
        #             "ship_type": ship_type,
        #             "scale": scale,
        #             "timestep": solution_time, 
        #             "variable": var_name,
        #             "is_element": True,
        #             "data": self.Element_Coordinates[i].tolist()
        #         }
        #         data_records.append(record)
        for i in range(min(3, len(self.Element_Coordinates))):
            var_name = coordinate_vars[i] if i < len(coordinate_vars) else f"Coord_{i}"
            record = {
                "ship_type": ship_type,
                "scale": scale,
                "timestep": solution_time,
                "variable": var_name,
                "is_element": True,
                "data": self.Element_Coordinates[i].tolist() if hasattr(self.Element_Coordinates[i], 'tolist') else list(self.Element_Coordinates[i])
            }
            data_records.append(record)
            print(f"  添加单元坐标: {var_name}, 长度: {len(record['data'])}")

        # 处理物理变量数据 (is_element = True)
        # physics_vars = self.Variables[3:] if len(self.Variables) > 3 else ['u', 'v', 'w', 'p', 'k', 'e']
        # for i, var_data in enumerate(self.Element_Variables):
        #     var_name = physics_vars[i] if i < len(physics_vars) else f"var_{i}"
        #     physics_vars = var_data.tolist() if hasattr(var_data, 'tolist') else list(var_data)
        #     record = {
        #         "ship_type": ship_type,
        #         "scale": scale,
        #         "timestep": solution_time,
        #         "variable": var_name, 
        #         "is_element": True,
        #         "data": var_data.tolist()
        #     }
        #     data_records.append(record)
        for i in range(len(self.Element_Variables)):
            # 确保使用正确的物理变量名
            if i < len(physics_vars):
                var_name = physics_vars[i]
            else:
                var_name = f"Var_{i+1}"  # 备用命名
            
            var_data = self.Element_Variables[i]
            record = {
                "ship_type": ship_type,
                "scale": scale,
                "timestep": solution_time,
                "variable": var_name,
                "is_element": True,
                "data": var_data.tolist() if hasattr(var_data, 'tolist') else list(var_data)
            }
            data_records.append(record)
            print(f"  添加物理变量: {var_name}, 长度: {len(record['data'])}")

        # 创建DataFrame
        columns_order = ["ship_type", "scale", "timestep", "variable", "is_element", "data"]
        new_structure_df = pd.DataFrame(data_records)[columns_order]
        
        print(f"DataFrame创建完成，共{len(new_structure_df)}条记录")

        unique_vars = new_structure_df['variable'].unique()
        print(f"生成的变量: {list(unique_vars)}")

        # 调试信息
        print("数据预览:")
        for i, record in enumerate(data_records[:3]):  # 显示前3条记录
            print(f"  记录{i+1}: {record['variable']}, is_element={record['is_element']}, 数据长度={len(record['data'])}")

        return {
            "cae_simulation_data": new_structure_df
        }

    def set_db_zone_id(self, db_zone_id):
        self.db_zone_id = db_zone_id

    def get_db_zone_id(self):
        return self.db_zone_id

    def get_local_zone_id(self):
        return self.zone_local_id

    def construct_element_face_and_nodes(self):
        element_nodes_dict = defaultdict(set)
        element_faces_dict = defaultdict(set)

        for f in tqdm(range(0, self.Face_count)):
            face_nodes = self.FN[f]
            # Processing left elements
            tmp_element_id = self.LE[f]

            # Filtering unwanted element ids
            if tmp_element_id == -1:
                # element '-1' represents the boundary element, omit it during computation
                continue

            # Checking if element_id has been processed
            if tmp_element_id in element_nodes_dict:
                tmp_element_nodes = element_nodes_dict[tmp_element_id]
                tmp_element_faces = element_faces_dict[tmp_element_id]
            else:
                tmp_element_nodes = set()
                tmp_element_faces = set()

            # Adding phase
            tmp_element_faces.add(f)
            for p in face_nodes:
                tmp_element_nodes.add(p)

            element_faces_dict[tmp_element_id] = tmp_element_faces
            element_nodes_dict[tmp_element_id] = tmp_element_nodes

            # Processing right elements
            tmp_element_id = self.RE[f]

            # Checking if element_id has been processed
            if tmp_element_id in element_nodes_dict:
                tmp_element_nodes = element_nodes_dict[tmp_element_id]
                tmp_element_faces = element_faces_dict[tmp_element_id]
            else:
                tmp_element_nodes = set()
                tmp_element_faces = set()

            # Adding phase
            tmp_element_faces.add(f)
            for p in face_nodes:
                tmp_element_nodes.add(p)

            element_faces_dict[tmp_element_id] = tmp_element_faces
            element_nodes_dict[tmp_element_id] = tmp_element_nodes

        return [element_nodes_dict, element_faces_dict]

    def decode_element_centroids_using_element_nodes(self):
        X = self.Node_Coordinates[0]
        Y = self.Node_Coordinates[1]
        Z = self.Node_Coordinates[2]

        print("Loading X, Y, Z:")
        Element_X = np.zeros(self.Element_count)
        Element_Y = np.zeros(self.Element_count)
        Element_Z = np.zeros(self.Element_count)
        for i in tqdm(range(0, self.Element_count)):
            tmp_element_nodes = self.EN[i]
            centroid = np.zeros(3)
            for n in tmp_element_nodes:
                centroid[0] += X[n]
                centroid[1] += Y[n]
                centroid[2] += Z[n]

            centroid = centroid / 8
            Element_X[i] = centroid[0]
            Element_Y[i] = centroid[1]
            Element_Z[i] = centroid[2]

        self.Element_Coordinates.append(Element_X)
        self.Element_Coordinates.append(Element_Y)
        self.Element_Coordinates.append(Element_Z)