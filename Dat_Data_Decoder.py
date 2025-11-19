from genericpath import isdir
import Zone
import numpy as np
from tqdm import tqdm, tqdm_gui
import os
import pandas as pd
import re


class CAE_Decoder:
    # Func: Decode_dat_file
    # Inputs:
    #   path: the path of the dat file.
    #   return: the decoded CAE data structure.
    #   description: Decode a given CAE data. The file structure of the .dat file is default.
    Title = ""
    Variables = []
    Var_count = -1
    Zones = []
    N_DIM = -1

    def __init__(self, DIM):
        # Re-initialize the object everytime to avoid left over problems in previous iterations.
        self.Title = ""
        self.Variables = []
        self.Var_count = -1
        self.Zones = []
        self.N_DIM = DIM
        self.zone_counter = 0
        return

    def Decode_dat_file(self, path):
        file_object = open(path, 'r', encoding="UTF-8")
        raw_content = file_object.read()
        file_object.close()

        solution_time = 0
        solution_time_match = re.search(r'STRANDID\s*=\s*\d+,\s*SOLUTIONTIME\s*=\s*([\d.]+)', raw_content)
        if solution_time_match:
            solution_time_str = solution_time_match.group(1)
            try:
                solution_time = float(solution_time_str)
                print(f"从STRANDID行解析SOLUTIONTIME：{solution_time}")
            except ValueError:
                print(f"STRANDID行解析SOLUTIONTIME失败：{solution_time_str}")

        # Processing header first, extracting title and variables:

        paragraphs = raw_content.split("ZONE  T=")

        header_lines = paragraphs[0].split("\n")
        for line in header_lines:
            line = line.strip()
            # Extracting title
            if line.startswith('TITLE'):
                tokens = line.split('=')
                self.Title = tokens[1].replace('"', '')

            # Extracting variables
            if line.startswith('VARIABLES'):
                tokens = line.split('=')
                _vars = tokens[1].replace('"', '').replace(',', ' ').split()
                # for var in _vars:
                #     self.Variables.append(var)
                self.Variables = [var.strip() for var in _vars if var.strip()]
                self.Var_count = len(self.Variables)
                print(f"解析到变量，共 {self.Var_count} 个: {self.Variables}")

        # Each of the remaining paragraphs represent a ZONE, process them using the same logic:
        for i in range(1, len(paragraphs)):
            if i > 1:
                # Thus far we only decode the first Zone, i.e. fluid zone. The rest of the zones have problem understanding their organization
                # consider the rest as future works.
                break
            # Decoding Each Zone
            paragraph = paragraphs[i]

            zone_local_id = self.zone_counter
            self.zone_counter += 1

            zone = Zone.Zone_3D(paragraph, self.Var_count, self.N_DIM, self.Variables, zone_local_id, solution_time)
            self.Zones.append(zone)

        return


def print_array(arr, N, name):
    with open(name + ".txt", "w") as f:
        f.write(f"{N}\n")
        for item in arr:
            f.write(f"{item}\n")


def main(input_path=None):
    PRINT_DAT = True
    if input_path is not None:
        path = input_path
    else:
        path = 'tecplot/'
    print("Decoding Post-processing data")
    if not os.path.exists(path):
        print(f"Error, path does not exist: {path}")
        return
    elif not os.path.isdir(path):
        print(f"Error, please input the directory to which the .dat files are located: {path}")

    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        if os.path.isfile(filepath):  # Process Files only

            # Check if the filename ends with .dat
            if filepath.lower().endswith('.dat'):
                print(f"Processing file: {filepath}")
                data = CAE_Decoder(3)
                data.Decode_dat_file(filepath)
                # Thus far, only the fluid zone is our interest
                fluid_zone: Zone.Zone_3D = data.Zones[0]

                if PRINT_DAT:
                    decoded_path = 'Decoded_Data/'
                    print('Outputting decoded .dat file')
                    if not os.path.exists(decoded_path):
                        os.mkdir(decoded_path)

                    filename_without_ext = os.path.splitext(file)[0]
                    path_written_to = os.path.join(decoded_path, filename_without_ext)

                    if os.path.exists(path_written_to):
                        print(
                            f'WARNING, data corresponding to timestep {filename_without_ext} has been already generated. Skipping...')
                        continue
                    else:
                        os.makedirs(path_written_to, exist_ok=True)

                    for i in range(0, 3):  # 3 dimensions
                        print_array(fluid_zone.Element_Coordinates[i], fluid_zone.Element_count,
                                    os.path.join(path_written_to, 'Element_' + data.Variables[i]))
                        print_array(fluid_zone.Node_Coordinates[i], fluid_zone.Node_count,
                                    os.path.join(path_written_to, 'Node_' + data.Variables[i]))
                    for i in range(3, data.Var_count):
                        print_array(fluid_zone.Element_Variables[i - 3], fluid_zone.Element_count,
                                    os.path.join(path_written_to, 'Element_' + data.Variables[i]))

                    print("DONE!!!")

                else:
                    print(f"Warning: '{filepath}' is not a .dat file, ignoring...")


if __name__ == "__main__":
    main()