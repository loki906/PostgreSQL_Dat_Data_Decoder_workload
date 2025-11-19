import pickle
import vtk
from vtk import vtkCell, vtkCellLocator, vtkCellTypes, vtkPlane, vtkPolyData, vtkIdTypeArray, vtkUnstructuredGrid, vtkIdList
import time
import numpy as np
from Zone import Zone_3D
from Dat_Data_Decoder import CAE_Decoder
from tqdm import tqdm

''' Examplar script '''
def main():
    
    # vtk file containing the geometrical information of the CFD mesh, excluding physical variable infos.
    SHIP_TYPE = "JBC_615k"

    vtk_file_name = SHIP_TYPE + "_GEO.vtk"
    # Load mesh info from .vtk file
    mesh = load_unstructured_grid_from_vtk_file(vtk_file_name)

    connectivity_name = SHIP_TYPE + "_connectivity.pkl"
    
    with open(connectivity_name, 'rb') as f:
        connectivity = pickle.load(f)

    vtk_check_watertight(mesh)
    

    # I. Point intersection
    
    # coordinates of the points
    points = [[0.01,0.02,-0.1],[0.5,0.2,0.1]]
    
    # Returning ids of the cells containing the points, -1 if not found
    intersected_cells_point = workload_locate_cells_by_points(mesh, points)
                    
                
    # II. Line intersection
    line_start = [0.01,0.00,-0.1]
    line_end = np.random.random(3)
    _, intersected_cells_line = workload_line_interesction(mesh, line_start, line_end, 0.001)
    for cell_id in intersected_cells_line:
        print(cell_id)
    
    # III. Plane intersection
    plane_origin = [0.01,0.02,-0.1]
    plane_norm = np.random.random(3)
    intersected_cells_plane = workload_plane_intersection(mesh, line_start, line_end)
    for cell_id in intersected_cells_line:
        print(cell_id)
    

    # # IV. Face Norms
    # workload_norm_extraction(mesh)
    
    # # V. Isosurface Analysis
    # iso_value = 0.001
    # iso_variable = "P"
    # workload_isosurface_extraction(mesh, iso_value, iso_variable)
    
    # VI. Compute Q-Criterion
    # data = CAE_Decoder(3)
    # data.Decode_dat_file("tecplot\\200.dat")
        

    # Element_Coordinates = data.Zones[0].Element_Coordinates
    # Element_Variables = data.Zones[0].Element_Variables
    
    # with open('Element_Coordinates.pkl', 'wb') as f:
    #     pickle.dump(Element_Coordinates, f)
        
    # with open('Element_Variables.pkl', 'wb') as f:
    #     pickle.dump(Element_Variables, f)

    with open('Element_Coordinates.pkl', 'rb') as f:
        Element_Coordinates = pickle.load(f)
        
    with open('Element_Variables.pkl', 'rb') as f:
        Element_Variables = pickle.load(f)

    compute_gradients_least_squares(Element_Coordinates[0],Element_Coordinates[1], Element_Coordinates[2], Element_Variables[0], Element_Variables[1], Element_Variables[2], connectivity)
    


''' Below are the tools for vtk processing'''

def load_unstructured_grid_from_vtk_file(file):
    # Check if the filename ends with .vtk
    if file.lower().endswith('.vtk'):
        # Create a reader for legacy VTK files
        reader = vtk.vtkUnstructuredGridReader()  # For unstructured grids
        # reader = vtk.vtkDataSetReader()        # For general VTK files (structured/unstructured)
        # Set the filename
        reader.SetFileName(file)
        # Read the file
        reader.Update()  # Triggers the reading process
        # Get the output data
        mesh = reader.GetOutput()  # For unstructured grids
        # Access key properties
        print(f"Number of points: {mesh.GetNumberOfPoints()}")
        print(f"Number of cells: {mesh.GetNumberOfCells()}")
        return mesh
    else:
        print(f"Error, file {file} is not a .vtk file.")
        return -1

def workload_locate_cells_by_point(mesh:vtkUnstructuredGrid, point: list):
    """
    Find the cell that contains the given point

    """
    # point = [x, y, z]  # Your target coordinate (in the same coordinate system as the grid)
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(mesh)
    cell_locator.BuildLocator()
    

    
    cell_id = -1
        # Find the cell
    cell_id = cell_locator.FindCell(point)

        # Check if a cell was found
    if cell_id == -1:
        print(f"Point {point} is outside the grid or no cell found.")
                
    return cell_id

def workload_locate_cells_by_points(mesh:vtkUnstructuredGrid, points: list):
    """
    Find the cells that contains the given points
    
    Args:
        mesh: vtkUnstructuredGrid input
        points: A list of point coordinates in the form of [[x1,y1,z1],[x2,y2,z2],...]
    
    Returns:
        Ids of cells containing the points, -1 if cell not found
    """
    # point = [x, y, z]  # Your target coordinate (in the same coordinate system as the grid)
    cell_locator = vtk.vtkCellLocator()
    cell_locator.SetDataSet(mesh)
    cell_locator.BuildLocator()
    results = []

    for p in points:
        # Query parameters
        cell_id = -1
        # Find the cell
        cell_id = cell_locator.FindCell(p)
        results.append(cell_id)
        # Check if a cell was found
        if cell_id == -1:
            print(f"Point {p} is outside the grid or no cell found.")
            
    return results


def workload_line_interesction(mesh:vtkUnstructuredGrid, line_start, line_end, tolerance):
    """
    
    Compute intersection points between an unstructured grid and a line using vtkCellLocator. A line is defined by its starting and ending points.
    
    Args:
        unstructured_grid: vtkUnstructuredGrid input
        line_start: Start point of the line [x, y, z]
        line_end: End point of the line [x, y, z]
    
    Returns:
        List of intersection points and corresponding cell IDs
    """

    # Variables to store intersection results
    tmp_points = vtk.vtkPoints()  # To store intersection points
    tmp_cells = vtk.vtkIdList()  # To store intersected cell IDs
    param_coords = []  # Parametric coordinates (t) along the line
    
    cell_locator = vtkCellLocator()
    cell_locator.SetDataSet(mesh)
    cell_locator.BuildLocator()  # Preprocess to build the octree
    
    # Compute intersections
    start = time.time()
    cell_locator.IntersectWithLine(line_start, line_end, tolerance, tmp_points, tmp_cells)
    end = time.time()
#    print(f"Line intersection time: {end - start:.6f} seconds...")
    # Extract intersection points and corresponding cell IDs
    
    intersection_point_count = tmp_points.GetNumberOfPoints()
    intersection_cell_count = tmp_cells.GetNumberOfIds()

    intersection_points = []
    intersected_cells = []
    
    for i in range(0, intersection_point_count):
        intersection_points.append(tmp_points.GetPoint(i))
        
    for i in range(0, intersection_cell_count):
        intersected_cells.append(tmp_cells.GetId(i))
    
    return intersection_points, intersected_cells

def workload_plane_intersection(mesh:vtkUnstructuredGrid, plane_origin, plane_norm):
    """
    
    Compute intersection points between an unstructured grid and a line using vtkExtractGeometry. A line is defined by its starting and ending points.

    NOTE: To use this funtion, the ids of the mesh cells must be explicted stored in a cell_array. In this version, the cell_array is named "cell_ids"
    
    Args:
        unstructured_grid: vtkUnstructuredGrid input
        plane_origin: the origin of the plane in the form of vector
        plane_norm: the normal vector of the plane
    
    Returns:
        List of cell IDs intersected with the plane.
    """
    # --- 确保网格有 "cell_ids" 数组（如果没有，添加） ---
    if not mesh.GetCellData().GetArray("cell_ids"):
        cell_ids_array = vtk.vtkIntArray()
        cell_ids_array.SetName("cell_ids")
        cell_ids_array.SetNumberOfComponents(1)
        cell_ids_array.SetNumberOfTuples(mesh.GetNumberOfCells())
        for i in range(mesh.GetNumberOfCells()):
            cell_ids_array.SetValue(i, i)  # 假设 ID 为单元格索引
        mesh.GetCellData().AddArray(cell_ids_array)

    plane = vtk.vtkPlane()
    plane.SetOrigin(plane_origin)
    plane.SetNormal(plane_norm)

    # Use vtkExtractGeometry to get intersected cells
    extractor = vtk.vtkExtractGeometry()
    extractor.SetInputData(mesh)
    extractor.SetImplicitFunction(plane)
    extractor.SetExtractInside(0)  # 0 = extract boundary (intersected) cells
    extractor.SetExtractOnlyBoundaryCells(1)  # Only cells intersected by plane
    extractor.Update()

    # Get the output containing intersected cells
    intersected_cells = extractor.GetOutput()
    
    # Extract cell IDs from the original grid
    original_ids = intersected_cells.GetCellData().GetArray("cell_ids")

    if original_ids:
        cell_ids = []
        for i in range(original_ids.GetNumberOfTuples()):
            cell_ids.append(original_ids.GetValue(i))
#        print(f"Extracted cell Successfully, total {len(cell_ids)} cells intersected with the plane.")
    else:
        print("No 'cell_ids' array found in intersected cells.")

    return cell_ids


def workload_cell_extraction(mesh: vtkUnstructuredGrid, cell_ids: list):
    """
    Extracting a sub-unstructured mesh from the original mesh by a given cell_id list. The geometrical topology between the cells should be preserved (if there is any).
    
    Args:
        mesh: vtkUnstructuredGrid input
        cell_ids: a list of cell_ids in the format of integers
    
    Returns:
        a new vtkUnstructuredGrid
    """
  

    # --- 1. Use vtkExtractCells to create the smaller grid ---
    extract_cells = vtk.vtkExtractCells()
    extract_cells.SetInputData(mesh)
    extract_cells.SetCellList(cell_ids)
    extract_cells.Update()

    # --- 2. Get the output ---
    extracted_grid = extract_cells.GetOutput()

    print("Extracted Grid:")
    print(f"  Number of Points: {extracted_grid.GetNumberOfPoints()}")
    print(f"  Number of Cells: {extracted_grid.GetNumberOfCells()}")

    return 


def workload_isosurface_extraction(mesh: vtkUnstructuredGrid, iso_value:np.double, iso_variable: str):
    print(f"Grid created with {mesh.GetNumberOfPoints()} points and {mesh.GetNumberOfCells()} cells.")

    # --- Part 2: Perform Isosurface Extraction ---
    # The iso-value we want to extract. For our distance field, this will be a sphere of radius 3.0.
    
    contour_filter = vtk.vtkContourFilter()
    contour_filter.SetInputData(mesh)
    contour_filter.SetValue(0, iso_value)
    
    # CRITICAL: Tell the filter which scalar array to use
    # Arguments: (idx, port, connection, fieldAssociation, arrayName)
    # We use the first array (0) on the point data (FIELD_ASSOCIATION_POINTS)
    contour_filter.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "distance")
    
    print(f"Extracting isosurface at value {iso_value}...")
    contour_filter.Update()
    
    isosurface_polydata = contour_filter.GetOutput()
    print(f"Isosurface extraction complete. Output has {isosurface_polydata.GetNumberOfPoints()} points and {isosurface_polydata.GetNumberOfPolys()} polygons.")
    
def workload_norm_extraction(mesh:vtkUnstructuredGrid):
    
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(mesh)

    # 3. Execute the filter
    surface_filter.Update()

    # 4. Get the output boundary surface
    boundary_faces = surface_filter.GetOutput()

    print("--- Boundary Faces Found ---")
    print(f"Original grid has {mesh.GetNumberOfCells()} 3D cells.")
    print(f"Extracted boundary has {boundary_faces.GetNumberOfCells()} 2D faces.")
    print(f"Boundary surface has {boundary_faces.GetNumberOfPoints()} points.")

    # You can now iterate through the boundary faces
    for i in range(boundary_faces.GetNumberOfCells()):
        face = boundary_faces.GetCell(i)
        point_ids = face.GetPointIds()
        print(f"Boundary Face {i} is a {face.GetClassName()} with point IDs: {[point_ids.GetId(j) for j in range(point_ids.GetNumberOfIds())]}")

    return

def compute_gradients_least_squares(X, Y, Z, U, V, W, connectivity):
    """
    Computes velocity gradients for each cell using a least-squares fit with its neighbors.

    Args:
        X, Y, Z (np.ndarray): 1D arrays of length n, containing cell center coordinates.
        U, V, W (np.ndarray): 1D arrays of length n, containing velocity components.
        connectivity (dict): Dictionary mapping cell index to a list of neighbor indices.

    Returns:
        np.ndarray: An (n, 9) array of velocity gradients.
    """
    # Reconstruct the (n, 3) arrays for easier processing
    points = np.column_stack((X, Y, Z))
    velocities = np.column_stack((U, V, W))
    
    n = points.shape[0]
    gradients = np.zeros((n, 9))
    
    print("Computing gradients (this may take a moment for large datasets)...")
    for i in tqdm(range(n)):
        neighbor_indices = connectivity[i]
        
        if len(neighbor_indices) < 3:
            continue
            
        r_i = points[i]
        v_i = velocities[i]
        
        r_neighbors = points[neighbor_indices]
        v_neighbors = velocities[neighbor_indices]
        
        A = r_neighbors - r_i
        
        try:
            b_u = v_neighbors[:, 0] - v_i[0]
            grad_u, _, _, _ = np.linalg.lstsq(A, b_u, rcond=None)
            
            b_v = v_neighbors[:, 1] - v_i[1]
            grad_v, _, _, _ = np.linalg.lstsq(A, b_v, rcond=None)
            
            b_w = v_neighbors[:, 2] - v_i[2]
            grad_w, _, _, _ = np.linalg.lstsq(A, b_w, rcond=None)
            
            gradients[i] = np.concatenate([grad_u, grad_v, grad_w])
            
        except np.linalg.LinAlgError:
            pass
            
    print("Gradient computation complete.")
    return gradients



def vtk_check_watertight(mesh: vtkUnstructuredGrid):
    # Create a filter to find boundary edges
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(mesh)
    # We only care about boundary edges and non-manifold edges
    feature_edges.BoundaryEdgesOn()
    feature_edges.NonManifoldEdgesOn()
    feature_edges.FeatureEdgesOff() # Turn off sharp feature edges
    feature_edges.ManifoldEdgesOff()
    feature_edges.Update()

    boundary_edges = feature_edges.GetOutput()

    print(f"Found {boundary_edges.GetNumberOfCells()} boundary/non-manifold edges.")
    if boundary_edges.GetNumberOfCells() > 0:
        print("The mesh is NOT watertight.")
        # You can visualize 'boundary_edges' to see exactly where the holes are.
    else:
        print("The mesh appears to be watertight.")





''' Main function Entrance '''

if __name__ == "__main__":
    main()