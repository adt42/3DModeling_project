# Just a little code to try to compare the groundtruh with what we got with Mast3r
#You need to change th path but you have the example that i used
#also this code isn't finish because i have a problem with the orientation of the two mesh


import pyvista as pv
import numpy as np
from ipywidgets import interact
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import time


#my_path = r"C:\Users\benja\3D_Modeling\ground_truth\non.glb"
my_path = r"C:\Users\benja\Downloads\tmpti3uiuo1_scene.glb"
my_path_2 = r"C:\Users\benja\Downloads\tmpx_cpv8st_scene.glb"
multiblock = pv.read((my_path))
multiblock_2 = pv.read((my_path_2))

ground_truth = pv.read(r"C:\Users\benja\Downloads\terrace\scan_clean\scan1.ply")

view = multiblock_2[0][0][0][0]
#print(multiblock[0][0][0][0].points)


def extract_data_from_multiblock(multiblock, level=0):
    indent = "  " * level  # Indentation for readability

    # Loop through each block in the MultiBlock
    for i in range(multiblock.n_blocks):
        block = multiblock[i]
        
        print(f"{indent}--- Block {i} ---")
        
        if block is None:
            print(f"{indent}This block is empty.")
            continue
        
        # Print the type of the block
        print(f"{indent}Block type: {type(block)}")
        
        # Check if the block is a MultiBlock itself (nested case)
        if isinstance(block, pv.MultiBlock):
            print(f"{indent}This is a nested MultiBlock.")
            extract_data_from_multiblock(block, level + 1)  # Recursively explore this nested MultiBlock
        
        # If it's a PolyData or other valid type, extract points, etc.
        elif isinstance(block, pv.PolyData):
            print(f"{indent}PolyData: {block.n_points} points, {block.n_cells} cells")
            print(f"{indent}Points:\n{block.points[:]}")

        # Handle other types like UnstructuredGrid
        elif isinstance(block, pv.UnstructuredGrid):
            print(f"{indent}UnstructuredGrid: {block.n_points} points, {block.n_cells} cells")
            print(f"{indent}Points:\n{block.points[:]}")

        # Handle StructuredPoints
        elif isinstance(block, pv.StructuredPoints):
            print(f"{indent}StructuredPoints: Dimensions {block.dimensions}")
        
        else:
            print(f"{indent}Unknown block type: {type(block)}")

        print(f"{indent}{'='*40}\n")

# Load your GLB or other file


# Start recursive extraction from the top level
#extract_data_from_multiblock(multiblock)


plotter = pv.Plotter()
# ground_truth = ground_truth.rotate_x(90)


g_bounds = ground_truth.bounds
v_bounds = view.bounds

size_a = (g_bounds[1] - g_bounds[0], g_bounds[3] - g_bounds[2], g_bounds[5] - g_bounds[4])
size_b = (v_bounds[1] - v_bounds[0], v_bounds[3] - v_bounds[2], v_bounds[5] - v_bounds[4])

scale_factors = [sb / sa if sa != 0 else 1 for sa, sb in zip(size_a, size_b)]

ground_truth = ground_truth.scale(scale_factors)

def get_center(bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2])

g_bounds = ground_truth.bounds
v_bounds = view.bounds

center_g = get_center(g_bounds)
center_v = get_center(v_bounds)

translation = center_v - center_g

ground_truth = ground_truth.translate(translation, inplace = True)

def compute_pca_axes(poly):
    pca = PCA(n_components=3)
    pca.fit(poly.points)
    return pca.components_

axes_g = compute_pca_axes(ground_truth)
axes_v = compute_pca_axes(view)

rotation_matrix = axes_v.T @ axes_g

rotation = R.from_matrix(rotation_matrix)

rotated_points = rotation.apply(ground_truth.points)

ground_truth.points = rotated_points

if np.linalg.det(rotation_matrix) < 0:
    print("Detected reflection (flip). Fixing by flipping an axis.")
    
plotter.add_mesh(view,style='points',cmap='gray', point_size=1, render_points_as_spheres=True)
plotter.add_mesh(ground_truth,style='points',cmap='jet', point_size=1, render_points_as_spheres=True)
plotter.show_axes()
plotter.show()

def epileptic_display():

    def update_colormap_jet():
        plotter.clear()
        plotter.add_mesh(multiblock[0][0][0][0],style='points',cmap='spring', point_size=1, render_points_as_spheres=True)
        plotter.update()
        
        update_colormap_viridis()
        
    
    def update_colormap_viridis():
        plotter.clear()
        plotter.add_mesh(multiblock[0][0][0][0],style='points',cmap ='jet', point_size=3, render_points_as_spheres=True)
        plotter.update()
        update_colormap_jet()

    plotter.add_mesh(multiblock[0][0][0][0],style='points',cmap ='jet', point_size=3, render_points_as_spheres=True)
    plotter.show(interactive_update=True)    
    update_colormap_jet()

#epileptic_display()
#plotter.add_mesh(multiblock[0][0][0][0],style='points',cmap ='jet', point_size=2, render_points_as_spheres=True)
#plotter.add_mesh(multiblock_2[0][0][0][0],style='points',cmap ='gray', point_size=2, render_points_as_spheres=True)
#plotter.add_mesh(ground_truth,style='points',cmap ='gray', point_size=2, render_points_as_spheres=True)
#plotter.show()    




