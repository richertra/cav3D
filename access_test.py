import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
import random

"""
This script is designed to analyze the differences between a target STL file and a set of other STL files in a directory.
The main goal is to identify a cavity region in the target mesh that deviates significantly from the other meshes.
The script performs the following steps:

1. Loads the target STL file and other STL files from the specified directory.
2. Resamples the meshes to a specified number of vertices (default is 10,000).
3. Calculates the vertex differences between the target mesh and each of the other meshes.
4. Identifies cavity points in the target mesh where the deviations exceed a specified threshold (default is 0.25).
5. Filters the cavity points by performing a Principal Component Analysis (PCA) on the original points and projecting the cavity points onto the first principal component.
6. Finds the bounding box corners within the filtered cavity points.
7. Finds the closest points in the filtered cavity points to the corner points.
8. Plots the mesh and the identified cavity points, highlighting the closest points with red stars.

The script returns the filtered cavity points and the closest points.

To use the script, you need to specify the target file, the directory containing the other STL files, the threshold for identifying cavity points, and the number of files to compare with the target file. The script will randomly select the specified number of other files from the directory for comparison.
"""


# Configuration
threshold = 0.25
num_files_to_compare = 6

def load_stl(file_path):
    return trimesh.load(file_path)

def resample_mesh(mesh, num_vertices=10000):
    # Sample points uniformly over the mesh surface
    points, _ = trimesh.sample.sample_surface(mesh, num_vertices)
    return points

def calculate_vertex_differences(points1, points2):
    kdtree = KDTree(points2)
    distances, _ = kdtree.query(points1)
    return distances

def identify_cavity_region(points, deviations_list, threshold=1.0):
    # Create a mask where all deviations exceed the threshold
    cavity_mask = np.all([deviations > threshold for deviations in deviations_list], axis=0)
    cavity_points = points[cavity_mask]
    return cavity_points

def filter_by_pca(points, cavity_points):
    # Perform PCA on the original points (First PCA)
    pca = PCA(n_components=3)
    pca.fit(points)
    
    # Project the cavity points onto the first principal component
    principal_axis = pca.components_[0]
    projected_points = cavity_points @ principal_axis
    
    # Define the lower two-thirds
    threshold_projection = np.percentile(points @ principal_axis, 66.67)
    lower_two_thirds_mask = projected_points < threshold_projection
    
    # Filter the cavity points
    filtered_cavity_points = cavity_points[lower_two_thirds_mask]
    return filtered_cavity_points

def find_bounding_box_corners(filtered_points):
    # Find the minimum and maximum bounds of the filtered points
    min_bounds = filtered_points.min(axis=0)
    max_bounds = filtered_points.max(axis=0)
    
    # Define the 8 corners of the bounding box within the filtered cavity points
    corners = np.array([[min_bounds[0], min_bounds[1], min_bounds[2]],
                        [min_bounds[0], min_bounds[1], max_bounds[2]],
                        [min_bounds[0], max_bounds[1], min_bounds[2]],
                        [min_bounds[0], max_bounds[1], max_bounds[2]],
                        [max_bounds[0], min_bounds[1], min_bounds[2]],
                        [max_bounds[0], min_bounds[1], max_bounds[2]],
                        [max_bounds[0], max_bounds[1], min_bounds[2]],
                        [max_bounds[0], max_bounds[1], max_bounds[2]]])
    return corners

def find_closest_points(filtered_points, corner_points):
    kdtree = KDTree(filtered_points)
    _, indices = kdtree.query(corner_points)
    closest_points = filtered_points[indices]
    return closest_points

def plot_mesh_and_points(mesh, cavity_points, closest_points, title="Mesh and Cavity Points"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface mesh with transparency
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, color='b', alpha=0.03, linewidth=0, antialiased=False)
    
    # Plot the green points
    ax.scatter(cavity_points[:, 0], cavity_points[:, 1], cavity_points[:, 2], c='g', marker='o', label='Cavity Points')
    # Plot the red closest points with stars
    ax.scatter(closest_points[:, 0], closest_points[:, 1], closest_points[:, 2], c='r', marker='*', s=200, label='Closest Points')
    
    # Set the view angle
    ax.view_init(elev=44, azim=-21)  # Adjust these values as needed
    
    ax.set_title(title)
    ax.legend()
    plt.show()

def main(target_file, directory, threshold, num_files_to_compare):
    # Load the target STL file
    mesh1 = load_stl(target_file)
    
    # Get all STL files in the directory
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.stl')]
    
    # Exclude the target file and randomly select the specified number of other files
    other_files = [f for f in all_files if f != target_file]
    if len(other_files) < num_files_to_compare:
        raise ValueError("Not enough other STL files found in the directory.")
    selected_files = random.sample(other_files, num_files_to_compare)
    
    # Load other meshes
    other_meshes = [load_stl(file) for file in selected_files]
    
    # Resample the meshes
    points1 = resample_mesh(mesh1)
    other_points = [resample_mesh(mesh) for mesh in other_meshes]
    
    # Calculate deviations
    deviations_list = [calculate_vertex_differences(points1, points) for points in other_points]
    
    # Identify cavity points
    cavity_points = identify_cavity_region(points1, deviations_list, threshold)
    
    # Filter cavity points by PCA (First PCA)
    filtered_cavity_points = filter_by_pca(points1, cavity_points)
    
    # Find the bounding box corners within the filtered cavity points
    corner_points = find_bounding_box_corners(filtered_cavity_points)
    
    # Find the closest points in filtered_cavity_points to the corner points
    closest_points = find_closest_points(filtered_cavity_points, corner_points)
    
    plot_mesh_and_points(mesh1, filtered_cavity_points, closest_points, title="Identified Cavity Points in Mesh")
    return filtered_cavity_points, closest_points

# Example usage
target_file = # ADAPTER ICI LE CHEMIN
directory = # ADAPTER ICI LE CHEMIN
cavity_points, closest_points = main(target_file, directory, threshold, num_files_to_compare)
