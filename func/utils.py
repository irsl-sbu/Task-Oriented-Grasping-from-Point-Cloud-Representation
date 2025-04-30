# Helper functions to aid with visualization and plotting.
# By: Aditya Patankar

import numpy as np
import matplotlib.pyplot as plt

def plot_cube(vertices):
    """
    Function for plotting a CUBE.

    Args: 
    
    Returns:
    """
    # Processing the faces for the cube: 
    # The array 'faces_vertices' is based on a predefined convention
    faces_vertices = np.asarray([[0,4,7,3], [0,1,5,4], [1,2,6,5], [3,7,6,2], [4,5,6,7], [0,1,2,3]])
    
    # Initialize a list of vertex coordinates for each face
    faces = []
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    faces.append(np.zeros([4,3]))
    
    for i, v in enumerate(faces_vertices):
        for j in range(faces_vertices.shape[1]):
            faces[i][j, 0] = vertices[faces_vertices[i,j],0]
            faces[i][j, 1] = vertices[faces_vertices[i,j],1]
            faces[i][j, 2] = vertices[faces_vertices[i,j],2]

    return faces

def transform_vertices(g, vertices):
    """
    Function to transform the vertices of a cuboid and express them with respect to the current pose.

    # NOTE: This function assumes that the pose in which the vertices are expressed initially is identity
    
    Args:

    Returns:

    """
    transformed_vertices = np.zeros([vertices.shape[0], vertices.shape[1]])
    for i, vertex in enumerate(vertices):
        hom_vertex = np.matmul(g, np.reshape(np.append(vertex, 1), [4,1]))
        transformed_vertices[i,:] = np.reshape(hom_vertex[0:3, :], [3])
    return transformed_vertices
            
def plot_reference_frames(R, p, scale_value, length_value, ax):
    """
    Function to plot a reference frame.
    
    Args:

    Returns: 

    """
    ax.quiver(p[0], p[1], p[2], scale_value*R[0, 0], scale_value*R[1, 0], scale_value*R[2, 0], color = "r", arrow_length_ratio = length_value)
    ax.quiver(p[0], p[1], p[2], scale_value*R[0, 1], scale_value*R[1, 1], scale_value*R[2, 1], color = "g", arrow_length_ratio = length_value)
    ax.quiver(p[0], p[1], p[2], scale_value*R[0, 2], scale_value*R[1, 2], scale_value*R[2, 2], color = "b", arrow_length_ratio = length_value)
    return ax