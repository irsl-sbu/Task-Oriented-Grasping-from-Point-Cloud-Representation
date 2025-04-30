# Helper functions to perform a simple heuristic-based collision 
# and penetration check between an end-effector pose and a cuboid
# By: Aditya Patankar

import numpy as np
from numpy import linalg as la

def check_point_penetration(ee_pose, dims):
    """
    Function the check whether a point is inside or outside an axis aligned cuboidal object:

    Args: 

    Returns:
    """

    x_min, x_max = -dims[0], dims[0]
    y_min, y_max = -dims[1], dims[1]
    z_min, z_max = -dims[2], dims[2]

    INSIDE = False

    if (x_min <= ee_pose[0, 3] <= x_max and
        y_min <= ee_pose[1, 3] <= y_max and
        z_min <= ee_pose[2, 3] <= z_max):

        INSIDE = True
    
    return INSIDE


def collision_helper(ee_pose, plane_points, center_point, grasp_center):
    """
    Function to check whether a vector from the origin of the end-effector to the grasp center intersects 
    with one of the faces of the bounding box: 

    Args:

    Returns:
    """

    vect = grasp_center - ee_pose[0:3, 3]

    # We need to compute the distance of the grasp center to the first plane only:
    u = np.subtract(plane_points[3, :], plane_points[0, :])
    v = np.subtract(plane_points[1, :], plane_points[0, :])

    # Normal vector corresponding to the plane:
    n = np.cross(u, v)
    # Computing the D in the equation of the plane Ax + By + Cz + D = 0:
    d = -np.dot(n, center_point)
    return n, d

def check_collision(ee_pose, vertices, grasp_center, dims):
    """
    Function to check to whether a end-effector pose would "potentially" collide with an object:

    Args:

    Returns:
    """

    # We need to check for collisions for all 5 of the approach directions:
    COLLISION = False

    # Approach Direction 1:
    plane_points_1 = np.asarray([vertices[3], vertices[0], vertices[1], vertices[6]])
    center_point_plane_1 = np.asarray([0, plane_points_1[0, 1], 0])
    n1, d1 = collision_helper(ee_pose, plane_points_1, center_point_plane_1, grasp_center)
    dist_ee_1 = np.dot(n1, ee_pose[0:3, 3]) + d1
    dist_gc_1 = np.dot(n1, grasp_center) + d1

    # Approach Direction 2:
    plane_points_2 = np.asarray([vertices[3], vertices[6], vertices[4], vertices[5]])
    center_point_plane_2 = np.asarray([0, 0, plane_points_2[0, 2]])
    n2, d2 = collision_helper(ee_pose, plane_points_2, center_point_plane_2, grasp_center)
    dist_ee_2 = np.dot(n2, ee_pose[0:3, 3]) + d2
    dist_gc_2 = np.dot(n2, grasp_center) + d2

    # Approach direction 3:
    plane_points_3 = np.asarray([vertices[3], vertices[5], vertices[2], vertices[0]])
    center_point_plane_3 = np.asarray([plane_points_3[0, 0], 0, 0])
    n3, d3 = collision_helper(ee_pose, plane_points_3, center_point_plane_3, grasp_center)
    dist_ee_3 = np.dot(n3, ee_pose[0:3, 3]) + d3
    dist_gc_3 = np.dot(n3, grasp_center) + d3

    # Approach Direction 4:
    plane_points_4 = np.asarray([vertices[4], vertices[7], vertices[2], vertices[5]])
    center_point_plane_4 = np.asarray([0, plane_points_4[0, 1], 0])
    n4, d4 = collision_helper(ee_pose, plane_points_4, center_point_plane_4, grasp_center)
    dist_ee_4 = np.dot(n4, ee_pose[0:3, 3]) + d4
    dist_gc_4 = np.dot(n4, grasp_center) + d4 

    # Approach direction 5:
    plane_points_5 = np.asarray([vertices[6], vertices[1], vertices[7], vertices[4]])
    center_point_plane_5 = np.asarray([plane_points_5[0, 0], 0, 0])
    n5, d5 = collision_helper(ee_pose, plane_points_5, center_point_plane_5, grasp_center)
    dist_ee_5 = np.dot(n5, ee_pose[0:3, 3]) + d5
    dist_gc_5 = np.dot(n5, grasp_center) + d5

    # Check whether the end-effector pose is inside the cuboid:
    INSIDE = check_point_penetration(ee_pose, dims)
    
    if (dist_ee_1 * dist_gc_1 <= 0 or 
        dist_ee_2 * dist_gc_2 <= 0 or
        dist_ee_3 * dist_gc_3 <= 0 or
        dist_ee_4 * dist_gc_4 <= 0 or
        dist_ee_5 * dist_gc_5 <= 0):

        COLLISION = True

    return COLLISION, INSIDE