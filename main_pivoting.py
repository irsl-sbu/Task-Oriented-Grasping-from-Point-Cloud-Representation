# Open3D for point cloud processing and visualization
import open3d as o3d

import numpy as np
from numpy import linalg as la
import csv
import math

# Matplotlib libraries for plotting and visualization in Python:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Functionalities for point cloud processing and computing the ideal grasping region:
from point_cloud_module.process_point_cloud import point_cloud

from time import perf_counter
import argparse

# CUDA for PyTorch:
device = "cpu"

''' Function to read a CSV file: '''
def read_csv(filename):
    columns = []
    datapoints = []
    with open(filename) as file:
        csvreader = csv.reader(file)
        for col in csvreader:
            columns.append(col)

    for col in columns:
        datapoint = np.asarray(col, dtype = np.float64)
        datapoints.append(datapoint)
    return datapoints

''' Function to get a rotation matrix given an axis and a angle:
    Note the angle should be in radians and axis should be a numpy array.'''
def axis_angle_to_rot(axis, angle):
    axis = axis/la.norm(axis)

    omega = np.asarray([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])

    R = np.eye(3) + (math.sin(angle)*omega) + ((1 - math.cos(angle))*la.matrix_power(omega, 2))
    return R

''' Function to compute a 4x4 transformation matrix given a screw axis and the magnitude of displacement about
    that axis:
    Note: By default for this function we always assume that the pitch is 0 i.e. pure rotation about the screw axis. '''
def get_transformation_for_screw(axis, pitch, theta, point):
    R = axis_angle_to_rot(axis, theta)
    p = np.reshape(np.matmul((np.eye(3) - R),point)  + (pitch*theta*axis), [3])
    g = np.eye(4,4)
    g[0:3, 0:3] = R
    g[0:3, 3] = p
    return g

'''Function builds an object of the point_cloud class'''
def build_cloud_object(cloud_object, pcd):
    cloud_object.processed_cloud = pcd
    cloud_object.points = np.asarray(cloud_object.processed_cloud.points)

    # Computing the normals for this point cloud:
    cloud_object.processed_cloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    cloud_object.processed_cloud.estimate_normals()
    cloud_object.processed_cloud.orient_normals_consistent_tangent_plane(30)
    cloud_object.normals_base_frame = np.asarray(cloud_object.processed_cloud.normals)
    return cloud_object

'''Function to compute the final grasp pose after the pivoting motion'''
def pivoting_motion(cloud_object, num_pose):
    # Computing the final end-effector pose after grasping based on the screw axis:
    pitch = 0
    theta = math.radians(90)
    g = get_transformation_for_screw(cloud_object.screw_axis, pitch, theta, cloud_object.point)
    
    # desired_grasp_pose_object_frame = cloud_object.computed_end_effector_poses[num_pose]
    # desired_grasp_pose_object_frame = cloud_object.approach_dir_2_poses[num_pose]
    desired_grasp_pose_object_frame = cloud_object.approach_dir_other_poses[num_pose]
    
    # desired_grasp_pose_object_frame = cloud_object.computgrasping_poses, grasp_flaged_end_effector_poses[num_pose]
    desired_grasp_pose_object_frame_final = np.matmul(g, desired_grasp_pose_object_frame)
    desired_grasp_pose_base_frame_final = np.eye(4,4)
    desired_grasp_pose_base_frame_final[0:3, 0:3] = np.matmul(cloud_object.R_bounding_box, desired_grasp_pose_object_frame_final[0:3, 0:3])
    desired_grasp_pose_base_frame_final[0:3, 3] = np.reshape(cloud_object.p_bounding_box + np.dot(cloud_object.R_bounding_box, np.reshape(desired_grasp_pose_object_frame_final[0:3, 3], [3,1])), [3])

    # Storing the poses for preprocessing: 
    # grasping_poses = np.asarray([cloud_object.computed_end_effector_poses_inter_base[num_pose], cloud_object.computed_end_effector_poses_base[num_pose], desired_grasp_pose_base_frame_final])
    grasping_poses = np.asarray([cloud_object.approach_dir_other_inter_poses_base[num_pose], cloud_object.approach_dir_other_poses_base[num_pose], desired_grasp_pose_base_frame_final])
    # grasping_poses = np.asarray([cloud_object.approach_dir_2_inter_poses_base[num_pose], cloud_object.approach_dir_2_poses_base[num_pose], desired_grasp_pose_base_frame_final])
    
    grasping_poses =np.reshape(grasping_poses, [len(grasping_poses)*4,4])
    # The grasp_flag depends on the type of motion:
    grasp_flag = np.asarray([0, 1, 1], dtype=np.int32)

    return desired_grasp_pose_base_frame_final, grasping_poses, grasp_flag

'''Function to visualize the results: '''
def visualize(cloud_object, num_pose, desired_grasp_pose_base_frame_final):
    # Extracting the points for visualization purposes:
    x_points = np.reshape(cloud_object.points[:, 0], [cloud_object.points.shape[0],1])
    y_points = np.reshape(cloud_object.points[:, 1], [cloud_object.points.shape[0],1])
    z_points = np.reshape(cloud_object.points[:, 2], [cloud_object.points.shape[0],1])

    x_transformed_points = np.reshape(cloud_object.transformed_points_object_frame[:, 0], [cloud_object.transformed_points_object_frame.shape[0],1])
    y_transformed_points = np.reshape(cloud_object.transformed_points_object_frame[:, 1], [cloud_object.transformed_points_object_frame.shape[0],1])
    z_transformed_points = np.reshape(cloud_object.transformed_points_object_frame[:, 2], [cloud_object.transformed_points_object_frame.shape[0],1])

    ## Plot 1:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    ax1.grid(False)
    cloud_object.vertices = cloud_object.transformed_vertices_object_frame
    cloud_object.plot_cube()
    ax1.add_collection3d(Poly3DCollection(cloud_object.faces, linewidths=1, edgecolors='b', alpha=.25))
    ax1.scatter(x_transformed_points, y_transformed_points, z_transformed_points, s = 0.2)

    # Visualize the screw axis: 
    ax1.scatter(cloud_object.point[0], cloud_object.point[1], cloud_object.point[2], marker = '*', s = 100, color = 'r')
    ax1.quiver(cloud_object.point[0], cloud_object.point[1], cloud_object.point[2], 0.25*cloud_object.screw_axis[0], 0.25*cloud_object.screw_axis[1], 0.25*cloud_object.screw_axis[2], color = "r", arrow_length_ratio = 0.25)

    # Base reference Frame: 
    cloud_object.R = cloud_object.R_base
    cloud_object.p = cloud_object.p_base
    cloud_object.scale_value = 0.45
    cloud_object.length_value = 0.25
    ax1 = cloud_object.plot_reference_frames(ax1)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-0.4, 0.4)
    ax1.set_ylim(-0.4, 0.4)
    ax1.set_zlim(-0.4, 0.4)

    # Plot 2: 
    # Plotting and visualizing the grasp centers
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    ax2.scatter(x_transformed_points, y_transformed_points, z_transformed_points, s = 0.2)
    ax2.scatter(cloud_object.point[0], cloud_object.point[1], cloud_object.point[2], marker = '*', s = 100, color = 'r')
    ax2.quiver(cloud_object.point[0], cloud_object.point[1], cloud_object.point[2], 0.25*cloud_object.screw_axis[0], 0.25*cloud_object.screw_axis[1], 0.25*cloud_object.screw_axis[2], color = "r", arrow_length_ratio = 0.25)
    
    # Base reference Frame: 
    cloud_object.R = cloud_object.R_base
    cloud_object.p = cloud_object.p_base
    cloud_object.scale_value = 0.45
    cloud_object.length_value = 0.25
    ax2 = cloud_object.plot_reference_frames(ax2)

    # Grasp centers:
    for i,v in enumerate(cloud_object.computed_end_effector_poses_base):
        g = cloud_object.grasp_centers[i]
        ax2.scatter(g[0], g[1], g[2], marker = '*', s = 20, color = 'r')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim(-0.4, 0.4)
    ax2.set_ylim(-0.4, 0.4)
    ax2.set_zlim(-0.4, 0.4)

    # Plot 3
    # Visualizing the pregrasp pose, grasping pose and the final pose after pivoting in the robot base reference frame:
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection='3d')
    ax3.grid(False)
    ax3.scatter(x_points, y_points, z_points, s = 0.2)

    # Base reference Frame:
    cloud_object.R = cloud_object.R_base
    cloud_object.p = cloud_object.p_base
    cloud_object.scale_value = 0.25
    cloud_object.length_value = 0.15
    ax3 = cloud_object.plot_reference_frames(ax3)

    # Bounding box reference frame:
    cloud_object.R = cloud_object.R_bounding_box
    cloud_object.p = cloud_object.p_bounding_box
    cloud_object.scale_value = 0.15
    cloud_object.length_value = 0.15
    ax3 = cloud_object.plot_reference_frames(ax3)

    # Pregrasp pose:
    # pose = cloud_object.computed_end_effector_poses_inter_base[num_pose]
    pose = cloud_object.approach_dir_other_inter_poses_base[num_pose]
    # pose = cloud_object.approach_dir_2_inter_poses_base[num_pose]
    cloud_object.R = pose[0:3, 0:3]
    cloud_object.p = pose[0:3, 3]
    cloud_object.scale_value = 0.025
    cloud_object.length_value = 0.025
    ax3 = cloud_object.plot_reference_frames(ax3)

    # Grasp pose:
    # pose = cloud_object.computed_end_effector_poses_base[num_pose]
    pose = cloud_object.approach_dir_other_poses_base[num_pose]
    # pose = cloud_object.approach_dir_2_poses_base[num_pose]
    cloud_object.R = pose[0:3, 0:3]
    cloud_object.p = pose[0:3, 3]
    cloud_object.scale_value = 0.025
    cloud_object.length_value = 0.025
    ax3 = cloud_object.plot_reference_frames(ax3)

    # Pose after pivoting:
    cloud_object.R = desired_grasp_pose_base_frame_final[0:3, 0:3]
    cloud_object.p = desired_grasp_pose_base_frame_final[0:3, 3]
    cloud_object.scale_value = 0.025
    cloud_object.length_value = 0.025
    ax3 = cloud_object.plot_reference_frames(ax3)

    plt.show()

# MAIN FUNCTION: 
if __name__ == "__main__":
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

    # Create an ArgumentParser Object
    parser = argparse.ArgumentParser(description='Task-Oriented Grasp Synthesis')

    # Add a command-line argument for the input filename
    parser.add_argument('--filename', type=str, help='Path to the input point cloud file')
    parser.add_argument('--visualize', action='store_true', help='Enable visualize flag')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Creating the cloud object and loading the necessary file:
    cloud_object = point_cloud()

    # Read the point cloud data from the specified file    
    pcd = o3d.io.read_point_cloud(args.filename)
    cloud_object = build_cloud_object(cloud_object, pcd)

    # Specifying gripper tolerances:
    cloud_object.gripper_width_tolerance = 0.08
    cloud_object.gripper_height_tolerance = 0.041

    # Attributes to compute the location of the reference frame at the flange for the grasp pose and pre-grasp pose
    cloud_object.g_delta = 0.0625
    cloud_object.g_delta_inter = 0.0925

    # STARTING TOTAL TIME:
    total_time_start = perf_counter()

    # Computing the bounding boxes corresponding to the object point cloud: 
    cloud_object.compute_bounding_box()

    print('Bounding box computed!')

    # APPROXIMATING THE TASK-DEPENDENT GRASP METRIC:
    # TASK: Pivoting a Cuboidal box:
    # Screw Parameters:
    # Axis 1:
    cloud_object.screw_axis = np.asarray([0, 1, 0])
    cloud_object.point = np.asarray([cloud_object.transformed_vertices_object_frame[1,0], np.divide((cloud_object.transformed_vertices_object_frame[1,1] + cloud_object.transformed_vertices_object_frame[7,1]),2), cloud_object.transformed_vertices_object_frame[1,2]])
    cloud_object.moment = np.cross(cloud_object.point, cloud_object.screw_axis)

    cloud_object.generate_contacts()
    
    print('Antipodal contact locations sampled from the surface of the bounding box!')

    # STARTING METRIC PREDICTION TIME:
    metric_time_start = perf_counter()
    
    # cloud_object.predict_metric()
    cloud_object.predict_metric_generic()

    print('Metric Value Predicted using Neural Network ... ')

    # END METRIC COMPUTATION TIME:
    metric_time_end = perf_counter()

    # Computing the ideal grasping region:
    cloud_object.get_ideal_grasping_region()

    # Computing the end-effector poses based on the ideal grasping region:
    cloud_object.get_end_effector_poses()

    # END TOTAL TIME:
    total_time_end = perf_counter()

    # Time required to sample the antipodal contact locations
    print(f'Total time required for the algorithm:  {total_time_end-total_time_start} seconds')

    # Time required to compute the metric values
    print(f'Time required for computing the metric: {metric_time_end-metric_time_start} seconds')

    # Number of contact locations generated on the surface:
    print("Number of contact locations generated on the bounding box: ", cloud_object.sampled_c1.shape[0])

    # Computed end-effector poses:
    print("Number of end-effector poses computed: ", len(cloud_object.computed_end_effector_poses_base))

    # Selecting a random pre-grasp and grasp pose:
    num_pose = 10
    # Computing the final grasp pose after pivoting the cuboidal object:
    desired_grasp_pose_base_frame_final, grasping_poses, grasp_flag = pivoting_motion(cloud_object, num_pose)
    
    if args.visualize:
        visualize(cloud_object, num_pose, desired_grasp_pose_base_frame_final)

    
