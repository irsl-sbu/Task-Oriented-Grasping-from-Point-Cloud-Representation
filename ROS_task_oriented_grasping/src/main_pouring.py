#!/usr/bin/env python
# Python script to output the end-effector poses for performing a particular task using sensor based information
# By: Aditya Patankar

import rospy
import actionlib
import cv2
import pyrealsense2 as rs2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge, CvBridgeError
from task_based_grasping.msg import MotionExecutionAction, MotionExecutionGoal, MotionExecutionActionFeedback, MotionExecutionActionResult

# Open3D for point cloud processing and visualization
import open3d as o3d

# Numpy packages 
import numpy as np
import csv
import math
from numpy import linalg as la
from scipy.spatial.transform import Rotation as Rot

# Matplotlib libraries for plotting and visualization in Python:
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

# Pytorch packages and classes for Neural Network based prediction: 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from neuralNet import metricNN
from dataLoader import metricNNDataset
from dataLoader import ToTensor

# Packages and classes for Point Cloud Processing:
from collections import Counter
from process_point_cloud import pointCloud

import pdb

# Importing custom classes: 

from get_point_cloud_semantic_seg import realsenseSubscriber

# from process_point_cloud_baseline1_apprdir2 import pointCloud
# from process_point_cloud_baseline1_pouring import pointCloud
# from process_point_cloud_baseline1 import pointCloud

# from process_point_cloud_baseline2_apprdir2 import pointCloud
# from process_point_cloud_baseline2_pouring import pointCloud
# from process_point_cloud_baseline2 import pointCloud

# from process_point_cloud_baseline3_apprdir2 import pointCloud
# from process_point_cloud_baseline3_apprdir2_trial import pointCloud
from process_point_cloud_baseline3_pouring import pointCloud
# from process_point_cloud_baseline3 import pointCloud
# from process_point_cloud_baseline3_trial import pointCloud

# from process_point_cloud import pointCloud
# from process_point_cloud_vanilla_ver2 import pointCloud

# Function to read a CSV file:
def readCSV(filename):
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

# Function to get a rotation matrix given an axis and a angle:
# Note the angle should be in radians and axis should be a numpy array.
def axisAngleToRot(axis, angle):
    axis = axis/la.norm(axis)

    omega = np.asarray([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])

    R = np.eye(3) + (math.sin(angle)*omega) + ((1 - math.cos(angle))*la.matrix_power(omega, 2))
    return R

# Function to compute a 4x4 transformation matrix given a screw axis and the magnitude of displacement about
# that axis:
# Note: By default for this function we always assume that the pitch is 0 i.e. pure rotation about the 
# screw axis. 
def getTransformationForScrew(axis, pitch, theta, point):
    R = axisAngleToRot(axis, theta)
    p = np.reshape(np.matmul((np.eye(3) - R),point)  + (pitch*theta*axis), [3])
    g = np.eye(4,4)
    g[0:3, 0:3] = R
    g[0:3, 3] = p
    return g

# MAIN FUNCTION: 
if __name__ == "__main__":

    rospy.init_node('listenerNode', anonymous=True)
    imageTopic = "/camera/color/image_rect_color"
    depthTopic = "/camera/aligned_depth_to_color/image_raw"
    cameraInfoTopic = "/camera/aligned_depth_to_color/camera_info"
    cameraPoseTopic = "/panda_camera_pose"
    path = "/home/Aditya/Saved_Images/"+"cheezit"

    print('Realsense Subscriber Object initialized')
    print('---------------------------------------')

    realsenseObject = realsenseSubscriber(imageTopic, depthTopic, cameraInfoTopic, cameraPoseTopic, path)
    # rospy.spin()

    dataPointsSegmented = realsenseObject.getCoordSemanticSeg()

    print(dataPointsSegmented.shape)

    # Creating a Open3d PointCloud Object for the cloud corresponding to just the bounding box
    objectCloud = o3d.geometry.PointCloud()
    objectCloud.points = o3d.utility.Vector3dVector(dataPointsSegmented.astype(np.float64))
    objectCloud.paint_uniform_color([0, 0, 1])

    # Visualizing just the CheezIt point cloud using open3D:
    o3d.visualization.draw_geometries([objectCloud])

    # Saving the point cloud in the necessary format:
    np.savetxt("dataPointsSegmented_pringles_pouring_pose7_EE8.csv", dataPointsSegmented, delimiter= ",")
    o3d.io.write_point_cloud("pringles_pouring_pose7_EE8_segmented.ply", objectCloud)
    print('Object point cloud corresponding to the bounding box saved!')

    # Creating a pointCloud() object:
    cloud_object = pointCloud()
    cloud_object.cloud = objectCloud

    # Rotation matrix and position vector for the robot base or world reference frame: 
    cloud_object.R_base = np.identity(3)
    cloud_object.p_base = np.zeros([3,1])

    ## Pose of the camera reference frame with respect to the base reference frame: 
    cloud_object.g_base_cam = realsenseObject.getEndEffectorPose()
    print('Printing the Pose of the camera with respect to the base reference frame!')
    print(cloud_object.g_base_cam)

    # Transforming the point cloud in the Panda base reference frame: 
    cloud_object.transformToBase()

    # Visualizing the downsampled point cloud. 
    print('Cloud transformed to base')
    # o3d.visualization.draw_geometries([cloud_object.cloud])

    # Downsample it and inspect the normals
    cloud_object.cloud = cloud_object.cloud.voxel_down_sample(voxel_size=0.009)
    cloud_object.removePlaneSurface()

    # Visualizing the downsampled point cloud. 
    print('Plane surface removed!')
    # o3d.visualization.draw_geometries([cloud_object.cloud])

    # Specifying parameters for DBSCAN Clustering:
    # Just like the parameters for downsampling even the parameters for DBSCAN Clustering are dependent on the 
    # units used computing and extracting the point cloud data.
    cloud_object.eps = 0.02
    cloud_object.min_points = 10
    cloud_object.getObjectPointCloud()

    # Visualizing the downsampled point cloud. 
    print('Visualizing the processed cloud!')
    # o3d.visualization.draw_geometries([cloud_object.processed_cloud])

    print('Point cloud transformation, plane surface removal and object point cloud extraction DONE!')

    # Computing the normals for this point cloud:
    cloud_object.processed_cloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))

    # Estimating new normals for the point cloud
    cloud_object.processed_cloud.estimate_normals()
    cloud_object.processed_cloud.orient_normals_consistent_tangent_plane(30)

    # Storing the normals as a numpy array: 
    cloud_object.normals_base_frame = np.asarray(cloud_object.processed_cloud.normals)

    # The object is aligned with the axis of the world/robot base reference frame.
    cloud_object.bounding_box_flag = 0

    # Computing the bounding boxes corresponding to the object point cloud: 
    cloud_object.computeAABB()

    # Getting the pose of the bounding box. This pose serves as our object reference frame:
    cloud_object.getPoseBoundingBox()

    # Transforming the points to the object reference frame:
    cloud_object.transformToObjectFrame()

    '''# If the object is not aligned with the axes of the robot base reference frame then we need to execute the following:
    cloud_object.computeOBB()

    print('Oriented Bounding Box Computed!')

    # We now change the bounding box flag and use the oriented bounding box to sample the contacts:
    cloud_object.bounding_box_flag = 1

    # Getting the pose of the bounding box:
    cloud_object.getPoseBoundingBox()

    # Transforming the points to the object reference frame:
    cloud_object.transformToObjectFrame()'''

    print('Points and Bounding Box vertices transformed to Object Reference Frame!')

    '''# Transforming point from world frame to body frame:
    point_world = np.reshape(np.asarray([0.305, 0.415, 0.115, 1]), [4, 1])
    g_base_object = np.eye(4,4)
    g_base_object[0:3, 0:3] = cloud_object.R_bounding_box
    g_base_object[0:3, 3] = np.reshape(cloud_object.p_bounding_box, [3])
    g_object_base = la.inv(g_base_object)
    point_object_frame = np.matmul(g_object_base, point_world)

    print("point world: ", point_world)
    cloud_object.point = np.reshape(point_object_frame[0:3], [3])
    print(cloud_object.point)
    print(cloud_object.point.shape)
    print("p_bounding_box: ", cloud_object.p_bounding_box)'''

    "pdb.set_trace()"

    # Specifying the screw parameters: 

    # Pouring to the right:
    '''screw_params = readCSV('PouringScrewParametersRight.csv')
    cloud_object.screw_axis = screw_params[0][0:3]
    cloud_object.point = screw_params[0][3:6]
    cloud_object.moment = np.cross(cloud_object.point, cloud_object.screw_axis)'''

    # Pouring to the left:
    screw_params = readCSV('PouringScrewParametersRight.csv')
    cloud_object.screw_axis = -1*screw_params[0][0:3]
    cloud_object.point = screw_params[0][3:6] + np.asarray([0.0001, -0.08, -0.09])
    cloud_object.moment = np.cross(cloud_object.point, cloud_object.screw_axis)
    
    print("Screw Axis: ", cloud_object.screw_axis)
    print("Point: ", cloud_object.point)
    print(cloud_object.point.shape)
    print("Moment: ", cloud_object.moment)

    # Define the increment:
    cloud_object.increment = 0.01

    cloud_object.sampleContactsYZ()
    # cloud_object.sampleContactsXZ()
    
    print('Antipodal contact locations sampled from the surface of the bounding box!')

    # NEURAL NETWORK BASED METRIC PREDICTION: 
    # HYPER PARAMETERS: 
    # (1) Network Size:
    cloud_object.input_size = 12
    cloud_object.hidden_size1 = 8
    cloud_object.hidden_size2 = 3

    # Batch size to divide the dataset into batches:
    cloud_object.batch_size = 2

    '''The DataLoader class needs to be changed here so that we don't need to read data from csv files. 
        We can directly initialize an object of the DataLoader class.'''

    testing_dataset = metricNNDataset(cloud_object.x_data, cloud_object.y_data, transform = transforms.Compose([ToTensor()]))

    # Using the DataLoader class from Pytorch to shuffle the data and divide it into batches:
    testLoader = DataLoader(testing_dataset, cloud_object.batch_size, 
                                shuffle=True, num_workers=0)

    # Defining the neural network model
    model = metricNN(cloud_object.input_size, cloud_object.hidden_size1, cloud_object.hidden_size2)

    # Loading the trained models: 
    PATH = '/home/marktwo/Aditya/RAL_Experiments/Baseline_2/Trained_Models/model8_150epochs_lr001.pth'
    # PATH = '/home/marktwo/Aditya/RAL_Experiments/Baseline_2/Trained_Models/model6_100epochs_lr001.pth'

    model.load_state_dict(torch.load(PATH))
    print('Weights Loaded!')

    print("Model: ", model)

    cloud_object.predictMetric(model, testLoader)
    print('Metric Value Predicted using Neural Network ... ')

    # Normalizing the values between 0 and 1:
    normalized_data = (cloud_object.predicted - np.min(cloud_object.predicted))/(np.max(cloud_object.predicted) - np.min(cloud_object.predicted))
    cloud_object.predicted = normalized_data

    # Function call to project the points: 
    cloud_object.projectPointsYZ()
    # cloud_object.projectPointsXZ()

    print('Points projected on the surface now generating grid ...')

    cloud_object.generateGridYZ()
    # cloud_object.generateGridXZ()

    cloud_object.checkOccupancyYZ()
    # cloud_object.checkOccupancyXZ()

    print('Occupancy check completed, proceed towards sampling poses ... ')

    # Creating a new point cloud object and assigning the transformed points expressed in the base frame to it.
    # This is being done so that the normals can be estimated:
    cloud_object.cloud_object_frame = o3d.geometry.PointCloud()
    cloud_object.cloud_object_frame.points = o3d.utility.Vector3dVector(cloud_object.transformed_points_object_frame)
    cloud_object.cloud_object_frame.paint_uniform_color([0, 0, 1])

    # Computing the normals for the newly transformed points:
    cloud_object.cloud_object_frame.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    # Estimating new normals for the point cloud
    cloud_object.cloud_object_frame.estimate_normals()
    cloud_object.cloud_object_frame.orient_normals_consistent_tangent_plane(30)

    # Saving the point cloud which has been transformed and expressed in its object reference frame:
    np.savetxt("pringles_pouring_pose7_EE8_object_frame.csv", cloud_object.transformed_points_object_frame, delimiter= ",")
    o3d.io.write_point_cloud("pringles_pouring_pose7_EE8_segmented.ply", cloud_object.cloud_object_frame)
    print('Object point cloud expressed in its object frame saved!')

    # Storing the normals as a numpy array: 
    cloud_object.normals_object_frame = np.asarray(cloud_object.cloud_object_frame.normals)

    # Specifying gripper width tolerance:
    cloud_object.gripper_width_tolerance = 0.10

    # Specifying the gripper height tolerance:
    cloud_object.gripper_height_tolerance = 0.07
    
    # Computing the ideal grasping region:
    cloud_object.getIdealGraspingRegion()

    # Computing the bounding box corresponding to the points in the ideal grasping region:
    # cloud_object.getBBIdealGraspingRegion()

    # Computing the end-effector poses based on the ideal grasping region:
    cloud_object.getEndEffectorPoses()

    print('Number of end-effector poses computed: ', len(cloud_object.computed_end_effector_poses_base))

    # Selecting a particular pose corresponding to a specified approach direction:
    num_pose = 0

    ''' PLOTTING AND VISUALIZATION: '''

    # Extracting the points for visualization purposes: 
    x_coords = np.reshape(cloud_object.aligned_bounding_box_vertices[:,0], [cloud_object.aligned_bounding_box_vertices.shape[0],1])
    y_coords = np.reshape(cloud_object.aligned_bounding_box_vertices[:,1], [cloud_object.aligned_bounding_box_vertices.shape[0],1])
    z_coords = np.reshape(cloud_object.aligned_bounding_box_vertices[:,2], [cloud_object.aligned_bounding_box_vertices.shape[0],1])

    x_points = np.reshape(cloud_object.points[:, 0], [cloud_object.points.shape[0],1])
    y_points = np.reshape(cloud_object.points[:, 1], [cloud_object.points.shape[0],1])
    z_points = np.reshape(cloud_object.points[:, 2], [cloud_object.points.shape[0],1])

    x_transformed_points = np.reshape(cloud_object.transformed_points_object_frame[:, 0], [cloud_object.transformed_points_object_frame.shape[0],1])
    y_transformed_points = np.reshape(cloud_object.transformed_points_object_frame[:, 1], [cloud_object.transformed_points_object_frame.shape[0],1])
    z_transformed_points = np.reshape(cloud_object.transformed_points_object_frame[:, 2], [cloud_object.transformed_points_object_frame.shape[0],1])

    x_sampled_c1 = cloud_object.sampled_c1[:, 0]
    y_sampled_c1 = cloud_object.sampled_c1[:, 1]
    z_sampled_c1 = cloud_object.sampled_c1[:, 2]
    
    x_sampled_c2 = cloud_object.sampled_c2[:, 0]
    y_sampled_c2 = cloud_object.sampled_c2[:, 1]
    z_sampled_c2 = cloud_object.sampled_c2[:, 2]

    x_c1 = cloud_object.test_datapoints[:, 0]
    y_c1 = cloud_object.test_datapoints[:, 1]
    z_c1 = cloud_object.test_datapoints[:, 2]

    x_c2 = cloud_object.test_datapoints[:, 3]
    y_c2 = cloud_object.test_datapoints[:, 4]
    z_c2 = cloud_object.test_datapoints[:, 5]

    x_projected = cloud_object.projected_points[:, 0]
    y_projected = cloud_object.projected_points[:, 1]
    z_projected = cloud_object.projected_points[:, 2]

    X = np.reshape(cloud_object.X_grid_points, [cloud_object.X_grid_points.shape[0]*cloud_object.X_grid_points.shape[1]])
    Y = np.reshape(cloud_object.Y_grid_points, [cloud_object.Y_grid_points.shape[0]*cloud_object.Y_grid_points.shape[1]])
    Z = np.reshape(cloud_object.Z_grid_points, [cloud_object.Z_grid_points.shape[0]*cloud_object.Z_grid_points.shape[1], 1])

    colormap = np.reshape(cloud_object.metric_grid, [cloud_object.metric_grid.shape[0], 1])
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.Spectral)
    mappable.set_array(colormap)

    # Plot 1: 
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    ax2.grid(False)
    #cloud_object.vertices = cloud_object.aligned_bounding_box_vertices
    #cloud_object.plotCube()
    #ax2.add_collection3d(Poly3DCollection(cloud_object.faces, linewidths=1, edgecolors='b', alpha=.25))

    #cloud_object.vertices = cloud_object.oriented_bounding_box_vertices
    #cloud_object.plotCube()
    #ax2.add_collection3d(Poly3DCollection(cloud_object.faces, linewidths=1, edgecolors='b', alpha=.25))
    # ax2.scatter(point_world[0], point_world[1], point_world[2], marker = '*', s = 100, color = 'r')
    ax2.scatter(x_points, y_points, z_points, s = 0.2)

    # Base reference Frame:
    cloud_object.R = cloud_object.R_base
    cloud_object.p = cloud_object.p_base
    cloud_object.scale_value = 0.15
    cloud_object.length_value = 0.15
    ax2 = cloud_object.plotReferenceFrames(ax2)

    # Camera reference Frame:
    cloud_object.R = cloud_object.R_base_cam
    cloud_object.p = cloud_object.p_base_cam
    cloud_object.scale_value = 0.25
    cloud_object.length_value = 0.15
    ax2 = cloud_object.plotReferenceFrames(ax2)

    # Bounding box reference frame:
    cloud_object.R = cloud_object.R_bounding_box
    cloud_object.p = cloud_object.p_bounding_box
    cloud_object.scale_value = 0.25
    cloud_object.length_value = 0.15
    ax2 = cloud_object.plotReferenceFrames(ax2)

    # Selected end-effector pose: 
    # pose = cloud_object.computed_end_effector_poses_base[num_pose]
    # pose = cloud_object.approach_dir_2_poses_base[num_pose]
    pose = cloud_object.approach_dir_1_poses_base[num_pose]
    # pose = cloud_object.approach_dir_3_poses_base[num_pose]
    # pose = cloud_object.approach_dir_4_poses_base[num_pose]
    # pose = cloud_object.approach_dir_5_poses_base[num_pose]
    cloud_object.R = pose[0:3, 0:3]
    cloud_object.p = pose[0:3, 3]
    cloud_object.scale_value = 0.09
    cloud_object.length_value = 0.09
    ax2 = cloud_object.plotReferenceFrames(ax2)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim(-0.6, 0.6)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_zlim(-0.6, 0.6)
    plt.show(block=True)
    # rospy.spin()
    plt.savefig('plot1.png', bbox_inches='tight', dpi=300)
    plt.savefig('plot1.pdf', bbox_inches='tight', dpi=300)

    # Plot 2:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(projection='3d')
    ax1.grid(False)
    cloud_object.vertices = cloud_object.transformed_vertices_object_frame
    cloud_object.plotCube()
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
    ax1 = cloud_object.plotReferenceFrames(ax1)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-0.4, 0.4)
    ax1.set_ylim(-0.4, 0.4)
    ax1.set_zlim(-0.4, 0.4)
    # plt.show(block=True)
    plt.savefig('plot2.png', bbox_inches='tight', dpi=300)
    plt.savefig('plot2.pdf', bbox_inches='tight', dpi=300)

    # Plot 3: 
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(projection='3d')
    ax3.grid(False)
    ax3.add_collection3d(Poly3DCollection(cloud_object.faces, linewidths=1, edgecolors='b', alpha=.25))
    ax3.scatter(x_sampled_c1, y_sampled_c1, z_sampled_c1, s = 0.2)
    ax3.scatter(x_sampled_c2, y_sampled_c2, z_sampled_c2, s = 0.2)
    # Visualize the screw axis: 
    ax3.scatter(cloud_object.point[0], cloud_object.point[1], cloud_object.point[2], marker = '*', s = 100, color = 'r')
    ax3.quiver(cloud_object.point[0], cloud_object.point[1], cloud_object.point[2], 0.25*cloud_object.screw_axis[0], 0.25*cloud_object.screw_axis[1], 0.25*cloud_object.screw_axis[2], color = "r", arrow_length_ratio = 0.25)

    # Base reference Frame: 
    cloud_object.R = cloud_object.R_base
    cloud_object.p = cloud_object.p_base
    cloud_object.scale_value = 0.45
    cloud_object.length_value = 0.25
    ax3 = cloud_object.plotReferenceFrames(ax3)

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_xlim(-0.4, 0.4)
    ax3.set_ylim(-0.4, 0.4)
    ax3.set_zlim(-0.4, 0.4)
    # plt.show(block=True)
    plt.savefig('plot3.png', bbox_inches='tight', dpi=300)
    plt.savefig('plot3.pdf', bbox_inches='tight', dpi=300)

    # Plot 4:
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(projection='3d')
    ax4.grid(False)
    ax4.add_collection3d(Poly3DCollection(cloud_object.faces, linewidths=1, edgecolors='b', alpha=.25))
    sc4 = ax4.scatter(x_c1, y_c1, z_c1, c=cloud_object.predicted, cmap='viridis')
    ax4.scatter(x_c2, y_c2, z_c2, c=cloud_object.predicted, cmap='viridis')
    # Visualize the screw axis: 
    ax4.scatter(cloud_object.point[0], cloud_object.point[1], cloud_object.point[2], marker = '*', s = 100, color = 'r')
    ax4.quiver(cloud_object.point[0], cloud_object.point[1], cloud_object.point[2], 0.25*cloud_object.screw_axis[0], 0.25*cloud_object.screw_axis[1], 0.25*cloud_object.screw_axis[2], color = "r", arrow_length_ratio = 0.25)
    
    # Base reference Frame: 
    cloud_object.R = cloud_object.R_base
    cloud_object.p = cloud_object.p_base
    cloud_object.scale_value = 0.45
    cloud_object.length_value = 0.25
    ax4 = cloud_object.plotReferenceFrames(ax4)

    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_xlim(-0.4, 0.4)
    ax4.set_ylim(-0.4, 0.4)
    ax4.set_zlim(-0.4, 0.4)
    fig4.colorbar(sc4)
    # plt.show(block=True)
    plt.savefig('plot4.png', bbox_inches='tight', dpi=300)
    plt.savefig('plot4.pdf', bbox_inches='tight', dpi=300)

    # Plot 5: 
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(projection='3d')
    ax5.grid(False)
    ax5.add_collection3d(Poly3DCollection(cloud_object.faces, linewidths=1, edgecolors='b', alpha=.25))
    ax5.scatter(x_projected, y_projected, z_projected, s = 0.2)
    # Visualize the screw axis: 
    ax5.scatter(cloud_object.point[0], cloud_object.point[1], cloud_object.point[2], marker = '*', s = 100, color = 'r')
    ax5.quiver(cloud_object.point[0], cloud_object.point[1], cloud_object.point[2], 0.25*cloud_object.screw_axis[0], 0.25*cloud_object.screw_axis[1], 0.25*cloud_object.screw_axis[2], color = "r", arrow_length_ratio = 0.25)
    
    # Base reference Frame: 
    cloud_object.R = cloud_object.R_base
    cloud_object.p = cloud_object.p_base
    cloud_object.scale_value = 0.45
    cloud_object.length_value = 0.25
    ax5 = cloud_object.plotReferenceFrames(ax5)

    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.set_xlim(-0.4, 0.4)
    ax5.set_ylim(-0.4, 0.4)
    ax5.set_zlim(-0.4, 0.4)
    plt.show(block=True)
    plt.savefig('plot5.png', bbox_inches='tight', dpi=300)
    plt.savefig('plot5.pdf', bbox_inches='tight', dpi=300)

    # Plot 7:
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(projection='3d')
    ax7.add_collection3d(Poly3DCollection(cloud_object.faces, linewidths=1, edgecolors='b', alpha=.25))
    ax7.plot_surface(X, Y, Z, cmap = mappable.cmap)
    ax7.scatter(x_projected, y_projected, z_projected, s = 20, color = 'r')
    # Visualize the screw axis: 
    ax7.scatter(cloud_object.point[0], cloud_object.point[1], cloud_object.point[2], marker = '*', s = 100, color = 'r')
    ax7.quiver(cloud_object.point[0], cloud_object.point[1], cloud_object.point[2], 0.25*cloud_object.screw_axis[0], 0.25*cloud_object.screw_axis[1], 0.25*cloud_object.screw_axis[2], color = "r", arrow_length_ratio = 0.25)
    
    # Base reference Frame: 
    cloud_object.R = cloud_object.R_base
    cloud_object.p = cloud_object.p_base
    cloud_object.scale_value = 0.45
    cloud_object.length_value = 0.25
    ax7 = cloud_object.plotReferenceFrames(ax7)

    # ax7.plot_surface(X_grid_points, Y_grid_points, Z_grid_points, cmap = mappable.cmap, norm=mappable.norm, linewidth=0, antialiased=False)
    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    ax7.set_zlabel('Z')
    ax7.set_xlim(-0.4, 0.4)
    ax7.set_ylim(-0.4, 0.4)
    ax7.set_zlim(-0.4, 0.4)
    # plt.colorbar(mappable)
    # plt.show(block=True)
    plt.savefig('plot6.png', bbox_inches='tight', dpi=300)
    plt.savefig('plot6.pdf', bbox_inches='tight', dpi=300)

     # Plot 12:
    # Visualizing the sampled points and their associated values computed using the metric: 
    fig12 = plt.figure()
    ax12 = fig12.add_subplot(projection='3d')
    cloud_object.vertices = cloud_object.transformed_vertices_object_frame
    cloud_object.plotCube()
    ax12.add_collection3d(Poly3DCollection(cloud_object.faces, linewidths=1, edgecolors='r', alpha=.25))
    ax12.scatter(x_transformed_points, y_transformed_points, z_transformed_points, s = 0.2)
    ax12.scatter(cloud_object.point[0], cloud_object.point[1], cloud_object.point[2], marker = '*', s = 100, color = 'r')
    ax12.quiver(cloud_object.point[0], cloud_object.point[1], cloud_object.point[2], 0.25*cloud_object.screw_axis[0], 0.25*cloud_object.screw_axis[1], 0.25*cloud_object.screw_axis[2], color = "r", arrow_length_ratio = 0.25)
    # sc11 = ax11.scatter(x_highest_metric_values_points, y_highest_metric_values_points, z_highest_metric_values_points)
    
    # Base reference Frame: 
    cloud_object.R = cloud_object.R_base
    cloud_object.p = cloud_object.p_base
    cloud_object.scale_value = 0.45
    cloud_object.length_value = 0.25
    ax12 = cloud_object.plotReferenceFrames(ax12)

    '''# Plotting the sampled end effector reference frames:
    for i in range(len(cloud_object.computed_end_effector_poses)):
        pose = cloud_object.computed_end_effector_poses[i]
        cloud_object.R = pose[0:3, 0:3]
        cloud_object.p = pose[0:3, 3]
        cloud_object.scale_value = 0.05
        cloud_object.length_value = 0.05
        ax12 = cloud_object.plotReferenceFrames(ax12)

    # Plotting the contact reference frames C1:
    for i in range(len(cloud_object.sampled_contacts_c1)):
        pose = cloud_object.sampled_contacts_c1[i]
        cloud_object.R = pose[0:3, 0:3]
        cloud_object.p = pose[0:3, 3]
        cloud_object.scale_value = 0.025
        cloud_object.length_value = 0.025
        ax12 = cloud_object.plotReferenceFrames(ax12)

    # Plotting the contact reference frames C2:
    for i in range(len(cloud_object.sampled_contacts_c2)):
        pose = cloud_object.sampled_contacts_c2[i]
        cloud_object.R = pose[0:3, 0:3]
        cloud_object.p = pose[0:3, 3]
        cloud_object.scale_value = 0.025
        cloud_object.length_value = 0.025
        ax12 = cloud_object.plotReferenceFrames(ax12)'''

    # Grasp centers:
    for i in range(len(cloud_object.grasp_centers)):
        g = cloud_object.grasp_centers[i]
        ax12.scatter(g[0], g[1], g[2], marker = '*', s = 20, color = 'r')


    ax12.set_xlabel('X')
    ax12.set_ylabel('Y')
    ax12.set_zlabel('Z')
    ax12.set_xlim(-0.4, 0.4)
    ax12.set_ylim(-0.4, 0.4)
    ax12.set_zlim(-0.4, 0.4)

    fig12.show()
    print(fig12)
    plt.savefig('plot12.png', bbox_inches='tight', dpi=300)
    plt.savefig('plot12.pdf', bbox_inches='tight', dpi=300)


    '''COMMUNICATING WITH THE SCLERP-BASED MOTION PLANNER:'''

    # Converting the poses to a quaternion format to be sent via action server:
    cloud_object.computed_end_effector_quaternion_base = []
    cloud_object.computed_end_effector_quaternion_inter_base = []

    # Converting the grasp pose into quaternion:
    # for pose in cloud_object.computed_end_effector_poses_base:
    # for pose in cloud_object.approach_dir_2_poses_base:
    # for pose in cloud_object.approach_dir_1_poses_base:
    # for pose in cloud_object.approach_dir_3_poses_base:
    # for pose in cloud_object.approach_dir_4_poses_base:
    for pose in cloud_object.approach_dir_5_poses_base:
        rot_quat = Rot.from_matrix(pose[0:3, 0:3])
        rot_quat = rot_quat.as_quat()

        grasp_pose = Pose()
        grasp_pose.position.x = pose[0,3]
        grasp_pose.position.y = pose[1,3]
        grasp_pose.position.z = pose[2,3]

        grasp_pose.orientation.x = rot_quat[0]
        grasp_pose.orientation.y = rot_quat[1]
        grasp_pose.orientation.z = rot_quat[2]
        grasp_pose.orientation.w = rot_quat[3]

        cloud_object.computed_end_effector_quaternion_base.append(grasp_pose)

    # Converting the pre-grasp pose into quaternion:
    # for inter_pose in cloud_object.computed_end_effector_poses_inter_base:
    # for inter_pose in cloud_object.approach_dir_2_inter_poses_base:
    # for inter_pose in cloud_object.approach_dir_1_inter_poses_base:
    # for inter_pose in cloud_object.approach_dir_3_inter_poses_base:
    # for inter_pose in cloud_object.approach_dir_4_inter_poses_base:
    for inter_pose in cloud_object.approach_dir_5_inter_poses_base:
        rot_quat_inter = Rot.from_matrix(inter_pose[0:3, 0:3])
        rot_quat_inter = rot_quat_inter.as_quat()

        grasp_pose_inter = Pose()
        grasp_pose_inter.position.x = inter_pose[0,3]
        grasp_pose_inter.position.y = inter_pose[1,3]
        grasp_pose_inter.position.z = inter_pose[2,3]

        grasp_pose_inter.orientation.x = rot_quat_inter[0]
        grasp_pose_inter.orientation.y = rot_quat_inter[1]
        grasp_pose_inter.orientation.z = rot_quat_inter[2]
        grasp_pose_inter.orientation.w = rot_quat_inter[3]

        cloud_object.computed_end_effector_quaternion_inter_base.append(grasp_pose_inter)

    # Saving the necessary data to process and computing the end-effector pose in the base frame and the object frame after the pivoting motion:
    num_poses = len(cloud_object.computed_end_effector_poses)

    grasp_poses_object_frame = np.reshape(np.asarray([cloud_object.computed_end_effector_poses]), [num_poses*4, 4])
    pre_grasp_poses_base_frame = np.reshape(np.asarray([cloud_object.computed_end_effector_poses_inter_base]), [num_poses*4, 4])
    grasp_poses_base_frame = np.reshape(np.asarray([cloud_object.computed_end_effector_poses_base]), [num_poses*4, 4])

    print(grasp_poses_object_frame.shape)

    grasp_pose_object_frame = cloud_object.approach_dir_1_poses[num_pose]
    grasp_pose_base_frame = cloud_object.approach_dir_1_poses_base[num_pose]

    print("Number of grasp pose along approach direction 2: ", len(cloud_object.approach_dir_1_poses_base))

    # Saving just the selected grasp poses:
    np.savetxt('pringles_pouring_pose7_EE8_grasp_pose_object_frame_alg3.csv', grasp_pose_object_frame, delimiter=",")
    np.savetxt('pringles_pouring_pose7_EE8_grasp_pose_base_frame_alg3.csv', grasp_pose_base_frame, delimiter=",")

    # Saving all the grasp poses:
    np.savetxt('pringles_pouring_pose7_EE8_grasp_poses_object_frame_alg1.csv', grasp_poses_object_frame, delimiter=",")
    np.savetxt('pringles_pouring_pose7_EE8_grasp_poses_base_frame_alg1.csv', grasp_poses_base_frame, delimiter=",")

    # Saving the screw axis and point as expressed in the object reference frame:
    np.savetxt('pringles_pouring_pose7_EE8_screw_axis_object_frame_alg1.csv', cloud_object.screw_axis, delimiter = ',')
    np.savetxt('pringles_pouring_pose7_EE8_point_object_frame_alg1.csv', cloud_object.point, delimiter = ',')

    # Saving the pose of the object with respect to the base frame:
    np.savetxt('pringles_pouring_pose7_EE8_R_object_alg1.csv', cloud_object.R_bounding_box, delimiter = ',')
    np.savetxt('pringles_pouring_pose7_EE8_p_object_alg1.csv', cloud_object.p_bounding_box, delimiter = ',')


    
