#!/usr/bin/env python
# Python script to extract points corresponding to an object from a point cloud based on the bounding box.
# For objects that can be detected using the YoloV5 Object Detector.
# This script also includes the subscriber to subscribe to the topic publishing the end-effector pose.
# By: Aditya Patankar

import rospy
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# Numpy packages 
import numpy as np
from numpy import linalg as la
from scipy.spatial.transform import Rotation as Rot

# Matplotlib libraries for plotting and visualization in Python:
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

# Pytorch Packages for Neural Network based prediction: 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from neuralNet import metricNN
from dataLoader import metricNNDataset
from dataLoader import ToTensor

import pyrealsense2 as rs2

# Open3D for point cloud processing and visualization
import open3d as o3d

from get_point_cloud_new import realsenseSubscriber
from process_point_cloud import pointCloud
#from preprocess_point_cloud import pointCloud

if __name__ == '__main__':
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

    ##Getting the coordinates corresponding to the bounding box
    realsenseObject.getBoundingBox()
    realsenseObject.getCoordBoundingBox()
    dataPointsBB = realsenseObject.getCoordBoundingBox()

    # Creating a Open3d PointCloud Object for the cloud corresponding to just the bounding box
    objectCloud = o3d.geometry.PointCloud()
    objectCloud.points = o3d.utility.Vector3dVector(dataPointsBB.astype(np.float64))
    objectCloud.paint_uniform_color([0, 0, 1])

    # Visualizing just the CheezIt point cloud using open3D:
    o3d.visualization.draw_geometries([objectCloud])
    
    np.savetxt("dataPointsBB_cheezit_trial10.csv", dataPointsBB, delimiter= ",")
    # np.savetxt("dataPointsBB_tomatosoup_trial5.csv", dataPointsBB, delimiter= ",")
    # np.savetxt("dataPointsBB_dominosugar_trial9.csv", dataPointsBB, delimiter= ",")
    # np.savetxt("dataPointsBB_screwdriver_trial5.csv", dataPointsBB, delimiter= ",")

    # Saving the point cloud in the necessary format:
    o3d.io.write_point_cloud("cheezit_trial10_BB.ply", objectCloud)
    # o3d.io.write_point_cloud("tomatosoup_trial5_BB.ply", objectCloud)
    # o3d.io.write_point_cloud("dominosugar_trial9_BB.ply", objectCloud)
    # o3d.io.write_point_cloud("screwdriver_trial5_BB.ply", cheezitCloud)

    print('Bounding Box data saved!')

    # Getting the complete coordinates
    dataPoints = realsenseObject.getCoordComplete()

    # Creating a Open3d PointCloud Object for the entire cloud
    objectCloudComplete = o3d.geometry.PointCloud()
    objectCloudComplete.points = o3d.utility.Vector3dVector(dataPoints.astype(np.float64))
    objectCloudComplete.paint_uniform_color([0, 1, 0])

    # Saving the point cloud in the necessary format:
    o3d.io.write_point_cloud("cheezit_trial10_Complete.ply", objectCloudComplete)
    # o3d.io.write_point_cloud("tomatosoup_trial5_Complete.ply", objectCloudComplete)
    # o3d.io.write_point_cloud("dominosugar_trial9_Complete.ply", objectCloudComplete)
    #o3d.io.write_point_cloud("screwdriver_trial5_Complete.ply", cheezitCloudComplete)

    np.savetxt("dataPoints_cheezit_trial10_complete.csv", dataPoints, delimiter= ",")
    # np.savetxt("dataPoints_tomatosoup_trial5_complete.csv", dataPoints, delimiter= ",")
    # np.savetxt("dataPoints_dominosugar_trial9_complete.csv", dataPoints, delimiter= ",")
    # np.savetxt("dataPoints_screwdriver_trial5_complete.csv", dataPoints, delimiter= ",")    
    print('Complete data saved!')

    cloud_object = pointCloud()
    cloud_object.cloud = objectCloud

    # Pose of the camera reference frame with respect to the base reference frame: 
    cloud_object.g_base_cam = realsenseObject.getEndEffectorPose()
    print('Printing the Pose of the camera with respect to the base reference frame!')
    print(cloud_object.g_base_cam)

    # Saving the pose of the camera with respect to the base reference frame:
    np.savetxt("EndEffector.csv", cloud_object.g_base_cam, delimiter=',')

    # Transforming the point cloud in the Panda base reference frame: 
    cloud_object.transformToBase()

    # Visualizing the downsampled point cloud. 
    print('Cloud transformed to base')
    o3d.visualization.draw_geometries([cloud_object.cloud])

    # Downsample it and inspect the normals
    cloud_object.cloud = cloud_object.cloud.voxel_down_sample(voxel_size=0.009)
    cloud_object.removePlaneSurface()

    # Visualizing the downsampled point cloud. 
    print('Plane surface removed!')
    o3d.visualization.draw_geometries([cloud_object.cloud])

    o3d.visualization.draw_geometries([cloud_object.cloud], point_show_normal=True)

    # Specifying parameters for DBSCAN Clustering:
    # Just like the parameters for downsampling even the parameters for DBSCAN Clustering are dependent on the 
    # units used computing and extracting the point cloud data.
    cloud_object.eps = 0.02
    cloud_object.min_points = 10
    cloud_object.getObjectPointCloud()

    # Visualizing the downsampled point cloud. 
    print('Visualizing the processed cloud!')
    o3d.visualization.draw_geometries([cloud_object.processed_cloud])

    print('Visualizing the point cloud with the normals')
    o3d.visualization.draw_geometries([cloud_object.processed_cloud], point_show_normal=True)

    # The object is not aligned with the axis of the world/robot base reference frame.
    cloud_object.bounding_box_flag = 0

    # Rotation matrix and position vector for the robot base or world reference frame: 
    cloud_object.R_base = np.identity(3)
    cloud_object.p_base = np.zeros([3,1])

    # Computing the bounding boxes corresponding to the object point cloud: 
    cloud_object.computeBoundingBox()

    # Visualizing the bounding boxes along with the object point cloud: 
    o3d.visualization.draw_geometries([cloud_object.processed_cloud, cloud_object.aligned_bounding_box, cloud_object.oriented_bounding_box])

    # Saving the vertices and the centers of the bounding boxes: 
    np.savetxt('oriented_box_center_cheezit_trial10.csv', cloud_object.oriented_bounding_box_center, delimiter = ",")
    np.savetxt('oriented_box_vertices_cheezit_trial10.csv', cloud_object.oriented_bounding_box_vertices, delimiter = ",")

    # np.savetxt('oriented_box_center_tomatosoup_trial5.csv', cloud_object.oriented_bounding_box_center, delimiter = ",")
    # np.savetxt('oriented_box_vertices_tomatosoup_trial5.csv', cloud_object.oriented_bounding_box_vertices, delimiter = ",")

    # np.savetxt('oriented_box_center_dominosugar_trial9.csv', cloud_object.oriented_bounding_box_center, delimiter = ",")
    # np.savetxt('oriented_box_vertices_dominosugar_trial9.csv', cloud_object.oriented_bounding_box_vertices, delimiter = ",")

    #np.savetxt('oriented_box_center_screwdriver_trial5.csv', cloud_object.oriented_bounding_box_center, delimiter = ",")
    #np.savetxt('oriented_box_vertices_screwdriver_trial5.csv', cloud_object.oriented_bounding_box_vertices, delimiter = ",")

    np.savetxt('aligned_box_center_cheezit_trial10.csv', cloud_object.aligned_bounding_box_center, delimiter = ",")
    np.savetxt('aligned_box_vertices_cheezit_trial10.csv', cloud_object.aligned_bounding_box_vertices, delimiter = ",")

    #np.savetxt('aligned_box_center_tomatosoup_trial5.csv', cloud_object.aligned_bounding_box_center, delimiter = ",")
    #np.savetxt('aligned_box_vertices_tomatosoup_trial5.csv', cloud_object.aligned_bounding_box_vertices, delimiter = ",")

    #np.savetxt('aligned_box_center_dominosugar_trial9.csv', cloud_object.aligned_bounding_box_center, delimiter = ",")
    #np.savetxt('aligned_box_vertices_dominosugar_trial9.csv', cloud_object.aligned_bounding_box_vertices, delimiter = ",")

    #np.savetxt('aligned_box_center_screwdriver_trial5.csv', cloud_object.aligned_bounding_box_center, delimiter = ",")
    #np.savetxt('aligned_box_vertices_screwdriver_trial5.csv', cloud_object.aligned_bounding_box_vertices, delimiter = ",")

    # Saving the processed point cloud: 
    o3d.io.write_point_cloud("cheezit_trial10_Processed_Transformed.ply", cloud_object.processed_cloud)
    #o3d.io.write_point_cloud("tomatosoup_trial5_Processed_Transformed.ply", cloud_object.processed_cloud)
    #o3d.io.write_point_cloud("dominosugar_trial9_Processed_Transformed.ply", cloud_object.processed_cloud)
    #o3d.io.write_point_cloud("screwdriver_trial5_Processed_Transformed.ply", cloud_object.processed_cloud)


    ## PLOTTING AND VISUALIZING THE POINT CLOUD:
    cloud_object.points = np.asarray(cloud_object.processed_cloud.points)

    cloud_object.R_base_cam = cloud_object.g_base_cam[0:3, 0:3]
    cloud_object.p_base_cam = np.reshape(cloud_object.g_base_cam[0:3, 3], [3,1])

    x_points = np.reshape(cloud_object.points[:, 0], [cloud_object.points.shape[0],1])
    y_points = np.reshape(cloud_object.points[:, 1], [cloud_object.points.shape[0],1])
    z_points = np.reshape(cloud_object.points[:, 2], [cloud_object.points.shape[0],1])

    # Plot 1: 
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')
    # cloud_object.vertices = cloud_object.aligned_bounding_box_vertices
    # cloud_object.plotCube()
    # ax2.add_collection3d(Poly3DCollection(cloud_object.faces, linewidths=1, edgecolors='b', alpha=.25))

    # cloud_object.vertices = cloud_object.oriented_bounding_box_vertices
    # cloud_object.plotCube()
    # ax2.add_collection3d(Poly3DCollection(cloud_object.faces, linewidths=1, edgecolors='r', alpha=.25))
    ax2.scatter(x_points, y_points, z_points, s = 0.2)
    
    # Base reference Frame: 
    cloud_object.R = cloud_object.R_base
    cloud_object.p = cloud_object.p_base
    cloud_object.scale_value = 0.45
    cloud_object.length_value = 0.25
    ax2 = cloud_object.plotReferenceFrames(ax2)

    # Camera reference Frame:
    cloud_object.R = cloud_object.R_base_cam
    cloud_object.p = cloud_object.p_base_cam
    cloud_object.scale_value = 0.25
    cloud_object.length_value = 0.15
    ax2 = cloud_object.plotReferenceFrames(ax2)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    ax2.set_xlim(-0.6, 0.6)
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_zlim(-0.6, 0.6)
    
    plt.show()

    