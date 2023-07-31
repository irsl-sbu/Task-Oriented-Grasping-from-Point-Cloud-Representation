#!/usr/bin/env python
# Python script to extract points corresponding to an object from a point cloud based on the bounding box.
# For objects that can be detected using the YoloV5 Object Detector.
# By: Aditya Patankar

import rospy
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError

# Numpy packages 
import numpy as np
from numpy import linalg as la
from numpy import genfromtxt

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

from get_point_cloud import realsenseSubscriber
from process_point_cloud import pointCloud
#from preprocess_point_cloud import pointCloud

if __name__ == '__main__':
    rospy.init_node('listenerNode', anonymous=True)
    imageTopic = "/camera/color/image_rect_color"
    depthTopic = "/camera/aligned_depth_to_color/image_raw"
    cameraInfoTopic = "/camera/aligned_depth_to_color/camera_info"
<<<<<<< HEAD
    path = "/home/Aditya/Saved_Images/"+"cheezit"
=======
    path = "/home/aditya/catkin_ws/src/perception/src/Saved_Images/"+"cheezit"
>>>>>>> b6641299b1b791cecfab931ada1b5c90f71e07a9

    print('Realsense Subscriber Object initialized')
    print('---------------------------------------')

    realsenseObject = realsenseSubscriber(imageTopic, depthTopic, cameraInfoTopic, path)
    rospy.spin()

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
    
    #np.savetxt("dataPointsBB_cheezit_trial16.csv", dataPointsBB, delimiter= ",")
<<<<<<< HEAD
    # np.savetxt("dataPointsBB_tomatosoup_trial5.csv", dataPointsBB, delimiter= ",")
    np.savetxt("dataPointsBB_dominosugar_trial9.csv", dataPointsBB, delimiter= ",")
=======
    #np.savetxt("dataPointsBB_tomatosoup_trial1.csv", dataPointsBB, delimiter= ",")
    np.savetxt("dataPointsBB_dominosugar_trial2.csv", dataPointsBB, delimiter= ",")
>>>>>>> b6641299b1b791cecfab931ada1b5c90f71e07a9
    #np.savetxt("dataPointsBB_screwdriver_trial5.csv", dataPointsBB, delimiter= ",")

    # Saving the point cloud in the necessary format:
    #o3d.io.write_point_cloud("cheezit_trial16_BB.ply", objectCloud)
<<<<<<< HEAD
    # o3d.io.write_point_cloud("tomatosoup_trial5_BB.ply", objectCloud)
    o3d.io.write_point_cloud("dominosugar_trial9_BB.ply", objectCloud)
=======
    #o3d.io.write_point_cloud("tomatosoup_trial1_BB.ply", objectCloud)
    o3d.io.write_point_cloud("dominosugar_trial2_BB.ply", objectCloud)
>>>>>>> b6641299b1b791cecfab931ada1b5c90f71e07a9
    # o3d.io.write_point_cloud("screwdriver_trial5_BB.ply", cheezitCloud)

    print('Bounding Box data saved!')

    # Getting the complete coordinates
    dataPoints = realsenseObject.getCoordComplete()

    # Creating a Open3d PointCloud Object for the entire cloud
    objectCloudComplete = o3d.geometry.PointCloud()
    objectCloudComplete.points = o3d.utility.Vector3dVector(dataPoints.astype(np.float64))
    objectCloudComplete.paint_uniform_color([0, 1, 0])

    # Saving the point cloud in the necessary format:
    #o3d.io.write_point_cloud("cheezit_trial16_Complete.ply", objectCloudComplete)
<<<<<<< HEAD
    # o3d.io.write_point_cloud("tomatosoup_trial5_Complete.ply", objectCloudComplete)
    o3d.io.write_point_cloud("dominosugar_trial9_Complete.ply", objectCloudComplete)
    #o3d.io.write_point_cloud("screwdriver_trial5_Complete.ply", cheezitCloudComplete)

    #np.savetxt("dataPoints_cheezit_trial16_complete.csv", dataPoints, delimiter= ",")
    # np.savetxt("dataPoints_tomatosoup_trial5_complete.csv", dataPoints, delimiter= ",")
    np.savetxt("dataPoints_dominosugar_trial9_complete.csv", dataPoints, delimiter= ",")
=======
    #o3d.io.write_point_cloud("tomatosoup_trial1_Complete.ply", objectCloudComplete)
    o3d.io.write_point_cloud("dominosugar_trial2_Complete.ply", objectCloudComplete)
    #o3d.io.write_point_cloud("screwdriver_trial5_Complete.ply", cheezitCloudComplete)

    #np.savetxt("dataPoints_cheezit_trial16_complete.csv", dataPoints, delimiter= ",")
    #np.savetxt("dataPoints_tomatosoup_trial1_complete.csv", dataPoints, delimiter= ",")
    np.savetxt("dataPoints_dominosugar_trial2_complete.csv", dataPoints, delimiter= ",")
>>>>>>> b6641299b1b791cecfab931ada1b5c90f71e07a9
    # np.savetxt("dataPoints_screwdriver_trial5_complete.csv", dataPoints, delimiter= ",")    
    print('Complete data saved!')

    cloud_object = pointCloud()
    cloud_object.cloud = objectCloud

    # Pose of the camera reference frame with respect to the base reference frame: 
<<<<<<< HEAD
    '''cloud_object.g_base_cam = np.asarray([[-0.047594,  -0.794189,   0.605783,   0.365809],
                                         [-0.998696,  0.0484359, -0.0149638,  0.0362036],
                                         [-0.0174575,  -0.605708,  -0.795492,   0.608372],
                                         [0,          0,          0,          1]])'''

    cloud_object.g_base_cam = np.asarray([[0.0294749,   -0.872062,    0.488477,    0.406076],
                                          [-0.999538,  -0.0235534,    0.018264,   0.0456198],
                                          [-0.00442195,   -0.488793,   -0.872388,    0.526944],
                                          [0,           0,           0,           1]])

=======
    cloud_object.g_base_cam = np.asarray([[0.0562177,  -0.855105,    0.51537,   0.419833],
                                          [-0.998345, -0.0427043,  0.0380476,  0.0569145],
                                          [-0.0105261,  -0.516659,  -0.856125,   0.554612],
                                          [0,          0,          0,          1]])
     
>>>>>>> b6641299b1b791cecfab931ada1b5c90f71e07a9
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

    # Specifying parameters for DBSCAN Clustering:
    # Just like the parameters for downsampling even the parameters for DBSCAN Clustering are dependent on the 
    # units used computing and extracting the point cloud data.
    cloud_object.eps = 0.05
    cloud_object.min_points = 10
    cloud_object.getObjectPointCloud()

    # Visualizing the downsampled point cloud. 
    print('Visualizing the processed cloud!')
    o3d.visualization.draw_geometries([cloud_object.processed_cloud])

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
    #np.savetxt('oriented_box_center_cheezit_trial16.csv', cloud_object.oriented_bounding_box_center, delimiter = ",")
    #np.savetxt('oriented_box_vertices_cheezit_trial16.csv', cloud_object.oriented_bounding_box_vertices, delimiter = ",")

<<<<<<< HEAD
    # np.savetxt('oriented_box_center_tomatosoup_trial5.csv', cloud_object.oriented_bounding_box_center, delimiter = ",")
    # np.savetxt('oriented_box_vertices_tomatosoup_trial5.csv', cloud_object.oriented_bounding_box_vertices, delimiter = ",")

    np.savetxt('oriented_box_center_dominosugar_trial9.csv', cloud_object.oriented_bounding_box_center, delimiter = ",")
    np.savetxt('oriented_box_vertices_dominosugar_trial9.csv', cloud_object.oriented_bounding_box_vertices, delimiter = ",")
=======
    #np.savetxt('oriented_box_center_tomatosoup_trial1.csv', cloud_object.oriented_bounding_box_center, delimiter = ",")
    #np.savetxt('oriented_box_vertices_tomatosoup_trial1.csv', cloud_object.oriented_bounding_box_vertices, delimiter = ",")

    np.savetxt('oriented_box_center_dominosugar_trial2.csv', cloud_object.oriented_bounding_box_center, delimiter = ",")
    np.savetxt('oriented_box_vertices_dominosugar_trial2.csv', cloud_object.oriented_bounding_box_vertices, delimiter = ",")
>>>>>>> b6641299b1b791cecfab931ada1b5c90f71e07a9

    #np.savetxt('oriented_box_center_screwdriver_trial5.csv', cloud_object.oriented_bounding_box_center, delimiter = ",")
    #np.savetxt('oriented_box_vertices_screwdriver_trial5.csv', cloud_object.oriented_bounding_box_vertices, delimiter = ",")

    #np.savetxt('aligned_box_center_cheezit_trial16.csv', cloud_object.aligned_bounding_box_center, delimiter = ",")
    #np.savetxt('aligned_box_vertices_cheezit_trial16.csv', cloud_object.aligned_bounding_box_vertices, delimiter = ",")

<<<<<<< HEAD
    #np.savetxt('aligned_box_center_tomatosoup_trial5.csv', cloud_object.aligned_bounding_box_center, delimiter = ",")
    #np.savetxt('aligned_box_vertices_tomatosoup_trial5.csv', cloud_object.aligned_bounding_box_vertices, delimiter = ",")

    np.savetxt('aligned_box_center_dominosugar_trial9.csv', cloud_object.aligned_bounding_box_center, delimiter = ",")
    np.savetxt('aligned_box_vertices_dominosugar_trial9.csv', cloud_object.aligned_bounding_box_vertices, delimiter = ",")
=======
    #np.savetxt('aligned_box_center_tomatosoup_trial1.csv', cloud_object.aligned_bounding_box_center, delimiter = ",")
    #np.savetxt('aligned_box_vertices_tomatosoup_trial1.csv', cloud_object.aligned_bounding_box_vertices, delimiter = ",")

    np.savetxt('aligned_box_center_dominosugar_trial2.csv', cloud_object.aligned_bounding_box_center, delimiter = ",")
    np.savetxt('aligned_box_vertices_dominosugar_trial2.csv', cloud_object.aligned_bounding_box_vertices, delimiter = ",")
>>>>>>> b6641299b1b791cecfab931ada1b5c90f71e07a9

    #np.savetxt('aligned_box_center_screwdriver_trial5.csv', cloud_object.aligned_bounding_box_center, delimiter = ",")
    #np.savetxt('aligned_box_vertices_screwdriver_trial5.csv', cloud_object.aligned_bounding_box_vertices, delimiter = ",")

    # Saving the processed point cloud: 
    #o3d.io.write_point_cloud("cheezit_trial16_Processed_Transformed.ply", cloud_object.processed_cloud)
<<<<<<< HEAD
    #o3d.io.write_point_cloud("tomatosoup_trial5_Processed_Transformed.ply", cloud_object.processed_cloud)
    o3d.io.write_point_cloud("dominosugar_trial9_Processed_Transformed.ply", cloud_object.processed_cloud)
=======
    #o3d.io.write_point_cloud("tomatosoup_trial1_Processed_Transformed.ply", cloud_object.processed_cloud)
    o3d.io.write_point_cloud("dominosugar_trial2_Processed_Transformed.ply", cloud_object.processed_cloud)
>>>>>>> b6641299b1b791cecfab931ada1b5c90f71e07a9
    #o3d.io.write_point_cloud("screwdriver_trial5_Processed_Transformed.ply", cloud_object.processed_cloud)

    