#!/usr/bin/env python
# Python script to extract points corresponding to an object from a point cloud based on the bounding box.
# For objects that can be selected using the interface with Segment Anything from Meta for semantic segmentation
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

# Python API for Intel Realsense D415
import pyrealsense2 as rs2

# Libraries required to perform inference using DeepLab for semantic segmentation
from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image as PILImage 
from torchvision import transforms
from dataloaders.utils import  *

# Open3D for point cloud processing and visualization
import open3d as o3d

# Segment Anything by Meta for Semantic Segmentation:
from segment_anything import sam_model_registry, SamPredictor

import torch
from torchvision import transforms
from dataloaders.utils import  *
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

from get_point_cloud_segment_anything import realsenseSubscriber
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

    cloud_object = pointCloud()

    # Pose of the camera reference frame with respect to the base reference frame: 
    cloud_object.g_base_cam = realsenseObject.getEndEffectorPose()
    print('Printing the Pose of the camera with respect to the base reference frame!')
    print(cloud_object.g_base_cam)
    
    # Implementing semantic segmentation using Segment Anything by Meta:
    dataPointsSegmented = realsenseObject.getCoordsSAM()

    # Implementing semantic segmentation using DeepLab:
    # dataPointsSegmented = realsenseObject.getCoordSemanticSeg()

    print(dataPointsSegmented.shape)

    # Creating a Open3d PointCloud Object for the cloud corresponding to just the bounding box
    objectCloud = o3d.geometry.PointCloud()
    objectCloud.points = o3d.utility.Vector3dVector(dataPointsSegmented.astype(np.float64))
    objectCloud.paint_uniform_color([0, 0, 1])

    # Visualizing just the CheezIt point cloud using open3D:
    o3d.visualization.draw_geometries([objectCloud])