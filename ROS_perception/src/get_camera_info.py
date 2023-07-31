#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import sys
import numpy as np
import pyrealsense2 as rs2

#Initialize a bridge object of class CvBridge
bridge = CvBridge()

def cameraInfoCallback(cameraInfo):
    _intrinsics = rs2.intrinsics()
    _intrinsics.width = cameraInfo.width
    _intrinsics.height = cameraInfo.height
    _intrinsics.ppx = cameraInfo.K[2]
    _intrinsics.ppy = cameraInfo.K[5]
    _intrinsics.fx = cameraInfo.K[0]
    _intrinsics.fy = cameraInfo.K[4]
    if cameraInfo.distortion_model == 'plumb_bob':
        _intrinsics.model = rs2.distortion.brown_conrady
    elif cameraInfo.distortion_model == 'equidistant':
        _intrinsics.model = rs2.distortion.kannala_brandt4
    _intrinsics.coeffs = [i for i in cameraInfo.D]

    print("Params Loaded!")

def getCamInfoNode():
    rospy.init_node('camInfoNode', anonymous=True)
    rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, cameraInfoCallback)
    rospy.spin()

if __name__ == '__main__':
    getCamInfoNode()