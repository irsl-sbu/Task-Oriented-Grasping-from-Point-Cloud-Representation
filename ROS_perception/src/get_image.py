#!/usr/bin/env python
## Subscriber node for subscribing to the Image topic published by the Intel Realsense D415 Camera.
# By: Aditya Patankar


import rospy
import numpy as np
import argparse
import time
import cv2
import os
import sys
import torch
from PIL import Image
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


#Initialize a bridge object of class CvBridge
bridge = CvBridge()

#Callback funtion for the subscriber node
def image_callback(ros_image):
    global bridge
    try:
        cv2_image = bridge.imgmsg_to_cv2(ros_image, "rgb8")
        cv2_image_saved = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as error:
        print(error)
<<<<<<< HEAD
    # cv2.imwrite("/home/marktwo/realsense_ws/src/perception/src/Saved_Images/"+"table"+".png",cv2_image_saved)
    # cv2.imwrite("/home/marktwo/realsense_ws/src/perception/src/Saved_Images/"+"spatula_trial9"+".png",cv2_image_saved)
    # cv2.imwrite("/home/marktwo/realsense_ws/src/perception/src/Saved_Images/"+"cheezit_bb_trial38"+".png",cv2_image_saved)
    # cv2.imwrite("/home/marktwo/realsense_ws/src/perception/src/Saved_Images/"+"spamtuna_trial9"+".png",cv2_image_saved)
    cv2.imwrite("/home/marktwo/realsense_ws/src/perception/src/Saved_Images/"+"screwdriver_10"+".png",cv2_image_saved)
    # cv2.imwrite("/home/marktwo/realsense_ws/src/perception/src/Saved_Images/"+"dominosugar_trial1"+".png",cv2_image_saved)
    # cv2.imwrite("/home/marktwo/realsense_ws/src/perception/src/Saved_Images/"+"pringles_trial9"+".png",cv2_image_saved)
    # cv2.imwrite("/home/marktwo/realsense_ws/src/perception/src/Saved_Images/"+"tomato_soup_trial9"+".png",cv2_image_saved)

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/marktwo/Aditya/object-detection-main/best.pt')
    output = model(cv2_image)
    output.print()
    output.show()
    output_list_1 = output.pandas().xyxy[0].iloc[0].values.tolist()
    output_list_2 = output.pandas().xyxy[0].iloc[1].values.tolist()
    bounding_box_coords_1 = np.around(np.asarray(output_list_1[0:4]))
    bounding_box_coords_2 = np.around(np.asarray(output_list_2[0:4]))
    print('Output List 1: ', output_list_1)
    print('Bounding Box 1 Coordinates: ', bounding_box_coords_1)
    print('Output List 2: ', output_list_2)
    print('Bounding Box 2 Coordinates: ', bounding_box_coords_2)
    print('x_min: ', bounding_box_coords_1[0])
    print('y_min: ', bounding_box_coords_1[1])
    print('x_max: ', bounding_box_coords_1[2])
    print('y_max: ', bounding_box_coords_1[3])
    print('class of x_min: ', type(bounding_box_coords_1[0]))
=======
    cv2.imwrite("/home/aditya/catkin_ws/src/perception/src/Saved_Images/"+"object_scene_cheezit"+".png",cv2_image_saved)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/aditya/Robotics_Research/2022/object-detection-main/best.pt')
    output = model(cv2_image)
    output.print()
    output.show()
    print(type(output))
    
    output_list = output.pandas().xyxy[0].iloc[0].values.tolist()
    bounding_box_coords = np.around(np.asarray(output_list[0:4]))
    print('Output List: ', output_list)
    print('Bounding Box Coordinates: ', bounding_box_coords)
    print('x_min: ', bounding_box_coords[0])
    print('y_min: ', bounding_box_coords[1])
    print('x_max: ', bounding_box_coords[2])
    print('y_max: ', bounding_box_coords[3])
    print('class of x_min: ', type(bounding_box_coords[0]))
>>>>>>> b6641299b1b791cecfab931ada1b5c90f71e07a9


if __name__ == '__main__':
    rospy.init_node('image_converter', anonymous=True)
    
    # Subscriber node using rospy.wait_for_message
    '''image_sub = rospy.wait_for_message("/camera/color/image_rect_color", Image)
    image_callback(Image) '''  

    # Without using rospy.wait_for_message and using the standard rospy.Subscriber()
    image_sub = rospy.Subscriber("/camera/color/image_rect_color", Image, image_callback, queue_size=1)
    rospy.spin()
    image_sub.unregister()
<<<<<<< HEAD


=======
>>>>>>> b6641299b1b791cecfab931ada1b5c90f71e07a9
    # Without using rospy.wait_for_message and using the standard rospy.Subscriber()
    '''image_sub = rospy.Subscriber("/camera/color/image_rect_color", Image, image_callback)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()'''
