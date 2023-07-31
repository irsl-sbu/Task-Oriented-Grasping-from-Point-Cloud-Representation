#!/usr/bin/env python
# Script for publishing the grasping pose:
# By: Aditya Patankar

import rospy
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import os
import sys
from geometry_msgs.msg import Pose
import csv

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

class ObjectPosePublisher(object):

  def __init__(self):
    self.pub_handle = rospy.Publisher('/grasp_pose', Pose, queue_size=10)
    
    self.sequence_cnt = 0
    
  def publishPoses(self, obj_pose):
    self.pub_handle.publish(obj_pose)
    

pose_publisher = ObjectPosePublisher()

current_dir = '/home/marktwo/Aditya/IROS_Experiments/Pouring/Pringles/Pose10_EE10/'

# grasping_pose = np.asarray(readCSV(current_dir + 'pringles_pouring_pose1_EE1_grasp_pose_base_frame_alg1.csv'))
grasping_pose = np.asarray(readCSV(current_dir + 'pringles_pouring_pose7_EE8_grasp_pose_base_frame_alg3.csv'))
print(grasping_pose)
print(grasping_pose.shape)
rot_quat_inter = Rot.from_matrix(grasping_pose[0:3, 0:3])
rot_quat_inter = rot_quat_inter.as_quat()

grasp_pose_inter = Pose()
grasp_pose_inter.position.x = grasping_pose[0,3]
grasp_pose_inter.position.y = grasping_pose[1,3]
grasp_pose_inter.position.z = grasping_pose[2,3]

grasp_pose_inter.orientation.x = rot_quat_inter[0]
grasp_pose_inter.orientation.y = rot_quat_inter[1]
grasp_pose_inter.orientation.z = rot_quat_inter[2]
grasp_pose_inter.orientation.w = rot_quat_inter[3]



if __name__ == '__main__':
  
  # Initialize ROS node
  rospy.init_node('grasping_pose_publisher', anonymous=True)
  
  rate = rospy.Rate(1)
  
  while not rospy.is_shutdown():
    pose_publisher.publishPoses(grasp_pose_inter)
    rate.sleep()
    
  print("Shutting Down!")