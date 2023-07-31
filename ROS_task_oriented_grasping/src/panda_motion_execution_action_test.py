#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Pose
import actionlib
from task_based_grasping.msg import MotionExecutionAction, MotionExecutionGoal

rospy.init_node('panda_motion_test_node')

motion_execution_client = actionlib.SimpleActionClient('PandaMotionExecutionActionServer', MotionExecutionAction)
print('Waiting for server')
motion_execution_client.wait_for_server()

pose1 = Pose()
pose1.position.x = 0.65
pose1.position.y = 0
pose1.position.z = 0.35
pose1.orientation.w =  0.0
pose1.orientation.x =  1.0
pose1.orientation.y =  0.0
pose1.orientation.z =  0.0

pose2 = Pose()
pose2.position.x = 0.65
pose2.position.y = 0
pose2.position.z = 0.215
pose2.orientation.w =  0.0
pose2.orientation.x =  1.0
pose2.orientation.y =  0.0
pose2.orientation.z =  0.0

pose3 = Pose()
pose3.position.x = 0.55
pose3.position.y = 0.15
pose3.position.z = 0.50
pose3.orientation.w =  0.0
pose3.orientation.x =  1.0
pose3.orientation.y =  0.0
pose3.orientation.z =  0.0

gripper_state = [False, False, True]

motion_goal = MotionExecutionGoal()
motion_goal.ee_trajectory = [pose1, pose2, pose3]
motion_goal.gripper_state = gripper_state

motion_execution_client.send_goal(motion_goal)
