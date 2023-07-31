#!/usr/bin/env python
# Python script to extract points corresponding to an object from a point cloud based on the bounding box.
# For objects which cannot be detected using YOLOv5 Object Detector.
# By: Aditya Patankar

import rospy
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
# Numpy packages 
import numpy as np

# Importing the pyrealsense2 package:
import pyrealsense2 as rs2

# Open3D for point cloud processing and visualization
import open3d as o3d

from process_point_cloud import pointCloud
from process_point_cloud import pointCloud

class realsenseSubscriber(object):

    def __init__(self, imageTopic, depthTopic, cameraInfoTopic, path):
        self.bridge = CvBridge()
        self.subImage = rospy.Subscriber(imageTopic, Image, self.imageCallback)
        self.subDepth = rospy.Subscriber(depthTopic, Image, self.depthCallback)
        self.subCamInfo = rospy.Subscriber(cameraInfoTopic, CameraInfo, self.cameraInfoCallback)

        # boundingBoxCoords is a list which contains the pixel coordinates for the boudning box obtained from YOLO.
        #SPATULA TRIAL 1: self.boundingBoxCoords = np.asarray([425, 187, 1002, 365])

        #SCREWDRIVER: self.boundingBoxCoords = np.asarray([416,304,827,387])

        #SPATULA TRIAL 2:
        self.boundingBoxCoords = np.asarray([512,321,985,624])

        self.img = None
        self.imageArray = None
        self.imagePath = path
        self.depthImage = None
        self.depthPath = None
        self.depthArray = None
        self.result = None
        self.resultBoundingBox = None
        self.intrinsics = None

        # Verify this value with the parameters we get from pyrealsense2 
        self.depthScale = 0.001

    '''This function is used to save the image only corresponding to the image topic'''
    def imageCallback(self, image):
        try:
            self.img = self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough')
            print('class of image: ', type(self.img))
            print('shape of image: ', self.img.shape)
            # Subscribe only once and then unregister.
            self.subImage.unregister()
        except CvBridgeError as error:
            print(error)
        # cv2.imwrite(self.imagePath + ".jpg", cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(3)
        

    '''This function is used to save the depth data by subscribing to the depth topic'''
    def depthCallback(self, data):
        try:
            self.depthImage = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.depthArray = np.array(self.depthImage, dtype=np.float32)
            # Subscribe only once and then unregister.
            self.subDepth.unregister()
        except CvBridgeError:
            print('Error!!')

    '''This function is used to get the camera intrinsic parameters by subscribing to the camera intrinsics topic.
       The parameters are then used by the pyrealsense2 package for pixel to point deprojection'''
    def cameraInfoCallback(self, cameraInfo):
        self.intrinsics = rs2.intrinsics()
        print('Intrinsics Object Created using Pyrealsense!')
        self.intrinsics.width = cameraInfo.width
        self.intrinsics.height = cameraInfo.height
        self.intrinsics.ppx = cameraInfo.K[2]
        self.intrinsics.ppy = cameraInfo.K[5]
        self.intrinsics.fx = cameraInfo.K[0]
        self.intrinsics.fy = cameraInfo.K[4]
        if cameraInfo.distortion_model == 'plumb_bob':
            self.intrinsics.model = rs2.distortion.brown_conrady
        elif cameraInfo.distortion_model == 'equidistant':
            self.intrinsics.model = rs2.distortion.kannala_brandt4
        self.intrinsics.coeffs = [i for i in cameraInfo.D]
        # Subscribe only once and then unregister.
        self.subCamInfo.unregister()

        print("Params Loaded!")

    '''This function is used to get the bounding box coordinates using YOLOv5'''
    def getBoundingBox(self):
        print('Bounding Box Coordinates: ', self.boundingBoxCoords)
        print('x_min: ', self.boundingBoxCoords[0])
        print('y_min: ', self.boundingBoxCoords[1])
        print('x_max: ', self.boundingBoxCoords[2])
        print('y_max: ', self.boundingBoxCoords[3])
    
    '''This function is used to implement the rs2_deproject_pixel_to_point() function provided by pyrealsense2.
       This a trial implementation of the functionality.'''
    def getCoordinates(self):
        indices = np.array(np.where(self.depthImage == self.depthImage[self.depthImage > 0].min()))[:,0]
        pix = (indices[1], indices[0])
        depth = self.depthArray[pix[1], pix[0]]
        depthScaled = np.true_divide(depth, self.depthScale)
        result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix[0], pix[1]], depthScaled)
        return result

    '''This function is used to implement the rs2_deproject_pixel_to_point() function provided by pyrealsense2.
       We use this function to get the point corresponding to the entire scene by using the rs2_deproject_pixel_to_point() function.'''
    def getCoordComplete(self):
        index_pairs = []
        self.result = []
        for i in range(self.depthImage.shape[1]):
            for j in range(self.depthImage.shape[0]):
                index_pairs.append(np.array([i,j]))
                depth = self.depthArray[j][i]
                # We need to update the depth scaling and the way we are doing it. We need some information of where the depth sensors are located and everything.
                # depthScaled = np.true_divide(depth, self.depthScale)
                # coord = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [i,j], depthScaled)
                coord = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [i,j], depth)
                self.result.append(coord)
        self.result = np.array(self.result)
        return self.result

    '''This function is used to implement the rs2_deproject_pixel_to_point() function provided by pyrealsense2.
       We use this function to get the point corresponding to the bounding box by using the rs2_deproject_pixel_to_point() function.'''
    def getCoordBoundingBox(self):
        # bounding_box is a list which contains the coordinates as follows: [u_1, v_1, u_2, v_2]
        print('Shape of depth image: ', self.depthImage.shape)
        # This needs to be changed! We need to convert this to a function wherein we provide the coordinates as an input argument and don't have 
        
        for i in range(len(self.boundingBoxCoords)):
            num = self.boundingBoxCoords[i]
            if num%1 == 0:
                self.boundingBoxCoords[i] = int(num) 

        x_min = int(self.boundingBoxCoords[0])
        y_min = int(self.boundingBoxCoords[1])
        x_max = int(self.boundingBoxCoords[2])
        y_max = int(self.boundingBoxCoords[3])

        #depthImageCrop = self.depthImage[y_min:y_max, x_min:x_max]
        #depthArrayCrop = self.depthArray[y_min:y_max, x_min:x_max]
        
        index_pairs = []
        self.resultBoundingBox = []
        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                index_pairs.append(np.array([i,j]))
                depth = self.depthArray[j][i]
                # depthScaled = np.true_divide(depth, self.depthScale)
                # coord = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [i,j], depthScaled)
                coord = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [i,j], depth)
                self.resultBoundingBox.append(coord)
        
        self.resultBoundingBox = np.array(self.resultBoundingBox)
        return self.resultBoundingBox

if __name__ == '__main__':
    rospy.init_node('listenerNode', anonymous=True)
    imageTopic = "/camera/color/image_rect_color"
    depthTopic = "/camera/aligned_depth_to_color/image_raw"
    cameraInfoTopic = "/camera/aligned_depth_to_color/camera_info"
    path = "/home/aditya/catkin_ws/src/perception/src/Saved_Images/"+"cheezit"

    print('Realsense Subscriber Object initialized')
    print('---------------------------------------')

    realsenseObject = realsenseSubscriber(imageTopic, depthTopic, cameraInfoTopic, path)
    rospy.spin()

    print(realsenseObject.boundingBoxCoords)

    ##Getting the coordinates corresponding to the bounding box
    realsenseObject.getBoundingBox()
    realsenseObject.getCoordBoundingBox()
    dataPointsBB = realsenseObject.getCoordBoundingBox()

    # Creating a Open3d PointCloud Object for the cloud corresponding to just the bounding box
    cheezitCloud = o3d.geometry.PointCloud()
    cheezitCloud.points = o3d.utility.Vector3dVector(dataPointsBB.astype(np.float64))
    cheezitCloud.paint_uniform_color([0, 0, 1])

    # Visualizing just the CheezIt point cloud using open3D:
    o3d.visualization.draw_geometries([cheezitCloud])
    
    # np.savetxt("dataPointsBB_cheezit_trial10.csv", dataPointsBB, delimiter= ",")
    # np.savetxt("dataPointsBB_screwdriver_trial3.csv", dataPointsBB, delimiter= ",")
    #np.savetxt("dataPointsBB_spatula_trial5.csv", dataPointsBB, delimiter= ",")

    # Saving the point cloud in the necessary format:
    # o3d.io.write_point_cloud("cheezit_trial10_BB.ply", cheezitCloud)
    #o3d.io.write_point_cloud("spatula_trial5_BB.ply", cheezitCloud)
    #o3d.io.write_point_cloud("screwdriver_trial3_BB.ply", cheezitCloud)

    print('Bounding Box data saved!')

    ## Getting the complete coordinates
    dataPoints = realsenseObject.getCoordComplete()

    # Creating a Open3d PointCloud Object for the entire cloud
    cheezitCloudComplete = o3d.geometry.PointCloud()
    cheezitCloudComplete.points = o3d.utility.Vector3dVector(dataPoints.astype(np.float64))
    cheezitCloudComplete.paint_uniform_color([0, 1, 0])

    # Saving the point cloud in the necessary format:
    # o3d.io.write_point_cloud("cheezit_trial10_Complete.ply", cheezitCloudComplete)
    #o3d.io.write_point_cloud("spatula_trial5_Complete.ply", cheezitCloudComplete)
    # o3d.io.write_point_cloud("screwdriver_trial3_Complete.ply", cheezitCloudComplete)

    # np.savetxt("dataPoints_cheezit_trial10_complete.csv", dataPoints, delimiter= ",")
    #np.savetxt("dataPoints_spatula_trial5_complete.csv", dataPoints, delimiter= ",")    
    # np.savetxt("dataPoints_screwdriver_trial3_complete.csv", dataPoints, delimiter= ",")
    print('Complete data saved!')

    cloud_object = pointCloud()
    cloud_object.cloud = cheezitCloud

    # Pose of the camera reference frame with respect to the base reference frame: 
    cloud_object.g_base_cam = np.asarray([[0.264841,  -0.724186,   0.636703,   0.484874],
                                          [-0.960315,  -0.138317,   0.242136,   0.178317],
                                          [-0.0872844,  -0.675565,   -0.73211,   0.559299],
                                          [0,          0,          0,          1]])

    # Transforming the point cloud in the Panda base reference frame: 
    cloud_object.transformToBase()

    # Downsample it and inspect the normals
    cloud_object.cloud = cloud_object.cloud.voxel_down_sample(voxel_size=0.009)
    cloud_object.removePlaneSurface()

    # Visualizing the downsampled point cloud. 
    o3d.visualization.draw_geometries([cloud_object.cloud])

    # Specifying parameters for DBSCAN Clustering:
    # Just like the parameters for downsampling even the parameters for DBSCAN Clustering are dependent on the 
    # units used computing and extracting the point cloud data.
    cloud_object.eps = 0.05
    cloud_object.min_points = 10
    cloud_object.getObjectPointCloud()

    # Visualizing the downsampled point cloud. 
    o3d.visualization.draw_geometries([cloud_object.cloud])
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
    # Saving the vertices and the centers of the bounding boxes: 
    # np.savetxt('oriented_box_center_cheezit_trial10.csv', cloud_object.oriented_bounding_box_center, delimiter = ",")
    #np.savetxt('oriented_box_vertices_cheezit_trial10.csv', cloud_object.oriented_bounding_box_vertices, delimiter = ",")

    #np.savetxt('oriented_box_center_screwdriver_trial3.csv', cloud_object.oriented_bounding_box_center, delimiter = ",")
    #np.savetxt('oriented_box_vertices_screwdriver_trial3.csv', cloud_object.oriented_bounding_box_vertices, delimiter = ",")

    #np.savetxt('oriented_box_center_spatula_trial5.csv', cloud_object.oriented_bounding_box_center, delimiter = ",")
    #np.savetxt('oriented_box_vertices_spatula_trial5.csv', cloud_object.oriented_bounding_box_vertices, delimiter = ",")

    #np.savetxt('aligned_box_center_cheezit_trial10.csv', cloud_object.aligned_bounding_box_center, delimiter = ",")
    #np.savetxt('aligned_box_vertices_cheezit_trial10.csv', cloud_object.aligned_bounding_box_vertices, delimiter = ",")

    #np.savetxt('aligned_box_center_screwdriver_trial3.csv', cloud_object.aligned_bounding_box_center, delimiter = ",")
    #np.savetxt('aligned_box_vertices_screwdriver_trial3.csv', cloud_object.aligned_bounding_box_vertices, delimiter = ",")

    #np.savetxt('aligned_box_center_spatula_trial5.csv', cloud_object.aligned_bounding_box_center, delimiter = ",")
    #np.savetxt('aligned_box_vertices_spatula_trial5.csv', cloud_object.aligned_bounding_box_vertices, delimiter = ",")

    # Saving the processed point cloud: 
    #o3d.io.write_point_cloud("cheezit_trial10_Processed_Transformed.ply", cloud_object.processed_cloud)
    #o3d.io.write_point_cloud("screwdriver_trial3_Processed_Transformed.ply", cloud_object.processed_cloud)
    #o3d.io.write_point_cloud("spatula_trial5_Processed_Transformed.ply", cloud_object.processed_cloud)

