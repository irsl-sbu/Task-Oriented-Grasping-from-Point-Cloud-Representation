#!/usr/bin/env python
# Python script to extract points corresponding to an object from a point cloud based on the bounding box provided by pretrained YoloV5
# and semantic segmentation provided by pretrained DeepLab.
# By: Aditya Patankar

import rospy
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as Rot
import sys
import numpy as np
import torch
import pyrealsense2 as rs2
import open3d as o3d

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image as PILImage 
from torchvision import transforms
from dataloaders.utils import  *

class realsenseSubscriber(object):

    def __init__(self, imageTopic, depthTopic, cameraInfoTopic, cameraPoseTopic, path):
        self.bridge = CvBridge()

        # Topics related to Sensor Data from the Camera:
        self.subImage = rospy.Subscriber(imageTopic, Image, self.imageCallback)
        self.subDepth = rospy.Subscriber(depthTopic, Image, self.depthCallback)
        self.subCamInfo = rospy.Subscriber(cameraInfoTopic, CameraInfo, self.cameraInfoCallback)

        # Attribute to save the camera pose from the camerPoseTopic: 
        self.cameraPose = None
        self.cameraPoseTopic = cameraPoseTopic

        # boundingBoxCoords is a list which contains the pixel coordinates for the boudning box obtained from YOLO.
        self.boundingBoxCoords = None
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

        # Attributes corresponding to the output of the semantic segmentation:
        self.resultSemanticSegmentation = None
        self.index_pairs = None

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
        # Loading the YOLOv5 model and initializing it with the pretrained weights
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/marktwo/Aditya/object-detection-main/best.pt')
        output = model(self.img)
        output.print()
        output.show()
        output_list = output.pandas().xyxy[0].iloc[0].values.tolist()
        self.boundingBoxCoords = np.round(np.asarray(output_list[0:4]))
        print('Output List: ', output_list)
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
        
        self.index_pairs = []
        self.resultBoundingBox = []
        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                self.index_pairs.append(np.array([i,j]))
                depth = self.depthArray[j][i]
                # depthScaled = np.true_divide(depth, self.depthScale)
                # coord = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [i,j], depthScaled)
                coord = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [i,j], depth)
                self.resultBoundingBox.append(coord)
        
        self.resultBoundingBox = np.array(self.resultBoundingBox)
        return self.resultBoundingBox

    # Function which performs semantic segmentation on the input image using Deeplab and pretrained weights:
    def getCoordSemanticSeg(self):
        model = DeepLab(num_classes=28,
                    backbone='resnet',
                    output_stride=16,
                    sync_bn=False,
                    freeze_bn=False)
        
        ckpt = torch.load('/home/marktwo/realsense_ws/src/perception/src/model_best.pth', map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])

        composed_transforms = transforms.Compose([
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

        image = PILImage.fromarray(self.img).convert('RGB')
        target = PILImage.fromarray(self.img).convert('L')
        
        sample = {'image': image, 'label': target}
        tensor_in = composed_transforms(sample)['image'].unsqueeze(0)

        model.eval()
        with torch.no_grad():
            seg_logit = model(tensor_in)
        seg_pred = seg_logit.argmax(dim=1)[0]
            
        unique, counts = np.unique(seg_pred.numpy(), return_counts=True)
        labels = dict(zip(unique, counts))
        print("Labels Dictionary: ", labels)

        img_w, img_h = image.size[0], image.size[1]

        print("img_w: ", img_w)
        print("img_h: ", img_h)

        print('seg_pred Shape before reshaping: ', seg_pred.shape)

        # Checking how we can access the labels:
        seg_pred_reshaped = np.reshape(seg_pred, [img_w, img_h])
        print("seg_pred Shape after reshaping: ", seg_pred_reshaped.shape)

        # Label of the desired object:
        '''label 4: Cheezit'''
        # desired_label = 4
        '''label 5: Domino Sugar'''
        # desired_label = 5
        '''label 15: Spam Tuna'''
        # desired_label = 15
        '''label for red screw driver: 18'''
        # desired_label = 18
        '''label for tomato soup can: 23'''
        # desired_label = 17
        '''desired label for pringles:'''
        desired_label = 5

        print("shape of the depth array: ", self.depthArray.shape)
        
        self.index_pairs = []
        self.resultSemanticSegmentation = []
        for i in range(img_h):
            for j in range(img_w):
                # if seg_pred_reshaped[i,j] == desired_label:
                if seg_pred[i,j] == desired_label:
                    self.index_pairs.append(np.array([i,j]))
                    # depth = self.depthArray[j][i]
                    depth = self.depthArray[i, j]
                    # depthScaled = np.true_divide(depth, self.depthScale)
                    # coord = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [i,j], depthScaled)
                    coord = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [j,i], depth)
                    self.resultSemanticSegmentation.append(coord)
        
        self.resultSemanticSegmentation = np.array(self.resultSemanticSegmentation)
        
        return self.resultSemanticSegmentation

    def getEndEffectorPose(self):
        self.cameraPose = rospy.wait_for_message(self.cameraPoseTopic, Pose, timeout=5)
        pose = np.zeros([4,4])
        pose[0, 3] = self.cameraPose.position.x
        pose[1, 3] = self.cameraPose.position.y
        pose[2, 3] = self.cameraPose.position.z
        pose[3, 3] = 1
        quat = [self.cameraPose.orientation.x, self.cameraPose.orientation.y, self.cameraPose.orientation.z, self.cameraPose.orientation.w]
        r = Rot.from_quat(quat)
        rotation_matrix = r.as_matrix()
        pose[0:3, 0:3] = rotation_matrix
        return pose