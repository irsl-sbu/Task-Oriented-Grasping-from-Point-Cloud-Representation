# Open3D for point cloud processing and visualization
import open3d as o3d

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

from collections import Counter


class pointCloud(object):

   def __init__(self): 

      # Attributes associated with Point Cloud Preprocessing: 
      self.cloud = None
      self.processed_cloud = None

      self.normals = None
      self.points = None
      
      self.oriented_bounding_box = None
      self.aligned_bounding_box = None
      
      self.oriented_bounding_box_vertices = None
      self.aligned_bounding_box_vertices = None
      
      self.oriented_bounding_box_center = None
      self.aligned_bounding_box_center = None
      
      self.eps = None
      self.min_points = None

      # Attributes associated with Point Cloud Transformation 
      self.bounding_box_flag = None

      self.R_object = None
      self.p_object = None

      self.R_base = None
      self.p_base = None

      self.g_base_cam = None
      self.R_base_cam = None
      self.p_base_cam = None

      self.R_bounding_box = None
      self.p_bounding_box = None

      self.R_local = None
      self.p_local = None

      self.dimensions = None

      self.transformed_points_object_frame = None
      self.transformed_vertices_object_frame = None
      
      # Attributes associated with sampling contact locations on the bounding box
      self.screw_axis = None
      self.point = None
      self.moment = None

      self.increment = None
      self.x_axis_increments = None
      self.y_axis_increments = None
      self.z_axis_increments = None

      self.sampled_c1 = None
      self.sampled_c2 = None

      # Attributes associated with Metric Prediction for the sampled contacts
      self.x_data = None
      self.y_data = None

      self.input_size = 12
      self.hidden_size_1 = 8
      self.hidden_size_2 = 3
      self.batch_size = 2

      self.test_datapoints = None
      self.predicted = None 
      self.ground_truth = None

      # Attributes associated with Point Cloud Projection, Grid Generation and Occupancy checking
      self.projected_points = None
      self.projected_points_local = None

      self.grid_points = None
      self.metric_values = None

      self.X_grid_points = None
      self.Y_grid_points = None
      self.Z_grid_points = None

      self.X_grid_points_matrix = None
      self.X_grid_points_occupied = None

      self.Y_grid_points_matrix = None
      self.Y_grid_points_occupied = None

      self.Z_grid_points_matrix = None
      self.Z_grid_points_occupied = None 

      self.grid_metric_values = None
      self.metric_grid = None
      self.grid_metric_values_occupied = None

      self.x_counter = None
      self.y_counter = None
      self.z_counter = None

      # Attributes associated with sampling object-robot contacts and end-effector reference frames:
      self.sampled_metric_values = None

      self.object_robot_contact_c1 = None
      self.object_robot_contact_c2 = None
      self.object_robot_contacts_object_frame = None
      self.object_robot_contacts_base_frame = None

      self.base_center_point = None

      self.end_effector_position_1 = None
      self.end_effector_position_2 = None
      self.end_effector_position_object_frame = None
      self.end_effector_position_base_frame = None

      # Attributes associated with visualization:
      # Plotting a cuboid: 
      self.faces = None
      self.faces_vertices = None
      self.vertices = None

      # Plotting a reference frame: 
      self.R = None
      self.p = None
      self.scale_value = None
      self.length_value = None
      
      
   '''Function to process the point clouds based on the normal information.
   Input: Downsampled Point Cloud Object
   Output: Point Cloud Object after removing the points corresponding to the flat surfaces/tables'''
   
   def removePlaneSurface(self):
      # Invalidating the existing normals
      self.cloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
      # Estimating new normals for the point cloud
      self.cloud.estimate_normals()
      self.cloud.orient_normals_consistent_tangent_plane(30)

      self.points = np.asarray(self.cloud.points)
      # Extracting the points corresponding to the flat surfaces where the normals are along the (+/-) Z axis. 
      '''Here we are making the assumption that the unit normal vector corresponding to a point on a flat surface will 
         be pointing towards the Z axis. The Z axis of the local object reference frame points upwards is another assumption
         we are making'''
      self.normals = np.asarray(self.cloud.normals)
      point_indices = []
      point_normals = []
      for i in range(len(self.normals)):
         max_element = np.amax(np.absolute(self.normals[i]))
         max_idx = np.where(abs(self.normals[i]) == max_element)
         #if max_idx[0] == 2 and self.points[i, 2] <= 0:
         if max_idx[0] == 2 and self.points[i, 2] <= -0.01:
         #if max_idx[0] == 0:
               point_indices.append(i)
               point_normals.append(self.normals[i])
               
      # Get the points corresponding to the normals pointing upward.
      '''Here we assume that the normals and points are ordered in the same way. That is the index of the normal is same
         as the index for the corresponding point'''
      # self.points = np.asarray(self.cloud.points)
      self.points = np.delete(self.points, point_indices, axis=0)
      
      # Converting the processed points back to a Open3D PointCloud Object
      self.cloud = o3d.geometry.PointCloud()
      self.cloud.points = o3d.utility.Vector3dVector(self.points)
      self.cloud.paint_uniform_color([0, 0, 1])


   '''Function to preprocess the point cloud based on the clusters.
   Input: Processd Point Cloud after removing the points corresponding to the falt surfaces like tables.
   Output: (1)Point Cloud Object corresponding to the only the object in the scene which has to be manipulated.
           (2)Processed Point Cloud with Clusters.
   '''
   def getObjectPointCloud(self):
      # Implementing DBSCAN Clustering to group local point cloud clusters together:
      with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
         labels = np.array(self.cloud.cluster_dbscan(eps=self.eps, min_points=self.min_points, print_progress=True))
         
      # Visualizing the clusters (this part needs to be understood better)
      max_label = labels.max()
      print(f"point cloud has {max_label + 1} clusters")
      colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
      colors[labels < 0] = 0
      self.cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
      
      # Getting the maximum occurences of an element in a list:
      labels_list = labels.tolist()
      var = Counter(labels_list)
      common_cluster = var.most_common(1)[0][0]
      
      # Convert point cloud into numpy array:
      points = np.asarray(self.cloud.points)
      points_list = []
      for i in range(len(labels)):
         if labels[i] == common_cluster:
               points_list.append(points[i])
      
      # Converting the processed points back to a Open3D PointCloud Object
      object_points = np.asarray(points_list)
      self.processed_cloud = o3d.geometry.PointCloud()
      self.processed_cloud.points = o3d.utility.Vector3dVector(object_points)
      self.processed_cloud.paint_uniform_color([0, 0, 1])

   '''Function to transform the point cloud into the base reference frame from the camera reference frame. '''
   def transformToBase(self):
      # Convert point cloud into numpy array:
      points = np.asarray(self.cloud.points)

      # Rotation matrix of the camera frame with respect to the base frame: 
      self.R_base_cam = self.g_base_cam[0:3, 0:3]
      self.p_base_cam = np.reshape(self.g_base_cam[0:3, 3], [3,1])

      transformed_points = np.zeros([points.shape[0], points.shape[1]])

      # Transforming the points from the camera reference frame to the base reference frame: 
      for i in range(len(points)):
         point = np.reshape(points[i,:], [3,1])
         # We convert all the datapoints into metres before applying the transform. 
         scaled_point = np.multiply(point, 0.001)
         prod = np.dot(self.R_base_cam, scaled_point)
         result = np.add(prod, self.p_base_cam)
         transformed_points[i,:] = np.reshape(result, [1,3])

      transformed_cloud = o3d.geometry.PointCloud()
      transformed_cloud.points = o3d.utility.Vector3dVector(transformed_points)
      transformed_cloud.paint_uniform_color([0, 0, 1])

      self.cloud = transformed_cloud

      '''Function to compute the axis aligned and oriented bounding boxes corresponding 
         to the processed point cloud.
      '''
   def computeBoundingBox(self):

      # Computing the axis aligned bounding box: 
      self.aligned_bounding_box = self.processed_cloud.get_axis_aligned_bounding_box()
      # This is just for visualization:
      self.aligned_bounding_box.color = (1, 0, 0)
    
      # Computing the oriented bounding box: 
      self.oriented_bounding_box = self.processed_cloud.get_oriented_bounding_box()
      # This is just for visualization:
      self.oriented_bounding_box.color = (0, 1, 0)
    
      # Extracting the coordinates of the vertices of the oriented bounding box:
      self.oriented_bounding_box_vertices = np.asarray(self.oriented_bounding_box.get_box_points())

      # Extracting the center of the oriented bounding box:
      self.oriented_bounding_box_center = np.asarray(self.oriented_bounding_box.get_center())
    
      # Extracting the coordinates of the vertices of the oriented bounding box:
      self.aligned_bounding_box_vertices = np.asarray(self.aligned_bounding_box.get_box_points())

      # Extracting the center of the oriented bounding box:
      self.aligned_bounding_box_center = np.asarray(self.aligned_bounding_box.get_center())


   ''' Function to get the pose of the bounding box (aligned or oriented): '''
   def getPoseBoundingBox(self):
      # Computing the position and orienation of the axis aligned bounding box reference frame with respect to the 
      # base reference frame: 

      # We define local variables vertices and center which are not attributes of the class. 
      # They are set based on the boundin_box_flag, which tells us whether our object is axis aligned or not.
      if self.bounding_box_flag == 0:
         vertices = self.aligned_bounding_box_vertices
         center = self.aligned_bounding_box_center
      elif self.bounding_box_flag == 1:
         vertices = self.oriented_bounding_box_vertices
         center = self.oriented_bounding_box_center
      else:
         print('Please update the bounding box flag')

      # Getting the dimensions of the box in terms of the X, Y and Z directions:
      x_L = np.round(np.absolute(vertices[0,0] - vertices[1,0]),2)
      y_W = np.round(np.absolute(vertices[2,1] - vertices[0,1]),2)
      z_H = np.round(np.absolute(vertices[3,2] - vertices[0,2]),2)
      
      # Storing the dimensions of the bounding box in a single numpy array
      self.dimensions = np.zeros([3,1])
      self.dimensions[0] = x_L
      self.dimensions[1] = y_W
      self.dimensions[2] = z_H

      # Getting the unit vectors corresponding the X and Y axis with respect to
      # the orientation of the aligned bounding box:
      x_axis = np.divide(np.subtract(vertices[1, :], vertices[0, :]), la.norm(np.subtract(vertices[1, :], vertices[0, :])))
      y_axis = np.divide(np.subtract(vertices[2, :], vertices[0, :]), la.norm(np.subtract(vertices[2, :], vertices[0, :])))
      z_axis = np.cross(x_axis, y_axis)

      # Orientation of the bounding box and its position: 
      self.R_bounding_box = np.zeros([3,3])
      self.R_bounding_box[0, :] = x_axis
      self.R_bounding_box[1, :] = y_axis
      self.R_bounding_box[2, :] = z_axis
      self.p_bounding_box = np.reshape(center, [3,1])

   '''Function to transform the object point cloud and its bounding box to an object reference frame based on the 
   bounding box: '''
   def transformToObjectFrame(self):
      # Transforming all the points such that they are expressed in the object reference frame:
      self.points = np.asarray(self.processed_cloud.points)
      self.R_object = np.matmul(self.R_base, self.R_bounding_box)
      self.transformed_points_object_frame = np.zeros([self.points.shape[0], self.points.shape[1]])

      if self.bounding_box_flag == 0:
         vertices = self.aligned_bounding_box_vertices
      elif self.bounding_box_flag == 1:
         vertices = self.oriented_bounding_box_vertices
      else:
         print('Please update the bounding box flag')

      for i in range(len(self.points)):
         point = np.reshape(self.points[i,:], [3,1])
         sub = np.transpose(np.subtract(point, self.p_bounding_box))
         result = np.dot(sub, self.R_object)
         self.transformed_points_object_frame[i,:] = np.reshape(result, [1,3])

      # Transforming the vertices of the bounding box also to the object reference frame:
      self.transformed_vertices_object_frame = np.zeros([vertices.shape[0], vertices.shape[1]])

      for i in range(len(vertices)):
         vertex = np.reshape(vertices[i,:], [3,1])
         sub = np.transpose(np.subtract(vertex, self.p_bounding_box))
         result = np.dot(sub, self.R_object)
         self.transformed_vertices_object_frame[i,:] = np.reshape(result, [1,3])


   '''Function to sample contacts from the bounding box. At this moment we are just sampling from two parallel faces, 
      we need to sample from the other two as well.'''
   def sampleContactsYZ(self):
      # Sampling contacts as input to the neural network. We will be using the transformed points and transformed 
      # vertices for sampling the antipodal contact locations. 
      counter = 0
      self.y_axis_increments = np.arange(self.transformed_vertices_object_frame[1,1], self.transformed_vertices_object_frame[7,1], self.increment)
      self.z_axis_increments = np.arange(self.transformed_vertices_object_frame[0,2], self.transformed_vertices_object_frame[3,2], self.increment)

      # Initializing arrays to sample the contact locations. These two arrays are just for visualization purposes:
      self.sampled_c1 = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 3])
      self.sampled_c2 = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 3])

      # We need to initialize sampled_c2 as a different array. Even if we sampled_c1 as an array of zeros by writing sampled_c1 = sampled_c2 we will get errors.
      
      # N x 12 array for storing the datapoints used as an input to the neural network.
      self.x_data = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 12])

      # Sampling Antipodal contact locations: 
      for i in range(len(self.z_axis_increments)):
         for j in range(len(self.y_axis_increments)):
            position_vect_c1 = [self.transformed_vertices_object_frame[0,0], self.y_axis_increments[j], self.z_axis_increments[i]]
            # print("C1: ", position_vect_c1)
            position_vect_c2 = [self.transformed_vertices_object_frame[1,0], self.y_axis_increments[j], self.z_axis_increments[i]]
            # print("C2: ", position_vect_c2)
            self.sampled_c1[counter, :] = position_vect_c1
            # print("Sampled C1: ", sampled_c1[counter, :])
            self.sampled_c2[counter, :] = position_vect_c2
            # print("Sampled C2: ", sampled_c2[counter, :])
            self.x_data[counter, :] = np.reshape(np.asarray([position_vect_c1, position_vect_c2, self.screw_axis, self.moment]), [1,12])
            counter += 1

      # Generate empty data for 
      self.y_data = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 1])

   '''Function to sample contacts from the bounding box. At this moment we are just sampling from two parallel faces, 
      we need to sample from the other two as well.'''
   def sampleContactsXZ(self):
      # Sampling contacts as input to the neural network. We will be using the transformed points and transformed 
      # vertices for sampling the antipodal contact locations. 
      counter = 0
      self.x_axis_increments = np.arange(self.transformed_vertices_object_frame[0,0], self.transformed_vertices_object_frame[1,0], self.increment)
      self.z_axis_increments = np.arange(self.transformed_vertices_object_frame[0,2], self.transformed_vertices_object_frame[3,2], self.increment)

      # Initializing arrays to sample the contact locations. These two arrays are just for visualization purposes:
      self.sampled_c1 = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 3])
      self.sampled_c2 = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 3])

      # We need to initialize sampled_c2 as a different array. Even if we sampled_c1 as an array of zeros by writing sampled_c1 = sampled_c2 we will get errors.
      
      # N x 12 array for storing the datapoints used as an input to the neural network.
      self.x_data = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 12])

      # Sampling Antipodal contact locations: 
      for i in range(len(self.z_axis_increments)):
         for j in range(len(self.x_axis_increments)):
            position_vect_c1 = [self.x_axis_increments[j], self.transformed_vertices_object_frame[0,1], self.z_axis_increments[i]]
            position_vect_c2 = [self.x_axis_increments[j], self.transformed_vertices_object_frame[2,1], self.z_axis_increments[i]]
            self.sampled_c1[counter, :] = position_vect_c1
            self.sampled_c2[counter, :] = position_vect_c2
            self.x_data[counter, :] = np.reshape(np.asarray([position_vect_c1, position_vect_c2, self.screw_axis, self.moment]), [1,12])
            counter += 1

      # Generate empty data for the corresponding y labels required as input to the Pytorch DataLoader class
      self.y_data = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 1])

   '''This function is used to predict the metric values using the datapoints as input'''
   def predictMetric(self, model, testLoader):
      # TESTING LOOP:
      # We define an empty list to store all the predicted values:
      output_predicted = []
      output_ground_truth = []
      output_test_datapoints = []
      # We need to verify what this command does and if it is in fact necessary:
      model.eval()
      with torch.no_grad():
         for i, test_batch in enumerate(testLoader):
               outputs = model(test_batch['X'])
               output_predicted.append(outputs.tolist())
               # predicted.append(outputs.flatten().tolist())
               output_test_datapoints.append(test_batch['X'].tolist())
               output_ground_truth.append(test_batch['Y'].tolist())

      # Extracting the output in the required format since the output of the neural network is in the form of 
      # multidimensional array which includes datapoints grouped together in batches.
      self.predicted = []
      self.ground_truth = []
      self.test_datapoints = []
      for i in range(len(output_test_datapoints)):
         if i != len(output_test_datapoints)-1:
               for j in range(len(output_test_datapoints[0])):
                  self.predicted.append(output_predicted[i][j])
                  self.ground_truth.append(output_ground_truth[i][j])
                  self.test_datapoints.append(output_test_datapoints[i][j])
         elif i == len(output_test_datapoints)-1:
               for j in range(len(output_test_datapoints[len(output_test_datapoints)-1])):
                  self.predicted.append(output_predicted[i][j])
                  self.ground_truth.append(output_ground_truth[i][j])
                  self.test_datapoints.append(output_test_datapoints[i][j])
         else:
               print('Invalid data!')

      self.test_datapoints = np.asarray(self.test_datapoints)
      self.predicted = np.asarray(self.predicted)
      self.ground_truth = np.asarray(self.ground_truth)

   ''' Function to project the points onto a one of the surfaces of the bounding box:'''
   def projectPointsYZ(self):
      # PROJECTING THE POINTS ON TO ONE OF THE PLANES: 
      # Computing the equation of the plane. For our case we will be considering two planes. But for the sake of implementation 
      # we are only considering one plane: 
      # plane_points = [transformed_vertices[1, :], transformed_vertices[6, :], transformed_vertices[4, :], transformed_vertices[7, :]]

      # Computing the two vectors associated with three points: 
      vect_1 = np.subtract(self.transformed_vertices_object_frame[7, :], self.transformed_vertices_object_frame[4, :])
      vect_2 = np.subtract(self.transformed_vertices_object_frame[7, :], self.transformed_vertices_object_frame[1, :])

      # Computing the center point on the plane just for visualization: 
      center_point_plane = [self.transformed_vertices_object_frame[1, 0], self.p_base[1], self.p_base[2]]

      # Normal vector corresponding to the plane:
      n = np.cross(vect_1, vect_2)
      unit_n = np.divide(n, la.norm(n))

      # Translating each of the points by a specified distance: 
      # Computing the D in the equation of the plane Ax + By + Cz + D = 0:
      D = np.dot(center_point_plane, unit_n)

      # Initializing an empty array to store the projected points: 
      self.projected_points = np.zeros([self.transformed_points_object_frame.shape[0], self.transformed_points_object_frame.shape[1]])

      for i in range(len(self.transformed_points_object_frame)):
         new_point = self.transformed_points_object_frame[i, :]
         num = la.norm(unit_n[0]*new_point[0] + unit_n[1]*new_point[1] + unit_n[2]*new_point[2] - D)
         den = np.sqrt(unit_n[0]**2 + unit_n[1]**2 + unit_n[2]**2)
         distance = np.divide(num, den)
         projected_point = np.add(new_point, np.dot(distance, unit_n))
         self.projected_points[i, :] = projected_point

   ''' Function to project the points onto a one of the surfaces of the bounding box:'''
   def projectPointsXZ(self):
      # PROJECTING THE POINTS ON TO ONE OF THE PLANES: 
      # Computing the equation of the plane. For our case we will be considering two planes. But for the sake of implementation 
      # we are only considering one plane: 
      # plane_points = [transformed_vertices[1, :], transformed_vertices[6, :], transformed_vertices[4, :], transformed_vertices[7, :]]

      # Computing the two vectors associated with three points: 
      vect_1 = np.subtract(self.transformed_vertices_object_frame[3, :], self.transformed_vertices_object_frame[6, :])
      vect_2 = np.subtract(self.transformed_vertices_object_frame[1, :], self.transformed_vertices_object_frame[6, :])

      # Computing the center point on the plane just for visualization: 
      center_point_plane = [self.p_base[1], self.transformed_vertices_object_frame[1, 1], self.p_base[2]]

      # Normal vector corresponding to the plane:
      n = np.cross(vect_1, vect_2)
      unit_n = np.divide(n, la.norm(n))

      # Translating each of the points by a specified distance: 
      # Computing the D in the equation of the plane Ax + By + Cz + D = 0:
      D = np.dot(center_point_plane, unit_n)

      # Initializing an empty array to store the projected points: 
      self.projected_points = np.zeros([self.transformed_points_object_frame.shape[0], self.transformed_points_object_frame.shape[1]])

      for i in range(len(self.transformed_points_object_frame)):
         new_point = self.transformed_points_object_frame[i, :]
         num = la.norm(unit_n[0]*new_point[0] + unit_n[1]*new_point[1] + unit_n[2]*new_point[2] - D)
         den = np.sqrt(unit_n[0]**2 + unit_n[1]**2 + unit_n[2]**2)
         distance = np.divide(num, den)
         projected_point = np.add(new_point, np.dot(distance, unit_n))
         self.projected_points[i, :] = projected_point

   ''' Function to get a metric value corresponding to a datapoint:'''
   def getMetricValue(self, point):
      point = np.reshape(np.around(point, 3), [2,1])
      for i in range(len(self.grid_points)):
         grid_point = np.reshape(np.around(self.grid_points[i, :], 3), [2,1])
         if np.array_equal(point, grid_point):
               metric = self.metric_values[i, :]
      return metric

   ''' Function to generate a grid on the surface of the bounding box based on the computed metric values:''' 
   def generateGridYZ(self):
      # GRID GENERATION: 

      # Now we generate the grid using the multidimensional arrays 'x_data' and 'metric_values'
      # Since we are sampling contact locations along the Y and Z axis, we also need to do something similar to 
      # X and Z axis of the object lcoal reference frame: 
      self.y_counter = 0
      self.z_counter = 0
      counter = 0

      # We will be using x_data along with test_datapoints and predicted metric values: 
      self.metric_values = np.zeros([self.predicted.shape[0], self.predicted.shape[1]])
      self.grid_points = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 2])

      '''This code block needs to be changed and made more efficient!'''
      for i in range(len(self.x_data)):
         x_datapoint = np.around(self.x_data[i, :], 3)
         for j in range(len(self.test_datapoints)):
               test_datapoint = np.around(self.test_datapoints[j, :], 3)
               if np.array_equal(test_datapoint, x_datapoint):
                  label = self.predicted[j, :]
         self.metric_values[i, :] = label

      for i in range(len(self.z_axis_increments)):
         for j in range(len(self.y_axis_increments)):
               self.grid_points[counter, :] = np.reshape(np.asarray([self.y_axis_increments[j], self.z_axis_increments[i]]), [1,2])
               counter += 1

      # Initializing the empty arrays:
      self.grid_metric_values = np.zeros([len(self.y_axis_increments), len(self.z_axis_increments), 1])
      self.Y_grid_points_matrix = np.zeros([len(self.y_axis_increments), len(self.z_axis_increments), 4])
      self.Z_grid_points_matrix = np.zeros([len(self.y_axis_increments), len(self.z_axis_increments), 4])

      self.Y_grid_points = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 4])
      self.Z_grid_points = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 4])
      self.metric_grid = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 1])
      self.X_grid_points = self.transformed_vertices_object_frame[1,0]*np.ones([self.Y_grid_points.shape[0], self.Y_grid_points.shape[1]])

      # Resetting the counter value back to 0 so that it doesn't cause any further errors.
      counter = 0

      print('Generating Grid ... ')
      for i in range(len(self.z_axis_increments)):
         # Termination criteria for the Z Axis:

         if self.z_axis_increments[i] == self.z_axis_increments[len(self.z_axis_increments)-1]:
               break
         else:
               self.y_counter = 0
               for j in range(len(self.y_axis_increments)):
                  if self.y_axis_increments[j] == self.y_axis_increments[len(self.y_axis_increments)-1]:
                     break
                  else:
                     # For the grid the points are arranged in an anticlockwise order:
                     p_1 = [self.y_axis_increments[j], self.z_axis_increments[i]]
                     p_2 = [self.y_axis_increments[j] + self.increment, self.z_axis_increments[i]]
                     p_3 = [self.y_axis_increments[j] + self.increment, self.z_axis_increments[i] + self.increment]
                     p_4 = [self.y_axis_increments[j], self.z_axis_increments[i] + self.increment]

                     # Get the corresponding metric values:
                     '''eta_1 = getMetricValue(p_1, test_datapoints[:, 1:3], metric_values)
                     eta_2 = getMetricValue(p_1, test_datapoints[:, 1:3], metric_values)
                     eta_3 = getMetricValue(p_1, test_datapoints[:, 1:3], metric_values)
                     eta_4 = getMetricValue(p_1, test_datapoints[:, 1:3], metric_values)'''

                     eta_1 = self.getMetricValue(p_1)
                     eta_2 = self.getMetricValue(p_2)
                     eta_3 = self.getMetricValue(p_3)
                     eta_4 = self.getMetricValue(p_4)

                     # Concatenate and store the values in a single numpy array:
                     eta_values = np.asarray([eta_1, eta_2, eta_3, eta_4])

                     # Computing the average of the 4 points:
                     eta_avg = np.mean(eta_values)

                     # Storing the grid points and the metric values:
                     self.grid_metric_values[self.y_counter, self.z_counter, :] = eta_avg
                     self.metric_grid[counter, :] = eta_avg

                     # Storing the grid intervals and the transformed grid intervals in the same format:
                     self.Y_grid_points_matrix[self.y_counter, self.z_counter, :] = np.reshape(np.asarray([self.y_axis_increments[j], self.y_axis_increments[j] + self.increment, self.y_axis_increments[j] + self.increment, self.y_axis_increments[j]]), [4])
                     self.Z_grid_points_matrix[self.y_counter, self.z_counter, :] = np.reshape(np.asarray([self.z_axis_increments[i], self.z_axis_increments[i], self.z_axis_increments[i] + self.increment, self.z_axis_increments[i] + self.increment]), [4])
                     self.Y_grid_points[counter, :] = np.reshape(np.asarray([self.y_axis_increments[j], self.y_axis_increments[j] + self.increment, self.y_axis_increments[j] + self.increment, self.y_axis_increments[j]]), [4])
                     self.Z_grid_points[counter, :] = np.reshape(np.asarray([self.z_axis_increments[i], self.z_axis_increments[i], self.z_axis_increments[i] + self.increment, self.z_axis_increments[i] + self.increment]), [4])
                     
                     # Updating the two inner loop counters.
                     self.y_counter += 1
                     counter += 1

               # Updating outer loop counter.
               self.z_counter += 1


   ''' Function to generate a grid on the surface of the bounding box based on the computed metric values:''' 
   def generateGridXZ(self):
      # GRID GENERATION: 

      # Now we generate the grid using the multidimensional arrays 'x_data' and 'metric_values'
      # Since we are sampling contact locations along the Y and Z axis, we also need to do something similar to 
      # X and Z axis of the object lcoal reference frame: 
      self.x_counter = 0
      self.z_counter = 0
      counter = 0

      # We will be using x_data along with test_datapoints and predicted metric values: 
      self.metric_values = np.zeros([self.predicted.shape[0], self.predicted.shape[1]])
      self.grid_points = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 2])

      '''This code block needs to be changed and made more efficient!'''
      for i in range(len(self.x_data)):
         x_datapoint = np.around(self.x_data[i, :], 3)
         for j in range(len(self.test_datapoints)):
               test_datapoint = np.around(self.test_datapoints[j, :], 3)
               if np.array_equal(test_datapoint, x_datapoint):
                  label = self.predicted[j, :]
         self.metric_values[i, :] = label

      for i in range(len(self.z_axis_increments)):
         for j in range(len(self.x_axis_increments)):
               self.grid_points[counter, :] = np.reshape(np.asarray([self.x_axis_increments[j], self.z_axis_increments[i]]), [1,2])
               counter += 1


      # Initializing the empty arrays:
      self.grid_metric_values = np.zeros([len(self.x_axis_increments), len(self.z_axis_increments), 1])
      self.X_grid_points_matrix = np.zeros([len(self.x_axis_increments), len(self.z_axis_increments), 4])
      self.Z_grid_points_matrix = np.zeros([len(self.x_axis_increments), len(self.z_axis_increments), 4])

      self.X_grid_points = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 4])
      self.Z_grid_points = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 4])
      self.metric_grid = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 1])
      self.Y_grid_points = self.transformed_vertices_object_frame[1,1]*np.ones([self.X_grid_points.shape[0], self.X_grid_points.shape[1]])

      # Resetting the counter value back to 0 so that it doesn't cause any further errors.
      counter = 0

      print('Generating Grid ... ')
      for i in range(len(self.z_axis_increments)):
         # Termination criteria for the Z Axis:

         if self.z_axis_increments[i] == self.z_axis_increments[len(self.z_axis_increments)-1]:
               break
         else:
               self.x_counter = 0
               for j in range(len(self.x_axis_increments)):
                  if self.x_axis_increments[j] == self.x_axis_increments[len(self.x_axis_increments)-1]:
                     break
                  else:
                     # For the grid the points are arranged in an anticlockwise order:
                     p_1 = [self.x_axis_increments[j], self.z_axis_increments[i]]
                     p_2 = [self.x_axis_increments[j] + self.increment, self.z_axis_increments[i]]
                     p_3 = [self.x_axis_increments[j] + self.increment, self.z_axis_increments[i] + self.increment]
                     p_4 = [self.x_axis_increments[j], self.z_axis_increments[i] + self.increment]

                     # Get the corresponding metric values:
                     '''eta_1 = getMetricValue(p_1, test_datapoints[:, 1:3], metric_values)
                     eta_2 = getMetricValue(p_1, test_datapoints[:, 1:3], metric_values)
                     eta_3 = getMetricValue(p_1, test_datapoints[:, 1:3], metric_values)
                     eta_4 = getMetricValue(p_1, test_datapoints[:, 1:3], metric_values)'''

                     eta_1 = self.getMetricValue(p_1)
                     eta_2 = self.getMetricValue(p_2)
                     eta_3 = self.getMetricValue(p_3)
                     eta_4 = self.getMetricValue(p_4)

                     # Concatenate and store the values in a single numpy array:
                     eta_values = np.asarray([eta_1, eta_2, eta_3, eta_4])

                     # Computing the average of the 4 points:
                     eta_avg = np.mean(eta_values)

                     # Storing the grid points and the metric values:
                     self.grid_metric_values[self.x_counter, self.z_counter, :] = eta_avg
                     self.metric_grid[counter, :] = eta_avg

                     # Storing the grid intervals and the transformed grid intervals in the same format:
                     self.X_grid_points_matrix[self.x_counter, self.z_counter, :] = np.reshape(np.asarray([self.x_axis_increments[j], self.x_axis_increments[j] + self.increment, self.x_axis_increments[j] + self.increment, self.x_axis_increments[j]]), [4])
                     self.Z_grid_points_matrix[self.x_counter, self.z_counter, :] = np.reshape(np.asarray([self.z_axis_increments[i], self.z_axis_increments[i], self.z_axis_increments[i] + self.increment, self.z_axis_increments[i] + self.increment]), [4])
                     self.X_grid_points[counter, :] = np.reshape(np.asarray([self.x_axis_increments[j], self.x_axis_increments[j] + self.increment, self.x_axis_increments[j] + self.increment, self.x_axis_increments[j]]), [4])
                     self.Z_grid_points[counter, :] = np.reshape(np.asarray([self.z_axis_increments[i], self.z_axis_increments[i], self.z_axis_increments[i] + self.increment, self.z_axis_increments[i] + self.increment]), [4])
                     
                     # Updating the two inner loop counters.
                     self.x_counter += 1
                     counter += 1

               # Updating outer loop counter.
               self.z_counter += 1


   '''# Function to check the occupancy of the the point corresponding to the point cloud within the generate grid:'''
   def checkOccupancyYZ(self):
      # TRANSFORMING THE POINTS WITH RESPECT TO THE REFERENCE FRAME ATTACHED AT THE CORNER OF THE BOUNDING BOX:
      ''' This will be different for sampling points along the XZ axes as compared to the YZ axes, but it will be same for different instances
         of both. '''

      # Rotation matrix of the new local reference frame with respect to the object base reference frame:
      self.R_local = self.R_bounding_box

      # Position vector of the new local reference frame with respec to the object base reference frame:
      self.p_local = self.transformed_vertices_object_frame[1,:]

      '''Transforming the projected points from the object reference frame {O} to the local reference {L}: 
         This computation is important and may not always be valid for all instances of sampling along XZ and YZ planes.
         In this case we are essentially shifting the reference frame to the
         corner:'''

      self.projected_points_local = np.zeros([self.projected_points.shape[0], self.projected_points.shape[1]])
      for i in range(len(self.projected_points)):
         self.projected_points_local[i, :] = np.subtract(self.projected_points[i, :], self.p_local)

      # OCCUPANCY CHECK:
      '''The main reason for transforming the points to a local reference frame in the corner of the bounding box is 
         so that we can perform a 2D occupancy check over the generated grid. This will allow us to only have the grids 
         and associated grid points which have points belonging to the point cloud within their 2D bounds.'''

      # IMPORTANT: The FOR loop will be slightly different for the XZ and YZ planes.
      print('Checking Occupancy ... ')

      self.X_grid_points_occupied = self.transformed_vertices_object_frame[1,0]*np.ones([self.projected_points.shape[0],self.X_grid_points.shape[1]])
      self.Y_grid_points_occupied = np.zeros([self.projected_points.shape[0], self.Y_grid_points.shape[1]])
      self.Z_grid_points_occupied = np.zeros([self.projected_points.shape[0], self.Z_grid_points.shape[1]])
      self.grid_metric_values_occupied = np.zeros([self.projected_points.shape[0], 1])

      # Arrays to store the q_y and q_z values so that they can be studied and understood properly:
      # q_y_array = np.zeros([self.projected_points.shape[0], 1])
      # q_z_array = np.zeros([self.projected_points.shape[0], 1])

      for i in range(len(self.projected_points_local)):
         q_y = np.around(np.divide(self.projected_points_local[i, 1], self.increment))
         q_z = np.around(np.divide(self.projected_points_local[i, 2], self.increment))
         q_y_actual = np.divide(self.projected_points_local[i, 1], self.increment)
         q_z_actual = np.divide(self.projected_points_local[i, 2], self.increment)
         if q_y == 0 and q_z != 0:
               q_y = 1
         elif q_z == 0 and q_y != 0:
               q_z = 1
         elif q_y == 0 and q_z == 0:
               q_y = 1
               q_z = 1 
         elif q_y <= q_y_actual and q_y_actual <= self.y_counter:
               q_y = q_y + 1
         elif q_z <= q_z_actual and q_z_actual <= self.z_counter:
               q_z = q_z + 1

         q_y = int(q_y)
         q_z = int(q_z)

         if q_y <= self.y_counter and q_z <= self.z_counter:
               self.Y_grid_points_occupied[i, :] = np.reshape(self.Y_grid_points_matrix[q_y, q_z, :], [1,4])
               self.Z_grid_points_occupied[i, :] = np.reshape(self.Z_grid_points_matrix[q_y, q_z, :], [1,4])
               self.grid_metric_values_occupied[i, :] = np.reshape(self.grid_metric_values[q_y, q_z, :], [1])     
         else:
               continue

   ''' Function to check the occupancy of the the point corresponding to the point cloud within the generate grid:'''
   def checkOccupancyXZ(self):
      # TRANSFORMING THE POINTS WITH RESPECT TO THE REFERENCE FRAME ATTACHED AT THE CORNER OF THE BOUNDING BOX:
      ''' This will be different for sampling points along the XZ axes as compared to the YZ axes, but it will be same for different instances
         of both. '''

      # Rotation matrix of the new local reference frame with respect to the object base reference frame:
      self.R_local = self.R_bounding_box

      # Position vector of the new local reference frame with respec to the object base reference frame:
      self.p_local = self.transformed_vertices_object_frame[1,:]

      '''Transforming the projected points from the object reference frame {O} to the local reference {L}: 
         This computation is important and may not always be valid for all instances of sampling along XZ and YZ planes.
         In this case we are essentially shifting the reference frame to the
         corner:'''

      self.projected_points_local = np.zeros([self.projected_points.shape[0], self.projected_points.shape[1]])
      for i in range(len(self.projected_points)):
         self.projected_points_local[i, :] = np.subtract(self.projected_points[i, :], self.p_local)

      # OCCUPANCY CHECK:
      '''The main reason for transforming the points to a local reference frame in the corner of the bounding box is 
         so that we can perform a 2D occupancy check over the generated grid. This will allow us to only have the grids 
         and associated grid points which have points belonging to the point cloud within their 2D bounds.'''

      # IMPORTANT: The FOR loop will be slightly different for the XZ and YZ planes.
      print('Checking Occupancy ... ')

      self.Y_grid_points_occupied = self.transformed_vertices_object_frame[1,1]*np.ones([self.projected_points.shape[0],self.Y_grid_points.shape[1]])
      self.X_grid_points_occupied = np.zeros([self.projected_points.shape[0], self.X_grid_points.shape[1]])
      self.Z_grid_points_occupied = np.zeros([self.projected_points.shape[0], self.Z_grid_points.shape[1]])
      self.grid_metric_values_occupied = np.zeros([self.projected_points.shape[0], 1])

      for i in range(len(self.projected_points_local)):
         q_x = np.around(np.divide(self.projected_points_local[i, 0], self.increment))
         q_z = np.around(np.divide(self.projected_points_local[i, 2], self.increment))
         q_x_actual = np.divide(self.projected_points_local[i, 0], self.increment)
         q_z_actual = np.divide(self.projected_points_local[i, 2], self.increment)
         if q_x == 0 and q_z != 0:
               q_x = 1
         elif q_z == 0 and q_x != 0:
               q_z = 1
         elif q_x == 0 and q_z == 0:
               q_x = 1
               q_z = 1 
         elif q_x <= q_x_actual and q_x_actual <= self.x_counter:
               q_x = q_x + 1
         elif q_z <= q_z_actual and q_z_actual <= self.z_counter:
               q_z = q_z + 1

         q_x = int(q_x)
         q_z = int(q_z)

         if q_x <= self.x_counter and q_z <= self.z_counter:
               self.X_grid_points_occupied[i, :] = np.reshape(self.X_grid_points_matrix[q_x, q_z, :], [1,4])
               self.Z_grid_points_occupied[i, :] = np.reshape(self.Z_grid_points_matrix[q_x, q_z, :], [1,4])
               self.grid_metric_values_occupied[i, :] = np.reshape(self.grid_metric_values[q_x, q_z, :], [1])     
         else:
            continue
   
   '''Functions for sampling a single pose: '''
   def sampleSinglePoseYZ(self):
      # Getting the grid with the max metric value: 
      max_metric_value = np.amax(self.grid_metric_values_occupied)
      max_metric_indices = np.where(self.grid_metric_values_occupied == max_metric_value)
      list_of_coordinates = list(zip(max_metric_indices[0], max_metric_indices[1]))
      print(list_of_coordinates)
      rand_index = list_of_coordinates[3]

      Y_grid_points = self.Y_grid_points_occupied[rand_index[0], :]
      Z_grid_points = self.Z_grid_points_occupied[rand_index[0], :]

      Y_mid = Y_grid_points[0] + self.increment/2
      Z_mid = Z_grid_points[0] + self.increment/2

      self.object_robot_contact_c1 = np.asarray([self.transformed_vertices_object_frame[0,0], Y_mid, Z_mid])
      self.object_robot_contact_c2 = np.asarray([self.transformed_vertices_object_frame[1,0], Y_mid, Z_mid])
      self.object_robot_contacts_object_frame = np.asarray([self.object_robot_contact_c1, self.object_robot_contact_c2])
      
      center_point = [(self.transformed_vertices_object_frame[0,0] + self.transformed_vertices_object_frame[1,0])/2, Y_mid, Z_mid]
      
      self.end_effector_position_1 = [(self.transformed_vertices_object_frame[0,0] + self.transformed_vertices_object_frame[1,0])/2, Y_mid, Z_mid + 0.1034]
      self.end_effector_position_2 = [(self.transformed_vertices_object_frame[0,0] + self.transformed_vertices_object_frame[1,0])/2, Y_mid, Z_mid + 0.2]
      self.end_effector_position_object_frame = np.asarray([self.end_effector_position_1, self.end_effector_position_2])

      # Transforming the object contact points to the world reference frame:
      self.object_robot_contact_base_frame = np.zeros([self.object_robot_contacts_object_frame.shape[0], self.object_robot_contacts_object_frame.shape[1]])
      self.end_effector_position_base_frame = np.zeros([self.end_effector_position_object_frame.shape[0], self.end_effector_position_object_frame.shape[1]])

      for i in range(len(self.object_robot_contacts_object_frame)):
         point_c = np.reshape(self.object_robot_contacts_object_frame[i, :], [3,1])
         prod_c = np.dot(self.R_object, point_c)
         self.object_robot_contact_base_frame[i, :] = np.reshape(np.add(prod_c, self.p_bounding_box),[3])

      center_point = np.reshape(center_point, [3,1])
      prod_point = np.dot(self.R_object, center_point)
      self.base_center_point = np.add(prod_point, self.p_bounding_box)

      for i in range(len(self.end_effector_position_object_frame)):
         point_ee = np.reshape(self.end_effector_position_object_frame[i, :], [3,1])
         prod_ee = np.dot(self.R_object, point_ee)
         self.end_effector_position_base_frame[i, :] = np.reshape(np.add(prod_ee, self.p_bounding_box), [3])

   '''Functions for sampling a single pose: '''
   def sampleSinglePoseXZ(self):
      # Getting the grid with the max metric value: 
      max_metric_value = np.amax(self.grid_metric_values_occupied)
      max_metric_indices = np.where(self.grid_metric_values_occupied == max_metric_value)
      list_of_coordinates = list(zip(max_metric_indices[0], max_metric_indices[1]))
      rand_index = list_of_coordinates[1]

      X_grid_points = self.X_grid_points_occupied[rand_index[0], :]
      Z_grid_points = self.Z_grid_points_occupied[rand_index[0], :]

      X_mid = X_grid_points[0] + self.increment/2
      Z_mid = Z_grid_points[0] + self.increment/2

      self.object_robot_contact_c1 = np.asarray([X_mid, self.transformed_vertices_object_frame[0,1], Z_mid])
      self.object_robot_contact_c2 = np.asarray([X_mid, self.transformed_vertices_object_frame[2,1], Z_mid])
      self.object_robot_contacts_object_frame = np.asarray([self.object_robot_contact_c1, self.object_robot_contact_c2])
      
      center_point = [X_mid, (self.transformed_vertices_object_frame[0,1] + self.transformed_vertices_object_frame[2,1])/2, Z_mid]
      
      self.end_effector_position_1 = [X_mid, (self.transformed_vertices_object_frame[0,1] + self.transformed_vertices_object_frame[2,1])/2, Z_mid + 0.1034]
      self.end_effector_position_2 = [X_mid, (self.transformed_vertices_object_frame[0,1] + self.transformed_vertices_object_frame[2,1])/2, Z_mid + 0.2]
      self.end_effector_position_object_frame = np.asarray([self.end_effector_position_1, self.end_effector_position_2])

      # Transforming the object contact points to the world reference frame:
      self.object_robot_contact_base_frame = np.zeros([self.object_robot_contacts_object_frame.shape[0], self.object_robot_contacts_object_frame.shape[1]])
      self.end_effector_position_base_frame = np.zeros([self.end_effector_position_object_frame.shape[0], self.end_effector_position_object_frame.shape[1]])

      for i in range(len(self.object_robot_contacts_object_frame)):
         point_c = np.reshape(self.object_robot_contacts_object_frame[i, :], [3,1])
         prod_c = np.dot(self.R_object, point_c)
         self.object_robot_contact_base_frame[i, :] = np.reshape(np.add(prod_c, self.p_bounding_box),[3])

      center_point = np.reshape(center_point, [3,1])
      prod_point = np.dot(self.R_object, center_point)
      self.base_center_point = np.add(prod_point, self.p_bounding_box)

      for i in range(len(self.end_effector_position_object_frame)):
         point_ee = np.reshape(self.end_effector_position_object_frame[i, :], [3,1])
         prod_ee = np.dot(self.R_object, point_ee)
         self.end_effector_position_base_frame[i, :] = np.reshape(np.add(prod_ee, self.p_bounding_box), [3])

   '''Function to sample a single pose without the use of the grasp metric values.'''
   def sampleSinglePoseWOMetricYZ(self):
      # Now here we sample a single pose without using the computed metric values: 
      height = self.dimensions[2]
      Z_position = 0.7*height/2

      self.object_robot_contact_c1 = np.asarray([self.transformed_vertices_object_frame[0,0], self.aligned_bounding_box_center[1], Z_position ])
      self.object_robot_contact_c2 = np.asarray([self.transformed_vertices_object_frame[1,0], self.aligned_bounding_box_center[1], Z_position])
      self.object_robot_contacts_object_frame = np.asarray([self.object_robot_contact_c1, self.object_robot_contact_c2])

      center_point = [(self.transformed_vertices_object_frame[0,0] + self.transformed_vertices_object_frame[1,0])/2, self.aligned_bounding_box_center[1], Z_position]
      
      self.end_effector_position_1 = [(self.transformed_vertices_object_frame[0,0] + self.transformed_vertices_object_frame[1,0])/2, self.aligned_bounding_box_center[1], Z_position + 0.1034]
      self.end_effector_position_2 = [(self.transformed_vertices_object_frame[0,0] + self.transformed_vertices_object_frame[1,0])/2, self.aligned_bounding_box_center[1], Z_position + 0.2]
      self.end_effector_position_object_frame = np.reshape(np.asarray([self.end_effector_position_1, self.end_effector_position_2]), [2,3])

      # Transforming the object contact points to the world reference frame:
      self.object_robot_contact_base_frame = np.zeros([self.object_robot_contacts_object_frame.shape[0], self.object_robot_contacts_object_frame.shape[1]])
      self.end_effector_position_base_frame = np.zeros([self.end_effector_position_object_frame.shape[0], self.end_effector_position_object_frame.shape[1]])

      for i in range(len(self.object_robot_contacts_object_frame)):
         point_c = np.reshape(self.object_robot_contacts_object_frame[i, :], [3,1])
         prod_c = np.dot(self.R_object, point_c)
         self.object_robot_contact_base_frame[i, :] = np.reshape(np.add(prod_c, self.p_bounding_box),[3])

      center_point = np.reshape(center_point, [3,1])
      prod_point = np.dot(self.R_object, center_point)
      self.base_center_point = np.add(prod_point, self.p_bounding_box)

      for i in range(len(self.end_effector_position_object_frame)):
         point_ee = np.reshape(self.end_effector_position_object_frame[i, :], [3,1])
         prod_ee = np.dot(self.R_object, point_ee)
         self.end_effector_position_base_frame[i, :] = np.reshape(np.add(prod_ee, self.p_bounding_box), [3])

   '''Function to sample a single pose without the use of the grasp metric values.'''
   def sampleSinglePoseWOMetricXZ(self):
      # Now here we sample a single pose without using the computed metric values: 
      height = self.dimensions[2]
      Z_position = 0.7*height/2

      self.object_robot_contact_c1 = np.asarray([self.aligned_bounding_box_center[0], self.transformed_vertices_object_frame[0,1], Z_position ])
      self.object_robot_contact_c2 = np.asarray([self.aligned_bounding_box_center[0], self.transformed_vertices_object_frame[2,1], Z_position])
      self.object_robot_contacts_object_frame = np.asarray([self.object_robot_contact_c1, self.object_robot_contact_c2])

      center_point = [self.aligned_bounding_box_center[0], (self.transformed_vertices_object_frame[0,1] + self.transformed_vertices_object_frame[2,1])/2, Z_position]
      
      self.end_effector_position_1 = [self.aligned_bounding_box_center[0], (self.transformed_vertices_object_frame[0,1] + self.transformed_vertices_object_frame[2,1])/2, Z_position + 0.1034]
      self.end_effector_position_2 = [self.aligned_bounding_box_center[0], (self.transformed_vertices_object_frame[0,1] + self.transformed_vertices_object_frame[2,1])/2, Z_position + 0.2]
      self.end_effector_position_object_frame = np.reshape(np.asarray([self.end_effector_position_1, self.end_effector_position_2]), [2,3])

      # Transforming the object contact points to the world reference frame:
      self.object_robot_contact_base_frame = np.zeros([self.object_robot_contacts_object_frame.shape[0], self.object_robot_contacts_object_frame.shape[1]])
      self.end_effector_position_base_frame = np.zeros([self.end_effector_position_object_frame.shape[0], self.end_effector_position_object_frame.shape[1]])

      for i in range(len(self.object_robot_contacts_object_frame)):
         point_c = np.reshape(self.object_robot_contacts_object_frame[i, :], [3,1])
         prod_c = np.dot(self.R_object, point_c)
         self.object_robot_contact_base_frame[i, :] = np.reshape(np.add(prod_c, self.p_bounding_box),[3])

      center_point = np.reshape(center_point, [3,1])
      prod_point = np.dot(self.R_object, center_point)
      self.base_center_point = np.add(prod_point, self.p_bounding_box)

      for i in range(len(self.end_effector_position_object_frame)):
         point_ee = np.reshape(self.end_effector_position_object_frame[i, :], [3,1])
         prod_ee = np.dot(self.R_object, point_ee)
         self.end_effector_position_base_frame[i, :] = np.reshape(np.add(prod_ee, self.p_bounding_box), [3])

   '''Function for plotting a CUBE:'''
   def plotCube(self):
      # Processing the faces for the cube: 
      
      # The array 'faces_vertices' is based on the convention 
      self.faces_vertices = np.asarray([[1,0,3,6], [0,2,5,3], [2,7,4,5], [7,1,6,4], [1,0,2,7], [6,3,5,4]])
      
      # Initialize a list of vertex coordinates for each face
      self.faces = []
      self.faces.append(np.zeros([4,3]))
      self.faces.append(np.zeros([4,3]))
      self.faces.append(np.zeros([4,3]))
      self.faces.append(np.zeros([4,3]))
      self.faces.append(np.zeros([4,3]))
      self.faces.append(np.zeros([4,3]))
      
      for i in range(len(self.faces_vertices)):
         for j in range(self.faces_vertices.shape[1]):
               self.faces[i][j, 0] = self.vertices[self.faces_vertices[i,j],0]
               self.faces[i][j, 1] = self.vertices[self.faces_vertices[i,j],1]
               self.faces[i][j, 2] = self.vertices[self.faces_vertices[i,j],2]
               
   '''Function to plot a reference frame:'''
   def plotReferenceFrames(self, ax):
      ax.quiver(self.p[0], self.p[1], self.p[2], self.scale_value*self.R[0, 0], self.scale_value*self.R[1, 0], self.scale_value*self.R[2, 0], color = "r", arrow_length_ratio = self.length_value)
      ax.quiver(self.p[0], self.p[1], self.p[2], self.scale_value*self.R[0, 1], self.scale_value*self.R[1, 1], self.scale_value*self.R[2, 1], color = "g", arrow_length_ratio = self.length_value)
      ax.quiver(self.p[0], self.p[1], self.p[2], self.scale_value*self.R[0, 2], self.scale_value*self.R[1, 2], self.scale_value*self.R[2, 2], color = "b", arrow_length_ratio = self.length_value )
      
      return ax
