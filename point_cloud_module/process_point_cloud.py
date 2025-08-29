# By: Aditya Patankar

# Open3D for point cloud processing and visualization
import open3d as o3d

import numpy as np
import math
from numpy import linalg as la
from scipy.spatial import ConvexHull

# Matplotlib for plotting and visualization in Python:
import matplotlib.pyplot as plt

# PyTorch for Neural Network Approximation: 
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from neural_network_module.neural_net import metric_nn
from neural_network_module.neural_net import metric_nn_generic
from neural_network_module.data_loader import metric_nn_dataset
from neural_network_module.data_loader import to_tensor

from collections import Counter

class point_cloud(object):
   
   def __init__(self): 
      
      # Attributes associated with Point Cloud Preprocessing: 
      self.cloud = None
      self.processed_cloud = None
      self.cloud_object_frame = None

      self.normals = None
      self.normals_base_frame = None
      self.normals_object_frame = None
      self.points = None
      
      self.oriented_bounding_box = None
      self.aligned_bounding_box = None
      
      self.oriented_bounding_box_vertices = None
      self.aligned_bounding_box_vertices = None
      
      self.oriented_bounding_box_center = None
      self.aligned_bounding_box_center = None
      
      self._eps = None
      self._min_points = None

      # Attributes associated with Point Cloud Transformation 
      self._bounding_box_flag = None

      self._R_object = None
      self._p_object = None

      self._R_base = None
      self._p_base = None
      self._g_base = None

      self.g_base_cam = None
      self.R_base_cam = None
      self.p_base_cam = None

      self.R_bounding_box = None
      self.p_bounding_box = None

      # Additional variable to store the pose of the bounding box as a 4x4 matrix:
      self.g_bounding_box = None

      # Dimensions of the bounding box:
      self.dimensions = None
      self.x_dim = None
      self.y_dim = None
      self.z_dim = None

      # Additional attribute introduced to determine pivoting edge:
      self.edge_dict_object = None
      self.edge_dict_base = None

      # Points of the point cloud transformed and expressed in the local object reference frame:
      self.transformed_points_object_frame = None

      # Vertices of the bounding box expressed in the local object reference frame:
      self.transformed_vertices_object_frame = None
 
   def _remove_plane_surface(self):
      '''
         Function to process the point clouds based on the normal information.
         Input: Downsampled Point Cloud Object
         Output: Point Cloud Object after removing the points corresponding to the flat surfaces/tables
      '''
      # Invalidating the existing normals
      self.cloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
      # Estimating new normals for the point cloud
      self.cloud.estimate_normals()
      self.cloud.orient_normals_consistent_tangent_plane(30)

      self.points = np.asarray(self.cloud.points)
      # Extracting the points corresponding to the flat surfaces where the normals are along the (+/-) Z axis. 
      # NOTE: Here we are making the assumption that the unit normal vector corresponding to a point on a flat surface will 
      # be pointing towards the Z axis. The Z axis of the local object reference frame points upwards is another assumption
      # we are making
      self.normals = np.asarray(self.cloud.normals)
      point_indices = []
      for i, normal in enumerate(self.normals):
         max_idx = np.where(abs(self.normals[i]) == np.amax(np.absolute(normal)))
         if max_idx[0] == 2 and self.points[i, 2] <= -0.01:
               point_indices.append(i)
           
      # Get the points corresponding to the normals pointing upward.
      # NOTE: Here we assume that the normals and points are ordered in the same way. That is the index of the normal is same
      # as the index for the corresponding point
      self.points = np.delete(self.points, point_indices, axis=0)
      
      # Converting the processed points back to a Open3D PointCloud Object
      self.cloud = o3d.geometry.PointCloud()
      self.cloud.points = o3d.utility.Vector3dVector(self.points)
      self.cloud.paint_uniform_color([0, 0, 1])

   def _get_object_point_cloud(self):
      '''
         Function to preprocess the point cloud based on the clusters.
         Input: Processed Point Cloud after removing the points corresponding to the falt surfaces like tables.
         Output: (1)Point Cloud Object corresponding to the only the object in the scene which has to be manipulated.
               (2)Processed Point Cloud with Clusters.
      '''
      # Implementing DBSCAN Clustering to group local point cloud clusters together:
      with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
         labels = np.array(self.cloud.cluster_dbscan(eps=self._eps, min_points=self._min_points, print_progress=True))
         
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

      # Converting the processed points back to a Open3D PointCloud Object
      object_points = np.asarray([points[i] for i, label in enumerate(labels) if label == common_cluster])
      self.processed_cloud = o3d.geometry.PointCloud()
      self.processed_cloud.points = o3d.utility.Vector3dVector(object_points)
      self.processed_cloud.paint_uniform_color([0, 0, 1])

   def _transform_to_base(self):
      '''
         Function to transform the point cloud into the base reference frame from the camera reference frame. 
      '''
      # Convert point cloud into numpy array:
      points = np.asarray(self.cloud.points)

      # Rotation matrix of the camera frame with respect to the base frame: 
      self.R_base_cam = self.g_base_cam[0:3, 0:3]
      self.p_base_cam = np.reshape(self.g_base_cam[0:3, 3], [3,1])

      transformed_points = np.zeros([points.shape[0], points.shape[1]])
      
      # Transforming the points from the camera reference frame to the base reference frame: 
      for i, point in enumerate(points):
         transformed_points[i,:] = np.reshape(np.add(np.dot(self.R_base_cam, np.multiply(np.reshape(point, [3,1]), 0.001)), self.p_base_cam), [1,3])

      transformed_cloud = o3d.geometry.PointCloud()
      transformed_cloud.points = o3d.utility.Vector3dVector(transformed_points)
      transformed_cloud.paint_uniform_color([0, 0, 1])

      self.cloud = transformed_cloud

   def _compute_aabb(self):
      '''
         Function to compute the axis aligned bounding box corresponding 
         to the processed point cloud using Open3D.
      '''
      # Computing the axis aligned bounding box: 
      self.aligned_bounding_box = self.processed_cloud.get_axis_aligned_bounding_box()
      
      # This is just for visualization:
      self.aligned_bounding_box.color = (1, 0, 0)
    
      # Extracting the coordinates of the vertices of the oriented bounding box:
      self.aligned_bounding_box_vertices = np.asarray(self.aligned_bounding_box.get_box_points())

      # Extracting the center of the oriented bounding box:
      self.aligned_bounding_box_center = np.asarray(self.aligned_bounding_box.get_center())

   def _compute_obb(self):
      '''
         Function to compute the oriented bounding box after all the points have been transferred to the 
         object base frame computed using the axis aligned bounding box.
      '''
      # We transform the points into the object reference frame computed using the axis aligned bounding box 
      # and then compute the oriented bounding box:
      projected_points_object_frame = np.zeros([self.transformed_points_object_frame.shape[0], self.transformed_points_object_frame.shape[1]])
    
      for i, point in enumerate(projected_points_object_frame):
        projected_points_object_frame[i, 0] = self.transformed_points_object_frame[i, 0]
        projected_points_object_frame[i, 1] = self.transformed_points_object_frame[i, 1]
        projected_points_object_frame[i, 2] = 0

      projected_points_object_frame_2D = projected_points_object_frame[:, 0:2]

      # Computing the eigen values and eigen vectors:
      covar = np.cov(np.transpose(projected_points_object_frame_2D))
      eval, evecs = la.eig(covar)
      evecs = np.around(evecs, 4)
    
      # Rearranging the eigen vectors to make sure they match the ones generated by MATLAB.
      # This is being done to maintain consistency in the results.
      evecs = -1*evecs
      evec_1 = np.asarray([evecs[0, 1], evecs[0, 0]])
      evec_2 = np.asarray([evecs[1, 1], evecs[1, 0]])
      evecs[0, :] = evec_1
      evecs[1, :] = evec_2
    
      # Centering the points:
      mean = np.mean(np.transpose(projected_points_object_frame_2D), axis = 1)
      projected_points_object_frame_centered =  np.transpose(projected_points_object_frame_2D) - mean[:, np.newaxis]

      # Transforming the points into the new coordinate basis corresponding to the eigen vectors:
      projected_points_object_frame_transformed = np.dot(evecs, projected_points_object_frame_centered)

      # Computing the min and max using the new coordinate basis:
      min_X = np.amin(projected_points_object_frame_transformed[0, :])
      max_X = np.amax(projected_points_object_frame_transformed[0, :])

      min_Y = np.amin(projected_points_object_frame_transformed[1, :])
      max_Y = np.amax(projected_points_object_frame_transformed[1, :])

      # OBB_coords_transformed = np.asarray([[max_X, min_Y],[max_X, max_Y],[min_X, min_Y],[min_X, max_Y]])
      OBB_coords_transformed = np.asarray([[max_X, max_Y],[max_X, min_Y],[min_X, max_Y],[min_X, min_Y]])
      OBB_center_transformed = np.reshape(np.asarray([(max_X + min_X)/2, (max_Y + min_Y)/2]), [1,2])

      # Transforming the bounding box coordinates to the object reference frame: 
      OBB_coords_2D = np.dot(la.inv(evecs), np.transpose(OBB_coords_transformed)) + mean[:, np.newaxis]
      OBB_center_2D = np.dot(la.inv(evecs), np.transpose(OBB_center_transformed)) + mean[:, np.newaxis]

      # Computing the center of the 3D oriented bounding box in the reference frame of the axis aligned bounding box:
      self.oriented_bounding_box_center = np.zeros([3, 1])
      self.oriented_bounding_box_center[0, :] = OBB_center_2D[0, :]
      self.oriented_bounding_box_center[1, :] = OBB_center_2D[1, :]
      self.oriented_bounding_box_center[2, :] = 0

      # Computing the vertices of the 3D oriented bounding box in the reference frame of the axis aligned bounding box:
      self.oriented_bounding_box_vertices = np.zeros([self.aligned_bounding_box_vertices.shape[1], self.aligned_bounding_box_vertices.shape[0]])
      self.oriented_bounding_box_vertices[0:2, 0:3] = OBB_coords_2D[0:2, 0:3] 
      self.oriented_bounding_box_vertices[0:2, 7] = OBB_coords_2D[0:2, 3]
      self.oriented_bounding_box_vertices[0:2, 3] = OBB_coords_2D[0:2, 0]
      self.oriented_bounding_box_vertices[0:2, 4] = OBB_coords_2D[0:2, 3]
      self.oriented_bounding_box_vertices[0:2, 5] = OBB_coords_2D[0:2, 2]
      self.oriented_bounding_box_vertices[0:2, 6] = OBB_coords_2D[0:2, 1]

      self.oriented_bounding_box_vertices = np.transpose(self.oriented_bounding_box_vertices)

      self.oriented_bounding_box_vertices[0, 2] = self.transformed_vertices_object_frame[0, 2]
      self.oriented_bounding_box_vertices[1, 2] = self.transformed_vertices_object_frame[1, 2]
      self.oriented_bounding_box_vertices[2, 2] = self.transformed_vertices_object_frame[2, 2]
      self.oriented_bounding_box_vertices[3, 2] = self.transformed_vertices_object_frame[3, 2]
      self.oriented_bounding_box_vertices[4, 2] = self.transformed_vertices_object_frame[4, 2]
      self.oriented_bounding_box_vertices[5, 2] = self.transformed_vertices_object_frame[5, 2]
      self.oriented_bounding_box_vertices[6, 2] = self.transformed_vertices_object_frame[6, 2]
      self.oriented_bounding_box_vertices[7, 2] = self.transformed_vertices_object_frame[7, 2]

      # We need to transform the oriented_bounding_box_vertices back into the object base frame:
      for i, vertex in enumerate(self.oriented_bounding_box_vertices):
        self.oriented_bounding_box_vertices[i,:] = np.reshape(np.dot(np.transpose(np.add(np.reshape(self.oriented_bounding_box_vertices[i, :], [3,1]), self.p_bounding_box)), la.inv(self._R_object)), [1,3])

      # We need to compute the center of the oriented bounding and it cannot be the same as the
      # axis aligned bounding box. It should based upon the dimensions of the oriented bounding box.
      self.oriented_bounding_box_center = np.reshape(np.dot(np.transpose(np.add(self.oriented_bounding_box_center, self.p_bounding_box)), la.inv(self._R_object)), [3])

   def _compute_obb_rotating_calipers(self):
      '''
         Function to compute the oriented bounding box using the rotating calipers algorithm 
         after all the points have been transferred to the object base frame computed using the 
         axis aligned bounding box.
      '''
      projected_points_object_frame = np.zeros([self.transformed_points_object_frame.shape[0], self.transformed_points_object_frame.shape[1]])
    
      for i, point in enumerate(projected_points_object_frame):
        projected_points_object_frame[i, 0] = self.transformed_points_object_frame[i, 0]
        projected_points_object_frame[i, 1] = self.transformed_points_object_frame[i, 1]
        projected_points_object_frame[i, 2] = 0

      projected_points_object_frame_2D = projected_points_object_frame[:, 0:2]

      # NOTE: The below code, including the Convex Hull and Rotating Caliper Functions, is added to achieve an
      # optimal bounding box using the rotating calipers method, which searches all bounding boxes that
      # includes one of the edges of the convex hull of a 2D point cloud. This is guaranteed to find the
      # optimal bounding box with the smallest area. The first step is finding the Convex Hull Vertices,
      # followed by rotating the Convex Hull so one edge is axis aligned and calculating the AABB.
      # Input: 2D Point Cloud in the Object Frame -> Convex Hull Vertices
      # Output: Optimal Bounding Box Corners and Center
      #### Convex Hull
      points = projected_points_object_frame_2D
      hull = ConvexHull(points)

      #### ROTATING CALIPERS #### added 7/23/23
      min_area = 0
      min_i = 0
      for i, hull_vertex in enumerate(hull.vertices):

          # Rotate by the unit vector pointing between adjacent points on the hull
          cis = points[hull.vertices[i]] - points[hull.vertices[i-1]]
          cis /= math.sqrt(cis[0]**2 + cis[1]**2)
          rot = [[cis[0],cis[1]],[-cis[1],cis[0]]]
          points_rotated = np.dot(rot,np.transpose(points))

          # Min/Max bounding box (contains the adjacent points on an edge)
          min_X = np.amin(points_rotated[0, :])
          max_X = np.amax(points_rotated[0, :])
          min_Y = np.amin(points_rotated[1, :])
          max_Y = np.amax(points_rotated[1, :])

          # Calculate the area and compare to minimum area, save if less
          area = np.multiply(np.subtract(max_X,min_X),np.subtract(max_Y,min_Y))
          if min_area == 0:
             min_area = area
          if area < min_area:
              min_area = area
              min_i = i

      # Repeat the above steps for the minimum area bounding box
      cis = points[hull.vertices[min_i]] - points[hull.vertices[min_i-1]]
      cis /= math.sqrt(cis[0]**2 + cis[1]**2)
      rot = [[cis[0],cis[1]],[-cis[1],cis[0]]]
      points_rotated = np.dot(rot,np.transpose(points))

      min_X = np.amin(points_rotated[0, :])
      max_X = np.amax(points_rotated[0, :])
      min_Y = np.amin(points_rotated[1, :])
      max_Y = np.amax(points_rotated[1, :])

      # Transform Corners and Center of the Optimal Bounding Box to non-rotated frame
      corners = [[min_X,max_X,max_X,min_X,min_X],[min_Y,min_Y,max_Y,max_Y,min_Y]]
      corners = np.dot(la.inv(rot),corners)
      center = [(min_X+max_X)/2,(min_Y+max_Y)/2]
      center = np.dot(la.inv(rot),center)
      
      # Computing the center of the 3D oriented bounding box in the reference frame of the axis aligned bounding box:
      self.oriented_bounding_box_center = np.zeros([3, 1])
      self.oriented_bounding_box_center[0, :] = center[0]
      self.oriented_bounding_box_center[1, :] = center[1]
      self.oriented_bounding_box_center[2, :] = 0

      # Computing the vertices of the 3D oriented bounding box in the reference frame of the axis aligned bounding box:
      self.oriented_bounding_box_vertices = np.zeros([self.aligned_bounding_box_vertices.shape[1], self.aligned_bounding_box_vertices.shape[0]])
      self.oriented_bounding_box_vertices[0:2, 0:2] = corners[0:2, 0:2] 
      self.oriented_bounding_box_vertices[0:2, 2] = corners[0:2, 3]
      self.oriented_bounding_box_vertices[0:2, 7] = corners[0:2, 2]
      self.oriented_bounding_box_vertices[0:2, 3] = corners[0:2, 0]
      self.oriented_bounding_box_vertices[0:2, 4] = corners[0:2, 2]
      self.oriented_bounding_box_vertices[0:2, 5] = corners[0:2, 3]
      self.oriented_bounding_box_vertices[0:2, 6] = corners[0:2, 1]

      self.oriented_bounding_box_vertices = np.transpose(self.oriented_bounding_box_vertices)

      self.oriented_bounding_box_vertices[0, 2] = self.transformed_vertices_object_frame[0, 2]
      self.oriented_bounding_box_vertices[1, 2] = self.transformed_vertices_object_frame[1, 2]
      self.oriented_bounding_box_vertices[2, 2] = self.transformed_vertices_object_frame[2, 2]
      self.oriented_bounding_box_vertices[3, 2] = self.transformed_vertices_object_frame[3, 2]
      self.oriented_bounding_box_vertices[4, 2] = self.transformed_vertices_object_frame[4, 2]
      self.oriented_bounding_box_vertices[5, 2] = self.transformed_vertices_object_frame[5, 2]
      self.oriented_bounding_box_vertices[6, 2] = self.transformed_vertices_object_frame[6, 2]
      self.oriented_bounding_box_vertices[7, 2] = self.transformed_vertices_object_frame[7, 2]

      # We need to transform the oriented_bounding_box_vertices back into the object base frame:
      for i, vertex in enumerate(self.oriented_bounding_box_vertices):
        self.oriented_bounding_box_vertices[i,:] = np.reshape(np.dot(np.transpose(np.add(np.reshape(self.oriented_bounding_box_vertices[i, :], [3,1]), self.p_bounding_box)), la.inv(self._R_object)), [1,3])

      # We need to compute the center of the oriented bounding and it cannot be the same as the
      # axis aligned bounding box. It should based upon the dimensions of the oriented bounding box.
      self.oriented_bounding_box_center = np.reshape(np.dot(np.transpose(np.add(self.oriented_bounding_box_center, self.p_bounding_box)), la.inv(self._R_object)), [3])

   def _get_pose_bounding_box(self):
      ''' 
         Function to get the pose of the bounding box (axis-aligned or oriented). 
      '''
      # Computing the position and orienation of the axis aligned bounding box reference frame with respect to the 
      # base reference frame.

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

      # Getting the unit vectors corresponding the X and Y axis with respect to
      # the orientation of the aligned bounding box:
      x_axis = np.divide(np.subtract(vertices[7, :], vertices[2, :]), la.norm(np.subtract(vertices[7, :], vertices[2, :])))
      y_axis = np.divide(np.subtract(vertices[7, :], vertices[1, :]), la.norm(np.subtract(vertices[7, :], vertices[1, :])))
      z_axis = np.cross(x_axis, y_axis)

      # Orientation of the bounding box and its position: 
      self.R_bounding_box = np.zeros([3,3])
      self.R_bounding_box[:, 0] = x_axis
      self.R_bounding_box[:, 1] = y_axis
      self.R_bounding_box[:, 2] = z_axis
      self.p_bounding_box = np.reshape(center, [3,1])

      # Pose of the bounding box:
      self.g_bounding_box = np.zeros([4,4])
      self.g_bounding_box[0:3, 0:3] = self.R_bounding_box
      self.g_bounding_box[0:3, 3] = np.reshape(self.p_bounding_box, [3])
      self.g_bounding_box[3,3] = 1

   def _transform_to_object_frame(self):
      '''
         Function to transform the object point cloud and its bounding box to an object reference frame based on the 
         bounding box.
      '''
      # Transforming all the points such that they are expressed in the object reference frame:
      self.points = np.asarray(self.processed_cloud.points)
      self._R_object = np.matmul(self._R_base, self.R_bounding_box)
      self.transformed_points_object_frame = np.zeros([self.points.shape[0], self.points.shape[1]])

      if self.bounding_box_flag == 0:
         vertices = self.aligned_bounding_box_vertices
      elif self.bounding_box_flag == 1:
         vertices = self.oriented_bounding_box_vertices
      else:
         print('Please update the bounding box flag')

      for i, point in enumerate(self.points):
         self.transformed_points_object_frame[i,:] = np.reshape(np.dot(np.transpose(np.subtract(np.reshape(point, [3,1]), self.p_bounding_box)), self._R_object), [1,3])

      # Transforming the vertices of the bounding box also to the object reference frame:
      self.transformed_vertices_object_frame = np.zeros([vertices.shape[0], vertices.shape[1]])

      for i, vertex in enumerate(vertices):
         self.transformed_vertices_object_frame[i,:] = np.reshape(np.dot(np.transpose(np.subtract(np.reshape(vertex, [3,1]), self.p_bounding_box)), self._R_object), [1,3])

      # Getting the dimensions of the box in terms of the X, Y and Z directions:
      self.x_dim = np.round(np.absolute(self.transformed_vertices_object_frame[0,0] - self.transformed_vertices_object_frame[1,0]),2)
      self.y_dim = np.round(np.absolute(self.transformed_vertices_object_frame[2,1] - self.transformed_vertices_object_frame[0,1]),2)
      self.z_dim = np.round(np.absolute(self.transformed_vertices_object_frame[3,2] - self.transformed_vertices_object_frame[0,2]),2)
      
      # Storing the dimensions of the bounding box in a single numpy array
      self.dimensions = np.zeros([3,1])
      self.dimensions[0] = self.x_dim
      self.dimensions[1] = self.y_dim
      self.dimensions[2] = self.z_dim

   def compute_bounding_box(self):
      '''
         Function to compute the bounding box given a the object point cloud with respect to a global/world/base reference frame.
         NOTE: The algorithm assumes that the point cloud corresponds to an object resting on a flat (tabular) surface and environment.
         It also assumes that the Z-axis of the global/base/world reference frame in which the point cloud is initially expressed points upwards. 
      '''
      # The object is aligned with the axis of the world/robot base reference frame.
      self.bounding_box_flag = 0

      # Computing the bounding boxes corresponding to the object point cloud: 
      self._compute_aabb()
      
      # Rotation matrix and position vector for the robot base or world reference frame: 
      self._R_base = np.identity(3)
      self._p_base = np.zeros([3,1])

      # Pose of the bounding box:
      self.g_base = np.zeros([4,4])
      self.g_base[0:3, 0:3] = self._R_base
      self.g_base[0:3, 3] = np.reshape(self._p_base, [3])
      self.g_base[3,3] = 1

      # Getting the pose of the bounding box:
      self._get_pose_bounding_box()

      # Transforming the points to the object reference frame:
      self._transform_to_object_frame()
      self._compute_obb_rotating_calipers()

      # We now change the bounding box flag and use the oriented bounding box to sample the contacts:
      self.bounding_box_flag = 1

      # Getting the pose of the bounding box:
      self._get_pose_bounding_box()

      # Transforming the points to the object reference frame:
      self._transform_to_object_frame()

      # Defining a dictionary containing the x, y and z dimensions along with the corresponding edges at the bottom face.
      # NOTE: This is especially useful while selecting the pivoting axis and it is based on the same convention which we use to assign the reference frame
      # 0 - x_dim, 1 - y_dim, 2 - z_dim
      self.edge_dict_object = {0:[self.transformed_vertices_object_frame[0,:],self.transformed_vertices_object_frame[1,:], self.transformed_vertices_object_frame[2,:], self.transformed_vertices_object_frame[7,:]], 
               1:[self.transformed_vertices_object_frame[0,:],self.transformed_vertices_object_frame[2,:], self.transformed_vertices_object_frame[1,:], self.transformed_vertices_object_frame[7,:]]
               }

      self.edge_dict_base = {0:[self.oriented_bounding_box_vertices[0,:],self.oriented_bounding_box_vertices[1,:], self.oriented_bounding_box_vertices[2,:], self.oriented_bounding_box_vertices[7,:]], 
               1:[self.oriented_bounding_box_vertices[0,:],self.oriented_bounding_box_vertices[2,:], self.oriented_bounding_box_vertices[1,:], self.oriented_bounding_box_vertices[7,:]]
               }

      # Saving the point cloud transformed to the object reference frame:
      # Creating a Open3d PointCloud Object for the cloud corresponding to just the bounding box
      self.cloud_object_frame = o3d.geometry.PointCloud()
      self.cloud_object_frame.points = o3d.utility.Vector3dVector(self.transformed_points_object_frame.astype(np.float64))
      self.cloud_object_frame.paint_uniform_color([0, 0, 1])

      self.cloud_object_frame.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))

      self.cloud_object_frame.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.], dtype="float64"))
      # Storing the normals as a numpy array: 
      self.normals_object_frame = np.asarray(self.cloud_object_frame.normals) 


   