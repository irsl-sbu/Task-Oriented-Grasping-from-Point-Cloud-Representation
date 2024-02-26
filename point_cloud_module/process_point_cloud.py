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

      # Additional variable to store the pose of the bounding box as a 4x4 matrix:
      self.g_bounding_box = None

      self.R_local = None
      self.p_local = None

      # Dimensions of the bounding box:
      self.dimensions = None
      self.x_dim = None
      self.y_dim = None
      self.z_dim = None

      # Points of the point cloud transformed and expressed in the local object reference frame:
      self.transformed_points_object_frame = None

      # Vertices of the bounding box expressed in the local object reference frame:
      self.transformed_vertices_object_frame = None
      
      # Screw parameters:
      self.screw_axis = None
      self.point = None
      self.moment = None

      # Attributes associated with sampling contact locations on the bounding box
      self.increment = None
      self.x_axis_increments = None
      self.y_axis_increments = None
      self.z_axis_increments = None

      # Contacts generated on the faces of the bounding box:
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
      self.projected_points_local_frame = None

      self.X_grid_points = None
      self.Y_grid_points = None
      self.Z_grid_points = None

      self.X_grid_points_matrix = None
      self.X_grid_points_occupied = None

      self.Y_grid_points_matrix = None
      self.Y_grid_points_occupied = None

      self.Z_grid_points_matrix = None
      self.Z_grid_points_occupied = None

      self.grid_points = None
      self.metric_values = None 

      self.grid_metric_values = None
      self.metric_grid = None
      self.grid_metric_values_occupied = None

      # Attributes to store the center of the grids:
      self.grid_centers = None
      self.grid_centers_matrix = None
      self.grid_centers_occupied = None

      # Additional attributes included for improving the efficiency:
      self.grid_centers_unique = None
      self.grid_centers_unique_dict = None
      self.grid_centers_dict = None

      self.x_counter = None
      self.y_counter = None
      self.z_counter = None

      self.q_x_array = None
      self.q_y_array = None
      self.q_z_array = None

      # Attributes to store the center of the grids:
      self.grid_centers = None
      self.grid_centers_matrix = None
      self.grid_centers_occupied = None

      # Attributes associated with sampling object-robot contacts and end-effector reference frames:
      self.num_points = None

      # Parameters associated with computing the ideal grasping region
      self.max_metric_value = None
      self.eta_threshold = None
      self.ideal_grasping_region_points = None
      self.ideal_grasping_region_normals = None
      self.ideal_grasping_region_metric_values = None
      self.ideal_grasping_region_indices = None
      self.ideal_grasping_region_object_frame = None

      # Attribute associated with the grid centers of occupied grids corresponding to the ideal grasping region:
      self.ideal_grasping_region_grid_centers = None

      # Parameters associated with the bounding box corresponding to the ideal grasping region.
      # The bounding box computed will be with respect to the object reference frame. Therefore, it will always be axis-aligned
      self.ideal_grasping_region_bounding_box = None
      self.ideal_grasping_region_bounding_box_vertices = None
      self.ideal_grasping_region_bounding_box_center = None

      # Scalar value representing maximum allowable gripper width in metres 
      self.gripper_width_tolerance = None

      # Scalar value to avoid the gripper from colliding with the object while grasping
      self.gripper_height_tolerance = None

      # Additional attributes for computing the location corresponding to the grasp pose and pre-grasp pose:
      # For further information please refer to the documentation or Section IV. C of the IROS paper
      self.g_delta = None
      self.g_delta_inter = None

      # List to store the computed end-effector poses with respect to the object reference frame. 
      self.computed_end_effector_poses = None
      self.computed_end_effector_poses_inter = None

      # List to store the computed end-effector poses with respect to the base reference frame. 
      # Each element is 4x4 transformation matrix.
      self.computed_end_effector_poses_base = None
      self.computed_end_effector_poses_inter_base = None

      # List to store the computed end-effector poses with respect to the base reference frame. 
      # Each element is 7x1 quaternion.
      self.computed_end_effector_quaternion_base = None
      self.computed_end_effector_quaternion_inter_base = None

      # Coordinates of the bounding box which form the planes(faces of the object bounding box) corresponding to the approach directions:
      self.plane_points_1 = None
      self.plane_points_2 = None
      self.plane_points_3 = None

      # Center points of the planes(faces of the object bounding box) corresponding to the approach directions: 
      self.center_point_1 = None
      self.center_point_2 = None
      self.center_point_3 = None

      # Distance from the plane (faces of the object bounding box) to the computed grasp center:
      self.distance_1 = None
      self.distance_2 = None
      self.distance_3 = None

      # Unit vectors corresponding to the orientation of the end-effector reference frame:
      self.x_EE = None
      self.y_EE = None
      self.z_EE = None

      # Pose of the end-effector reference expressed as a 4x4 matrix (element of SE(3)):
      # The pose of the end-effector reference frame is expressed with respect to the object reference frame:
      self.R_EE = None
      self.R_EE_inter = None
      self.p_EE = None
      self.p_EE_inter = None
      self.gripper_pose = None
      self.gripper_pose_inter = None

      # The pose of the end-effector reference frame expressed with respect to the robot base reference frame:
      self.R_EE_base = None
      self.R_EE_inter_base = None
      self.p_EE_base = None
      self.p_EE_inter_base = None
      self.gripper_pose_base = None
      self.gripper_pose_inter_base = None
      
      # Unit vectors corresponding to the orientation of the contact reference frame C1:
      self.x_C1 = None
      self.y_C1 = None
      self.z_C1 = None

      # Unit vectors corresponding to the orientation of the contact reference frame C2:
      self.x_C2 = None
      self.y_C2 = None
      self.z_C2 = None

      # Position of the contact references frame:
      self.position_C1 = None
      self.position_C2 = None

      # Pose of the end-effector reference expressed as a 4x4 matrix (element of SE(3)):
      # Note: The pose of the contact reference frames is expressed with respect to the object reference frame. 
      self.sampled_pose_c1 = None
      self.sampled_pose_c2 = None
      self.sampled_contacts_c1 = None
      self.sampled_contacts_c2 = None

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

      # These new attributes have only been included to evaluate end-effector pose sampling strategies and heuristics:
      self.unit_u1 = None
      self.unit_u2 = None
      self.unit_u3 = None
      self.unit_ulist = None
      self.unit_nlist = None

      # Additional attributes introduced for the purpose of conducting experiments:
      self.approach_dir_2_poses = None
      self.approach_dir_2_inter_poses = None
      self.approach_dir_2_poses_base = None
      self.approach_dir_2_inter_poses_base = None
      self.approach_dir_other_poses = None
      self.approach_dir_other_inter_poses = None
      self.approach_dir_other_poses_base = None
      self.approach_dir_other_inter_poses_base = None

   '''Function to process the point clouds based on the normal information.
   Input: Downsampled Point Cloud Object
   Output: Point Cloud Object after removing the points corresponding to the flat surfaces/tables'''
   
   def remove_plane_surface(self):
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
      for i, normal in enumerate(self.normals):
         max_idx = np.where(abs(self.normals[i]) == np.amax(np.absolute(normal)))
         if max_idx[0] == 2 and self.points[i, 2] <= -0.01:
               point_indices.append(i)
           
      # Get the points corresponding to the normals pointing upward.
      '''Here we assume that the normals and points are ordered in the same way. That is the index of the normal is same
         as the index for the corresponding point'''
      self.points = np.delete(self.points, point_indices, axis=0)
      
      # Converting the processed points back to a Open3D PointCloud Object
      self.cloud = o3d.geometry.PointCloud()
      self.cloud.points = o3d.utility.Vector3dVector(self.points)
      self.cloud.paint_uniform_color([0, 0, 1])

   '''Function to preprocess the point cloud based on the clusters.
   Input: Processed Point Cloud after removing the points corresponding to the falt surfaces like tables.
   Output: (1)Point Cloud Object corresponding to the only the object in the scene which has to be manipulated.
           (2)Processed Point Cloud with Clusters.
   '''
   def get_object_point_cloud(self):
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

      # Converting the processed points back to a Open3D PointCloud Object
      object_points = np.asarray([points[i] for i, label in enumerate(labels) if label == common_cluster])
      self.processed_cloud = o3d.geometry.PointCloud()
      self.processed_cloud.points = o3d.utility.Vector3dVector(object_points)
      self.processed_cloud.paint_uniform_color([0, 0, 1])

   '''Function to transform the point cloud into the base reference frame from the camera reference frame. '''
   def transform_to_base(self):
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

      '''Function to compute the axis aligned bounding box corresponding 
         to the processed point cloud using Open3D.
      '''
   def compute_aabb(self):
      # Computing the axis aligned bounding box: 
      self.aligned_bounding_box = self.processed_cloud.get_axis_aligned_bounding_box()
      
      # This is just for visualization:
      self.aligned_bounding_box.color = (1, 0, 0)
    
      # Extracting the coordinates of the vertices of the oriented bounding box:
      self.aligned_bounding_box_vertices = np.asarray(self.aligned_bounding_box.get_box_points())

      # Extracting the center of the oriented bounding box:
      self.aligned_bounding_box_center = np.asarray(self.aligned_bounding_box.get_center())

   '''Function to compute the oriented bounding box after all the points have been transferred to the 
      object base frame computed using the axis aligned bounding box:'''
   def compute_obb(self):

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
        self.oriented_bounding_box_vertices[i,:] = np.reshape(np.dot(np.transpose(np.add(np.reshape(self.oriented_bounding_box_vertices[i, :], [3,1]), self.p_bounding_box)), la.inv(self.R_object)), [1,3])

      # We need to compute the center of the oriented bounding and it cannot be the same as the
      # axis aligned bounding box. It should based upon the dimensions of the oriented bounding box.
      self.oriented_bounding_box_center = np.reshape(np.dot(np.transpose(np.add(self.oriented_bounding_box_center, self.p_bounding_box)), la.inv(self.R_object)), [3])

   '''Function to compute the oriented bounding box using the rotating calipers algorithm 
      after all the points have been transferred to the object base frame computed using the 
      axis aligned bounding box:'''
   def compute_obb_rotating_calipers(self):
      projected_points_object_frame = np.zeros([self.transformed_points_object_frame.shape[0], self.transformed_points_object_frame.shape[1]])
    
      for i, point in enumerate(projected_points_object_frame):
        projected_points_object_frame[i, 0] = self.transformed_points_object_frame[i, 0]
        projected_points_object_frame[i, 1] = self.transformed_points_object_frame[i, 1]
        projected_points_object_frame[i, 2] = 0

      projected_points_object_frame_2D = projected_points_object_frame[:, 0:2]

      '''The below code, including the Convex Hull and Rotating Caliper Functions, is added to achieve an
         optimal bounding box using the rotating calipers method, which searches all bounding boxes that
         includes one of the edges of the convex hull of a 2D point cloud. This is guaranteed to find the
         optimal bounding box with the smallest area. The first step is finding the Convex Hull Vertices,
         followed by rotating the Convex Hull so one edge is axis aligned and calculating the AABB.
         Input: 2D Point Cloud in the Object Frame -> Convex Hull Vertices
         Output: Optimal Bounding Box Corners and Center
      '''
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
        self.oriented_bounding_box_vertices[i,:] = np.reshape(np.dot(np.transpose(np.add(np.reshape(self.oriented_bounding_box_vertices[i, :], [3,1]), self.p_bounding_box)), la.inv(self.R_object)), [1,3])

      # We need to compute the center of the oriented bounding and it cannot be the same as the
      # axis aligned bounding box. It should based upon the dimensions of the oriented bounding box.
      self.oriented_bounding_box_center = np.reshape(np.dot(np.transpose(np.add(self.oriented_bounding_box_center, self.p_bounding_box)), la.inv(self.R_object)), [3])


   ''' Function to get the pose of the bounding box (axis-aligned or oriented): '''
   def get_pose_bounding_box(self):
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

   '''Function to transform the object point cloud and its bounding box to an object reference frame based on the 
   bounding box: '''
   def transform_to_object_frame(self):
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

      for i, point in enumerate(self.points):
         self.transformed_points_object_frame[i,:] = np.reshape(np.dot(np.transpose(np.subtract(np.reshape(point, [3,1]), self.p_bounding_box)), self.R_object), [1,3])

      # Transforming the vertices of the bounding box also to the object reference frame:
      self.transformed_vertices_object_frame = np.zeros([vertices.shape[0], vertices.shape[1]])

      for i, vertex in enumerate(vertices):
         self.transformed_vertices_object_frame[i,:] = np.reshape(np.dot(np.transpose(np.subtract(np.reshape(vertex, [3,1]), self.p_bounding_box)), self.R_object), [1,3])

      # Getting the dimensions of the box in terms of the X, Y and Z directions:
      self.x_dim = np.round(np.absolute(self.transformed_vertices_object_frame[0,0] - self.transformed_vertices_object_frame[1,0]),2)
      self.y_dim = np.round(np.absolute(self.transformed_vertices_object_frame[2,1] - self.transformed_vertices_object_frame[0,1]),2)
      self.z_dim = np.round(np.absolute(self.transformed_vertices_object_frame[3,2] - self.transformed_vertices_object_frame[0,2]),2)
      
      # Storing the dimensions of the bounding box in a single numpy array
      self.dimensions = np.zeros([3,1])
      self.dimensions[0] = self.x_dim
      self.dimensions[1] = self.y_dim
      self.dimensions[2] = self.z_dim

   '''Function to compute the bounding box: '''
   def compute_bounding_box(self):
       # The object is aligned with the axis of the world/robot base reference frame.
       self.bounding_box_flag = 0

       # Computing the bounding boxes corresponding to the object point cloud: 
       self.compute_aabb()
       
       # Rotation matrix and position vector for the robot base or world reference frame: 
       self.R_base = np.identity(3)
       self.p_base = np.zeros([3,1])

       # Getting the pose of the bounding box:
       self.get_pose_bounding_box()

       # Transforming the points to the object reference frame:
       self.transform_to_object_frame()
       self.compute_obb_rotating_calipers()

       # We now change the bounding box flag and use the oriented bounding box to sample the contacts:
       self.bounding_box_flag = 1

       # Getting the pose of the bounding box:
       self.get_pose_bounding_box()

       # Transforming the points to the object reference frame:
       self.transform_to_object_frame()

       # Saving the point cloud transformed to the object reference frame:
       # Creating a Open3d PointCloud Object for the cloud corresponding to just the bounding box
       self.cloud_object_frame = o3d.geometry.PointCloud()
       self.cloud_object_frame.points = o3d.utility.Vector3dVector(self.transformed_points_object_frame.astype(np.float64))
       self.cloud_object_frame.paint_uniform_color([0, 0, 1])

       self.cloud_object_frame.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))

       self.cloud_object_frame.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.], dtype="float64"))
       # Storing the normals as a numpy array: 
       self.normals_object_frame = np.asarray(self.cloud_object_frame.normals) 


   '''Function to sample contacts from the two parallel faces of the bounding box and generate the feature vector to be used as input to the
      neural network. In this function a single datapoint has dimensions 12x1 and contains plucker coordinates.'''
   def generate_contacts_yz(self):
      # Sampling contacts as input to the neural network. We will be using the transformed points and transformed 
      # vertices for sampling the antipodal contact locations. 
      self.y_axis_increments = np.arange(self.transformed_vertices_object_frame[1,1], self.transformed_vertices_object_frame[7,1], self.increment)
      self.z_axis_increments = np.arange(self.transformed_vertices_object_frame[0,2], self.transformed_vertices_object_frame[3,2], self.increment)

      # Efficiently sampling antipodal contacts:
      self.sampled_c1 = np.asarray([[self.transformed_vertices_object_frame[0,0], y, z] for z in self.z_axis_increments for y in self.y_axis_increments])
      self.sampled_c2 = np.asarray([[self.transformed_vertices_object_frame[1,0], y, z] for z in self.z_axis_increments for y in self.y_axis_increments])
      self.x_data = np.asarray([np.reshape(np.asarray([[self.transformed_vertices_object_frame[0,0], y, z], [self.transformed_vertices_object_frame[1,0], y, z], 
                                                       self.screw_axis, self.moment]), [1,12]) for z in self.z_axis_increments for y in self.y_axis_increments])
      self.x_data = np.reshape(self.x_data, [self.x_data.shape[0], self.x_data.shape[1]*self.x_data.shape[2]])

      # Generate empty data for the corresponding y labels required as input to the Pytorch DataLoader class
      self.y_data = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 1])

   '''Function to sample contacts from the two parallel faces of the bounding box and generate the feature vector to be used as input to the
      neural network. In this function a single datapoint has dimensions 15x1 and contains plucker as well as nonplucker coordinates.'''
   def generate_contacts_yz_plucker_non_plucker(self):
      # Sampling contacts as input to the neural network. We will be using the transformed points and transformed 
      # vertices for sampling the antipodal contact locations. 
      self.y_axis_increments = np.arange(self.transformed_vertices_object_frame[1,1], self.transformed_vertices_object_frame[7,1], self.increment)
      self.z_axis_increments = np.arange(self.transformed_vertices_object_frame[0,2], self.transformed_vertices_object_frame[3,2], self.increment)

      # Efficiently sampling antipodal contacts:
      self.sampled_c1 = np.asarray([[self.transformed_vertices_object_frame[0,0], y, z] for z in self.z_axis_increments for y in self.y_axis_increments])
      self.sampled_c2 = np.asarray([[self.transformed_vertices_object_frame[1,0], y, z] for z in self.z_axis_increments for y in self.y_axis_increments])
      self.x_data = np.asarray([np.reshape(np.asarray([[self.transformed_vertices_object_frame[0,0], y, z], [self.transformed_vertices_object_frame[1,0], y, z], self.screw_axis, self.moment, self.point]), [1,15]) for z in self.z_axis_increments for y in self.y_axis_increments])
      self.x_data = np.reshape(self.x_data, [self.x_data.shape[0], self.x_data.shape[1]*self.x_data.shape[2]])

      # Generate empty data for the corresponding y labels required as input to the Pytorch DataLoader class
      self.y_data = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 1])

   '''Function to sample contacts from the two parallel faces of the bounding box and generate the feature vector to be used as input to the
      neural network. In this function a single datapoint has dimensions 18x1 and contains additional features like the moment arms.'''
   def generate_contacts_yz_additional_features(self):
      # Sampling contacts as input to the neural network. We will be using the transformed points and transformed 
      # vertices for sampling the antipodal contact locations. 
      self.y_axis_increments = np.arange(self.transformed_vertices_object_frame[1,1], self.transformed_vertices_object_frame[7,1], self.increment)
      self.z_axis_increments = np.arange(self.transformed_vertices_object_frame[0,2], self.transformed_vertices_object_frame[3,2], self.increment)

      # Efficiently sampling antipodal contacts:
      self.sampled_c1 = np.asarray([[self.transformed_vertices_object_frame[0,0], y, z] for z in self.z_axis_increments for y in self.y_axis_increments])
      self.sampled_c2 = np.asarray([[self.transformed_vertices_object_frame[1,0], y, z] for z in self.z_axis_increments for y in self.y_axis_increments])
      
      self.x_data = np.zeros([self.sampled_c1.shape[0], 18])
      for i, c1 in enumerate(self.sampled_c1):
          self.x_data[i, :] = np.reshape(np.asarray([self.sampled_c1[i, 0], self.sampled_c1[i, 1], self.sampled_c1[i, 2], 
                              self.sampled_c2[i, 0], self.sampled_c2[i, 1], self.sampled_c2[i, 2], 
                              self.screw_axis[0], self.screw_axis[1], self.screw_axis[2],
                              self.moment[0], self.moment[1], self.moment[2], 
                              self.point[0], self.point[1],  self.point[2],
                              la.norm(self.sampled_c1[i, :]), la.norm(self.point), la.norm((np.subtract(self.sampled_c1[i, :], self.point)))]), [1, 18])

      # Generate empty data for the corresponding y labels required as input to the Pytorch DataLoader class
      self.y_data = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 1])

   '''Function to sample contacts from the two parallel faces of the bounding box and generate the feature vector to be used as input to the
      neural network. In this function a single datapoint has dimensions 12x1 and contains plucker coordinates.'''
   def generate_contacts_xz(self):
      # Sampling contacts as input to the neural network. We will be using the transformed points and transformed 
      # vertices for sampling the antipodal contact locations. 
      self.x_axis_increments = np.arange(self.transformed_vertices_object_frame[0,0], self.transformed_vertices_object_frame[1,0], self.increment)
      self.z_axis_increments = np.arange(self.transformed_vertices_object_frame[0,2], self.transformed_vertices_object_frame[3,2], self.increment)

      self.sampled_c1 = np.asarray([[x, self.transformed_vertices_object_frame[0,1], z] for z in self.z_axis_increments for x in self.x_axis_increments])
      self.sampled_c2 = np.asarray([[x, self.transformed_vertices_object_frame[2,1], z] for z in self.z_axis_increments for x in self.x_axis_increments])
      self.x_data = np.asarray([np.reshape(np.asarray([[x, self.transformed_vertices_object_frame[0,1], z], [x, self.transformed_vertices_object_frame[2,1], z], 
                                                       self.screw_axis, self.moment]), [1,12]) for z in self.z_axis_increments for x in self.x_axis_increments])
      self.x_data = np.reshape(self.x_data, [self.x_data.shape[0], self.x_data.shape[1]*self.x_data.shape[2]])
      
      # Generate empty data for the corresponding y labels required as input to the Pytorch DataLoader class
      self.y_data = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 1])


   '''Function to sample contacts from the two parallel faces of the bounding box and generate the feature vector to be used as input to the
      neural network. In this function a single datapoint has dimensions 15x1 and contains plucker as well as nonplucker coordinates.'''
   def generate_contacts_xz_plucker_non_plucker(self):
      # Sampling contacts as input to the neural network. We will be using the transformed points and transformed 
      # vertices for sampling the antipodal contact locations. 
      self.x_axis_increments = np.arange(self.transformed_vertices_object_frame[0,0], self.transformed_vertices_object_frame[1,0], self.increment)
      self.z_axis_increments = np.arange(self.transformed_vertices_object_frame[0,2], self.transformed_vertices_object_frame[3,2], self.increment)

      self.sampled_c1 = np.asarray([[x, self.transformed_vertices_object_frame[0,1], z] for z in self.z_axis_increments for x in self.x_axis_increments])
      self.sampled_c2 = np.asarray([[x, self.transformed_vertices_object_frame[2,1], z] for z in self.z_axis_increments for x in self.x_axis_increments])
      self.x_data = np.asarray([np.reshape(np.asarray([[x, self.transformed_vertices_object_frame[0,1], z], [x, self.transformed_vertices_object_frame[2,1], z], self.screw_axis, self.moment, self.point]), [1,15]) for z in self.z_axis_increments for x in self.x_axis_increments])
      self.x_data = np.reshape(self.x_data, [self.x_data.shape[0], self.x_data.shape[1]*self.x_data.shape[2]])

      # Generate empty data for the corresponding y labels required as input to the Pytorch DataLoader class
      self.y_data = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 1])

   '''Function to sample contacts from the two parallel faces of the bounding box and generate the feature vector to be used as input to the
      neural network. In this function a single datapoint has dimensions 18x1 and contains additional features like the moment arms.'''
   def generate_contacts_xz_additional_features(self):
      # Sampling contacts as input to the neural network. We will be using the transformed points and transformed 
      # vertices for sampling the antipodal contact locations. 
      self.x_axis_increments = np.arange(self.transformed_vertices_object_frame[0,0], self.transformed_vertices_object_frame[1,0], self.increment)
      self.z_axis_increments = np.arange(self.transformed_vertices_object_frame[0,2], self.transformed_vertices_object_frame[3,2], self.increment)

      self.sampled_c1 = np.asarray([[x, self.transformed_vertices_object_frame[0,1], z] for z in self.z_axis_increments for x in self.x_axis_increments])
      self.sampled_c2 = np.asarray([[x, self.transformed_vertices_object_frame[2,1], z] for z in self.z_axis_increments for x in self.x_axis_increments])

      self.x_data = np.zeros([self.sampled_c1.shape[0], 18])
      for i, c1 in enumerate(self.sampled_c1):
          self.x_data[i, :] = np.reshape(np.asarray([self.sampled_c1[i, 0], self.sampled_c1[i, 1], self.sampled_c1[i, 2], 
                              self.sampled_c2[i, 0], self.sampled_c2[i, 1], self.sampled_c2[i, 2], 
                              self.screw_axis[0], self.screw_axis[1], self.screw_axis[2],
                              self.moment[0], self.moment[1], self.moment[2], 
                              self.point[0], self.point[1],  self.point[2],
                              la.norm(self.sampled_c1[i, :]), la.norm(self.point), la.norm((np.subtract(self.sampled_c1[i, :], self.point)))]), [1, 18])

      # Generate empty data for the corresponding y labels required as input to the Pytorch DataLoader class
      self.y_data = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 1])

   '''Function to generate contacts depending on the gripper width and dimensions of the bounding box: '''
   def generate_contacts(self):
       # Define the increment:
       self.increment = 0.01
       if self.y_dim < self.gripper_width_tolerance:
           print('Generating contacts along XZ plane')
           # self.generate_contacts_xz()
           # self.generate_contacts_xz_plucker_non_plucker()
           self.generate_contacts_xz_additional_features()
       elif self.x_dim < self.gripper_width_tolerance:
           print('Generating contacts along YZ plane')
           # self.generate_contacts_yz()
           # self.generate_contacts_yz_plucker_non_plucker()
           self.generate_contacts_yz_additional_features()
       elif self.x_dim < self.gripper_width_tolerance and self.y_dim < self.gripper_width_tolerance:
           print('Both dimensions with gripper width tolerance. Generating contacts along XZ plane')
       else:
           print('Invalid Data')

   '''This function is used to predict the metric values using the datapoints as input'''
   def predict_metric(self):
      # OLDER NEURAL NETWORK
      # NEURAL NETWORK BASED METRIC PREDICTION: 
      # HYPER PARAMETERS: 
      # (1) Network Size:
      self.input_size = 12
      self.hidden_size1 = 8
      self.hidden_size2 = 3

      # Batch size to divide the dataset into batches:
      self.batch_size = 2

      # Defining the neural network model
      model = metric_nn(self.input_size, self.hidden_size1, self.hidden_size2)

      # Loading the trained models: 
      # PATH = 'Trained_Models/model8_150epochs_lr001.pth'
      PATH = 'Trained_Models/model6_100epochs_lr001.pth'
      # PATH = 'Trained_Models/model8_150epochs_lr001_1.pth'
      # PATH = 'Trained_Models/model1_100epochs_lr0001.pth'

      model.load_state_dict(torch.load(PATH))

      testing_dataset = metric_nn_dataset(self.x_data, self.y_data, transform = transforms.Compose([to_tensor()]))

      # Using the DataLoader class from Pytorch to shuffle the data and divide it into batches:
      testLoader = DataLoader(testing_dataset, self.batch_size, 
                                 shuffle=True, num_workers=0)

      print('Weights Loaded!')
      # TESTING LOOP:
      output_predicted = []
      output_ground_truth = []
      output_test_datapoints = []
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
      for i, output in enumerate(output_test_datapoints):
         if i != len(output_test_datapoints)-1:
               for j,v in enumerate(output_test_datapoints[0]):
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
      # Normalizing the values between 0 and 1:
      self.predicted = (self.predicted - np.min(self.predicted))/(np.max(self.predicted - np.min(self.predicted)))

   '''This function is used to predict the metric values using the datapoints as input'''
   def predict_metric_generic(self):
      # NEWLY TRAINED NEURAL NETWORK
      # NEURAL NETWORK BASED METRIC PREDICTION: 
      # HYPER PARAMETERS: 
      # (1) Network Size:

      # Batch size to divide the dataset into batches:
      # Hyperparameters that are part of the pointCloud class
      self.batch_size = 400
      # elf.input_size = 12
      # self.input_size = 15
      self.input_size = 18

      # Specifying the seed:
      seed = 3
      torch.manual_seed(seed)

      # Specifying the depth of the neural network:
      depth = 8

      # Batch norm:
      norm = nn.BatchNorm1d

      # Activation function:
      act = nn.ReLU

      # Defining the neural network model
      model = metric_nn_generic(self.input_size, depth=depth, residual=True, norm=norm, act_layer=act) 
      
      ## Dataset: Variation 1
      # Plucker:
      # best_weights = 'depth_8_norm_batch_act_relu_residual_True_input_12_test_all_train_variation_1_plucker_extra_False.pth'
      
      # Non-Plucker

      # Additional Features:
      best_weights = 'depth_8_norm_batch_act_relu_residual_True_input_18_test_all_train_variation_1_additional_features_extra_True.pth'

      ## Dataset: Variation 2
      # Plucker:
      #

      # Non-Plucker:
      # best_weights = 'depth_8_norm_batch_act_relu_residual_True_input_12_test_all_train_variation_2_plucker_extra_False.pth'

      # Additional Features:
      # best_weights = 'depth_8_norm_batch_act_relu_residual_True_input_18_test_all_train_variation_2_additional_features_extra_True.pth'

      # Loading the trained models: 
      PATH = 'Trained_Models/' + best_weights

      model.load_state_dict(torch.load(PATH, map_location = 'cpu'))

      testing_dataset = metric_nn_dataset(self.x_data, self.y_data, transform = transforms.Compose([to_tensor()]))

      # Using the DataLoader class from Pytorch to shuffle the data and divide it into batches:
      testLoader = DataLoader(testing_dataset, self.batch_size, 
                                 shuffle=True, num_workers=0)

      print('Weights Loaded!')
      # TESTING LOOP:
      output_predicted = []
      output_ground_truth = []
      output_test_datapoints = []
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
      for i, output in enumerate(output_test_datapoints):
         if i != len(output_test_datapoints)-1:
               for j,v in enumerate(output_test_datapoints[0]):
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
      # Normalizing the values between 0 and 1:
      self.predicted = (self.predicted - np.min(self.predicted))/(np.max(self.predicted - np.min(self.predicted)))

   '''# Function to project the points onto a one of the surfaces of the bounding box:'''
   def project_points_yz(self):
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

      for i, point in enumerate(self.transformed_points_object_frame):
        distance = np.divide(la.norm(unit_n[0]*point[0] + unit_n[1]*point[1] + unit_n[2]*point[2] - D), np.sqrt(unit_n[0]**2 + unit_n[1]**2 + unit_n[2]**2))
        self.projected_points[i, :] = np.add(point, np.dot(distance, unit_n))

   '''# Function to project the points onto a one of the surfaces of the bounding box:'''
   def project_points_xz(self):
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

      for i, point in enumerate(self.transformed_points_object_frame):
        distance = np.divide(la.norm(unit_n[0]*point[0] + unit_n[1]*point[1] + unit_n[2]*point[2] - D), np.sqrt(unit_n[0]**2 + unit_n[1]**2 + unit_n[2]**2))
        self.projected_points[i, :] = np.add(point, np.dot(distance, unit_n))
   
   ''' Function to generate a grid on the surface of the bounding box based on the computed metric values:''' 
   def generate_grid_yz(self):
      # GRID GENERATION: 
      '''Now we generate the grid using the multidimensional arrays 'x_data' and 'metric_values'
         Since we are sampling contact locations along the Y and Z axis of the object lcoal reference frame: '''
      self.y_counter = 0
      self.z_counter = 0
      counter = 0
      
      # Create a dictionary of test_datapoints and predicted metric values. 
      '''This is being done to access the predicted metric values so that it can efficiently used for grid generation'''
      dictionary_test_data = {}
      for test_point, label in zip(self.test_datapoints, self.predicted):
          test_point = np.around(test_point, 3)
          dictionary_test_data[tuple([test_point[0].item(), test_point[1].item(), test_point[2].item(), test_point[3].item(), 
                                      test_point[4].item(), test_point[5].item(),test_point[6].item(), test_point[7].item(), 
                                      test_point[8].item(), test_point[9].item(), test_point[10].item(), test_point[11].item()])] = label.item()
          
      # Query the dictionary for x_data to get the corresponding metric_values: 
      self.metric_values = np.zeros([self.predicted.shape[0], self.predicted.shape[1]])
      for i, x in enumerate(self.x_data):
          x = np.around(x, 3)    
          self.metric_values[i, :] = np.asarray(dictionary_test_data[tuple([x[0].item(), x[1].item(), x[2].item(), x[3].item(), 
                                      x[4].item(), x[5].item(), x[6].item(), x[7].item(), 
                                      x[8].item(), x[9].item(), x[10].item(), x[11].item()])])

      self.grid_points = np.around(np.asarray([np.reshape(np.asarray([y, z]), [2,1]) for z in self.z_axis_increments for y in self.y_axis_increments]), 3)
      
      # Create a dictionary of grid points and metric_values 
      dictionary_grid_points = {}
      for grid_point, metric_value in zip(self.grid_points, self.metric_values):
          dictionary_grid_points[tuple([grid_point[0].item(), grid_point[1].item()])] = metric_value.item()

      # Initializing the empty arrays:
      self.grid_metric_values = np.zeros([len(self.y_axis_increments), len(self.z_axis_increments), 1])
      self.grid_centers_matrix = np.zeros([len(self.y_axis_increments), len(self.z_axis_increments), 2])
      self.Y_grid_points_matrix = np.zeros([len(self.y_axis_increments), len(self.z_axis_increments), 4])
      self.Z_grid_points_matrix = np.zeros([len(self.y_axis_increments), len(self.z_axis_increments), 4])

      self.grid_centers = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 2])
      self.Y_grid_points = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 4])
      self.Z_grid_points = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 4])

      self.metric_grid = np.zeros([len(self.y_axis_increments)*len(self.z_axis_increments), 1])
      self.X_grid_points = self.transformed_vertices_object_frame[1,0]*np.ones([self.Y_grid_points.shape[0], self.Y_grid_points.shape[1]])

      # Resetting the counter value back to 0 so that it doesn't cause any further errors.
      counter = 0

      print('Generating Grid ... ')
      for i, z in enumerate(self.z_axis_increments):
         # Termination criteria for the Z Axis:
         if self.z_axis_increments[i] == self.z_axis_increments[len(self.z_axis_increments)-1]:
               break
         else:
               self.y_counter = 0
               for j, y in enumerate(self.y_axis_increments):
                  if self.y_axis_increments[j] == self.y_axis_increments[len(self.y_axis_increments)-1]:
                     break
                  else:
                     # For the grid the points are arranged in an anticlockwise order:
                     p_1 = np.reshape(np.around(np.asarray([self.y_axis_increments[j], self.z_axis_increments[i]]), 3), [2,1])
                     p_2 = np.reshape(np.around(np.asarray([self.y_axis_increments[j+1], self.z_axis_increments[i]]), 3), [2,1])
                     p_3 = np.reshape(np.around(np.asarray([self.y_axis_increments[j+1], self.z_axis_increments[i+1]]), 3), [2,1])
                     p_4 = np.reshape(np.around(np.asarray([self.y_axis_increments[j], self.z_axis_increments[i+1]]), 3), [2,1])

                     # Get the corresponding metric values:
                     eta_1 = np.asarray(dictionary_grid_points[tuple([p_1[0].item(), p_1[1].item()])])
                     eta_2 = np.asarray(dictionary_grid_points[tuple([p_2[0].item(), p_2[1].item()])])
                     eta_3 = np.asarray(dictionary_grid_points[tuple([p_3[0].item(), p_3[1].item()])])
                     eta_4 = np.asarray(dictionary_grid_points[tuple([p_4[0].item(), p_4[1].item()])])

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
                     self.grid_centers_matrix[self.y_counter, self.z_counter, :] = np.reshape(np.asarray([self.y_axis_increments[j] + (self.increment/2), self.z_axis_increments[i] + (self.increment/2)]), [2])
                     
                     self.Y_grid_points[counter, :] = np.reshape(np.asarray([self.y_axis_increments[j], self.y_axis_increments[j] + self.increment, self.y_axis_increments[j] + self.increment, self.y_axis_increments[j]]), [4])
                     self.Z_grid_points[counter, :] = np.reshape(np.asarray([self.z_axis_increments[i], self.z_axis_increments[i], self.z_axis_increments[i] + self.increment, self.z_axis_increments[i] + self.increment]), [4])
                     self.grid_centers[counter, :] = np.reshape(np.asarray([self.y_axis_increments[j] + (self.increment/2), self.z_axis_increments[i] + (self.increment/2)]), [2])

                     # Updating the two inner loop counters.
                     self.y_counter += 1
                     counter += 1

               # Updating outer loop counter.
               self.z_counter += 1

   ''' Function to generate a grid on the surface of the bounding box based on the computed metric values: ''' 
   def generate_grid_xz(self):
      # GRID GENERATION: 
      '''Now we generate the grid using the multidimensional arrays 'x_data' and 'metric_values'
         Since we are sampling contact locations along the Y and Z axis, we also need to do something similar to 
         X and Z axis of the object lcoal reference frame: '''
      self.x_counter = 0
      self.z_counter = 0
      counter = 0

      # Create a dictionary of test_datapoints and predicted metric values.
      '''This is being done to access the predicted metric values so that it can efficiently used for grid generation'''
      dictionary_test_data = {}
      for test_point, label in zip(self.test_datapoints, self.predicted):
          test_point = np.around(test_point, 3)
          dictionary_test_data[tuple([test_point[0].item(), test_point[1].item(), test_point[2].item(), test_point[3].item(), 
                                      test_point[4].item(), test_point[5].item(),test_point[6].item(), test_point[7].item(), 
                                      test_point[8].item(), test_point[9].item(), test_point[10].item(), test_point[11].item()])] = label.item()
          
      # Query the dictionary for x_data to get the corresponding metric_values: 
      self.metric_values = np.zeros([self.predicted.shape[0], self.predicted.shape[1]])
      for i, x in enumerate(self.x_data):
          x = np.around(x, 3)    
          self.metric_values[i, :] = np.asarray(dictionary_test_data[tuple([x[0].item(), x[1].item(), x[2].item(), x[3].item(), 
                                      x[4].item(), x[5].item(), x[6].item(), x[7].item(), 
                                      x[8].item(), x[9].item(), x[10].item(), x[11].item()])])
      
      self.grid_points = np.around(np.asarray([np.reshape(np.asarray([x, z]), [2,1]) for z in self.z_axis_increments for x in self.x_axis_increments]), 3)
      # Create a dictionary of grid points and metric_values 
      dictionary_grid_points = {}
      for grid_point, metric_value in zip(self.grid_points, self.metric_values):
          dictionary_grid_points[tuple([grid_point[0].item(), grid_point[1].item()])] = metric_value.item()

      # Initializing the empty arrays:
      self.grid_metric_values = np.zeros([len(self.x_axis_increments), len(self.z_axis_increments), 1])
      self.grid_centers_matrix = np.zeros([len(self.x_axis_increments), len(self.z_axis_increments), 2])
      self.X_grid_points_matrix = np.zeros([len(self.x_axis_increments), len(self.z_axis_increments), 4])
      self.Z_grid_points_matrix = np.zeros([len(self.x_axis_increments), len(self.z_axis_increments), 4])

      self.grid_centers = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 2])
      self.X_grid_points = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 4])
      self.Z_grid_points = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 4])

      self.metric_grid = np.zeros([len(self.x_axis_increments)*len(self.z_axis_increments), 1])
      self.Y_grid_points = self.transformed_vertices_object_frame[1,1]*np.ones([self.X_grid_points.shape[0], self.X_grid_points.shape[1]])

      # Resetting the counter value back to 0 so that it doesn't cause any further errors.
      counter = 0

      print('Generating Grid ... ')
      for i, z in enumerate(self.z_axis_increments):
         # Termination criteria for the Z Axis:
         if self.z_axis_increments[i] == self.z_axis_increments[len(self.z_axis_increments)-1]:
               break
         else:
               self.x_counter = 0
               for j, x in enumerate(self.x_axis_increments):
                  if self.x_axis_increments[j] == self.x_axis_increments[len(self.x_axis_increments)-1]:
                     break
                  else:
                     # For the grid the points are arranged in an anticlockwise order:
                     p_1 = np.reshape(np.around(np.asarray([self.x_axis_increments[j], self.z_axis_increments[i]]), 3), [2,1])
                     p_2 = np.reshape(np.around(np.asarray([self.x_axis_increments[j+1], self.z_axis_increments[i]]), 3), [2,1])
                     p_3 = np.reshape(np.around(np.asarray([self.x_axis_increments[j+1], self.z_axis_increments[i+1]]), 3), [2,1])
                     p_4 = np.reshape(np.around(np.asarray([self.x_axis_increments[j], self.z_axis_increments[i+1]]), 3), [2,1])

                    # Get the corresponding metric values:
                     eta_1 = np.asarray(dictionary_grid_points[tuple([p_1[0].item(), p_1[1].item()])])
                     eta_2 = np.asarray(dictionary_grid_points[tuple([p_2[0].item(), p_2[1].item()])])
                     eta_3 = np.asarray(dictionary_grid_points[tuple([p_3[0].item(), p_3[1].item()])])
                     eta_4 = np.asarray(dictionary_grid_points[tuple([p_4[0].item(), p_4[1].item()])])

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
                     self.grid_centers_matrix[self.x_counter, self.z_counter, :] = np.reshape(np.asarray([self.x_axis_increments[j] + (self.increment/2), self.z_axis_increments[i] + (self.increment/2)]), [2])
                     self.X_grid_points[counter, :] = np.reshape(np.asarray([self.x_axis_increments[j], self.x_axis_increments[j] + self.increment, self.x_axis_increments[j] + self.increment, self.x_axis_increments[j]]), [4])
                     self.Z_grid_points[counter, :] = np.reshape(np.asarray([self.z_axis_increments[i], self.z_axis_increments[i], self.z_axis_increments[i] + self.increment, self.z_axis_increments[i] + self.increment]), [4])
                     self.grid_centers[counter, :] = np.reshape(np.asarray([self.x_axis_increments[j] + (self.increment/2), self.z_axis_increments[i] + (self.increment/2)]), [2])
                     
                     # Updating the two inner loop counters.
                     self.x_counter += 1
                     counter += 1

               # Updating outer loop counter.
               self.z_counter += 1

   ''' Function to check the occupancy of the the point corresponding to the point cloud within the generate grid:'''
   def check_occupancy_yz(self):
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
         corner'''

      self.projected_points_local_frame = np.asarray([np.subtract(p, self.p_local) for p in self.projected_points])

      # OCCUPANCY CHECK:
      '''The main reason for transforming the points to a local reference frame in the corner of the bounding box is 
         so that we can perform a 2D occupancy check over the generated grid. This will allow us to only have the grids 
         and associated grid points which have points belonging to the point cloud within their 2D bounds.'''

      # IMPORTANT: The FOR loop will be slightly different for the XZ and YZ planes.
      print('Checking Occupancy ... ')

      self.X_grid_points_occupied = self.transformed_vertices_object_frame[1,0]*np.ones([self.projected_points.shape[0],self.X_grid_points.shape[1]])
      self.Y_grid_points_occupied = np.zeros([self.projected_points.shape[0], self.Y_grid_points.shape[1]])
      self.Z_grid_points_occupied = np.zeros([self.projected_points.shape[0], self.Z_grid_points.shape[1]])
      self.grid_centers_occupied = np.zeros([self.projected_points.shape[0], self.grid_centers.shape[1]])
      self.grid_metric_values_occupied = np.zeros([self.projected_points.shape[0], 1])
      self.grid_centers_dict = {}
      self.grid_centers_unique_dict = {}

      # Arrays to store the q_y and q_z values so that they can be studied and understood properly:
      self.q_y_array = np.zeros([self.projected_points.shape[0], 1])
      self.q_z_array = np.zeros([self.projected_points.shape[0], 1])

      for i, point in enumerate(self.projected_points_local_frame):
         q_y = np.around(np.divide(point[1], self.increment))
         q_z = np.around(np.divide(point[2], self.increment))
         q_y_actual = np.divide(point[1], self.increment)
         q_z_actual = np.divide(point[2], self.increment)
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
               self.q_y_array[i, :] = q_y
               self.q_z_array[i, :] = q_z
               self.Y_grid_points_occupied[i, :] = np.reshape(self.Y_grid_points_matrix[q_y, q_z, :], [1,4])
               self.Z_grid_points_occupied[i, :] = np.reshape(self.Z_grid_points_matrix[q_y, q_z, :], [1,4])
               self.grid_centers_occupied[i, :] = np.reshape(self.grid_centers_matrix[q_y, q_z, :], [1,2])
               self.grid_metric_values_occupied[i, :] = np.reshape(self.grid_metric_values[q_y, q_z, :], [1])
               self.grid_centers_dict[tuple([self.grid_centers_occupied[i, 0].item(), self.grid_centers_occupied[i, 1].item()])] = self.grid_metric_values_occupied[i, :].item()
         else:
            continue
      
      # Extracting the unique grid centers and corresponding metric values and storing them in a dictionary.
      self.grid_centers_unique = np.unique(self.grid_centers_occupied, axis = 0)
      for i, grid_center in enumerate(self.grid_centers_unique):
         self.grid_centers_unique_dict[tuple([grid_center[0].item(), grid_center[1].item()])] = self.grid_centers_dict[tuple([grid_center[0].item(), grid_center[1].item()])]


   ''' Function to check the occupancy of the the point corresponding to the point cloud within the generate grid:'''
   def check_occupancy_xz(self):
      # TRANSFORMING THE POINTS WITH RESPECT TO THE REFERENCE FRAME ATTACHED AT THE CORNER OF THE BOUNDING BOX:
      ''' This will be different for sampling points along the XZ axes as compared to the YZ axes, but it will be same for different instances
         of both. '''

      # Rotation matrix of the new local reference frame with respect to the object base reference frame:
      self.R_local = self.R_bounding_box

      # Position vector of the new local reference frame with respec to the object base reference frame:
      self.p_local = self.transformed_vertices_object_frame[0,:]

      '''Transforming the projected points from the object reference frame {O} to the local reference {L}: 
         This computation is important and may not always be valid for all instances of sampling along XZ and YZ planes.
         In this case we are essentially shifting the reference frame to the
         corner:'''
      
      self.projected_points_local_frame = np.asarray([np.subtract(p, self.p_local) for p in self.projected_points])
      # OCCUPANCY CHECK:
      '''The main reason for transforming the points to a local reference frame in the corner of the bounding box is 
         so that we can perform a 2D occupancy check over the generated grid. This will allow us to only have the grids 
         and associated grid points which have points belonging to the point cloud within their 2D bounds.'''

      # IMPORTANT: The FOR loop will be slightly different for the XZ and YZ planes.
      print('Checking Occupancy ... ')

      self.Y_grid_points_occupied = self.transformed_vertices_object_frame[1,1]*np.ones([self.projected_points.shape[0],self.Y_grid_points.shape[1]])
      self.X_grid_points_occupied = np.zeros([self.projected_points.shape[0], self.X_grid_points.shape[1]])
      self.Z_grid_points_occupied = np.zeros([self.projected_points.shape[0], self.Z_grid_points.shape[1]])
      self.grid_centers_occupied = np.zeros([self.projected_points.shape[0], self.grid_centers.shape[1]])
      self.grid_metric_values_occupied = np.zeros([self.projected_points.shape[0], 1])
      self.grid_centers_dict = {}
      self.grid_centers_unique_dict = {}

      # Arrays to store the q_y and q_z values so that they can be studied and understood properly:
      self.q_x_array = np.zeros([self.projected_points.shape[0], 1])
      self.q_z_array = np.zeros([self.projected_points.shape[0], 1])

      # 2D Occupancy Check using projected 
      for i, point in enumerate(self.projected_points_local_frame):
         q_x = np.around(np.divide(point[0], self.increment))
         q_z = np.around(np.divide(point[2], self.increment))
         q_x_actual = np.divide(point[0], self.increment)
         q_z_actual = np.divide(point[2], self.increment)
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
               self.q_x_array[i, :] = q_x
               self.q_z_array[i, :] = q_z
               self.X_grid_points_occupied[i, :] = np.reshape(self.X_grid_points_matrix[q_x, q_z, :], [1,4])
               self.Z_grid_points_occupied[i, :] = np.reshape(self.Z_grid_points_matrix[q_x, q_z, :], [1,4])
               self.grid_centers_occupied[i, :] = np.reshape(self.grid_centers_matrix[q_x, q_z, :], [1,2])
               self.grid_metric_values_occupied[i, :] = np.reshape(self.grid_metric_values[q_x, q_z, :], [1]) 
               self.grid_centers_dict[tuple([self.grid_centers_occupied[i, 0].item(), self.grid_centers_occupied[i, 1].item()])] = self.grid_metric_values_occupied[i, :].item()      
         else:
            continue
      
      # Extracting the unique grid centers and corresponding metric values and storing them in a dictionary.
      self.grid_centers_unique = np.unique(self.grid_centers_occupied, axis = 0)
      for i, grid_center in enumerate(self.grid_centers_unique):
         self.grid_centers_unique_dict[tuple([grid_center[0].item(), grid_center[1].item()])] = self.grid_centers_dict[tuple([grid_center[0].item(), grid_center[1].item()])]

   '''Function to compute the distance from a point to a plane'''
   def get_distance(self, plane_points, center_point, grasp_center):
      # We need to compute the distance of the grasp center to the first plane only:
      vect_1 = np.subtract(plane_points[3, :], plane_points[0, :])
      vect_2 = np.subtract(plane_points[1, :], plane_points[0, :])
      # vect_2 = np.subtract(plane_points[1, :], plane_points[2, :])

      # Normal vector corresponding to the plane:
      unit_u = np.divide(np.cross(vect_1, vect_2), la.norm(np.cross(vect_1, vect_2)))

      # Translating each of the points by a specified distance: 
      # Computing the D in the equation of the plane Ax + By + Cz + D = 0:
      D = np.dot(center_point, unit_u)

      distance = np.divide(la.norm(unit_u[0]*grasp_center[0] + unit_u[1]*grasp_center[1] + unit_u[2]*grasp_center[2] - D), np.sqrt(unit_u[0]**2 + unit_u[1]**2 + unit_u[2]**2))
      return distance, unit_u
   
   '''Function to extract and store the points corresponding to the ideal grasping region: '''
   def get_ideal_grasping_region(self):
        
        if self.y_dim < self.gripper_width_tolerance:
            self.project_points_xz()
            print('Points projected on the surface now generating grid ...')
            self.generate_grid_xz()
            self.check_occupancy_xz()
            print('Occupancy check completed, proceed towards sampling poses ... ')
        elif self.x_dim < self.gripper_width_tolerance:
            self.project_points_yz()
            print('Points projected on the surface now generating grid ...')
            self.generate_grid_yz()
            self.check_occupancy_yz()
            print('Occupancy check completed, proceed towards sampling poses ... ')
        elif self.x_dim < self.gripper_width_tolerance and self.y_dim < self.gripper_width_tolerance:
            print('Both dimensions with gripper width tolerance. Generating contacts along XZ plane')
            self.project_points_xz()
            print('Points projected on the surface now generating grid ...')
            self.generate_grid_xz()
            self.check_occupancy_xz()
            print('Occupancy check completed, proceed towards sampling poses ... ')
        else:
            print('Invalid Data')

        self.grid_metric_values_occupied = self.grid_metric_values_occupied.flatten()
        self.max_metric_value = max(self.grid_metric_values_occupied)

        # The metric threshold value is set to be 90% of the max metric value. All the points with a metric value higher than the threshold 
        # value are part of the ideal grasping region. 
        self.eta_threshold = 0.85*self.max_metric_value
        # self.eta_threshold = 0.7*self.max_metric_value
        # self.eta_threshold = self.max_metric_value
      
        self.ideal_grasping_region_metric_values = np.asarray([self.grid_metric_values_occupied[i] for i in range(0, self.grid_metric_values_occupied.shape[0]) if self.grid_metric_values_occupied[i] >= self.eta_threshold])
        self.ideal_grasping_region_indices = [i for i in range(0, self.grid_metric_values_occupied.shape[0]) if self.grid_metric_values_occupied[i] >= self.eta_threshold]
        self.ideal_grasping_region_grid_centers = [gc for gc in self.grid_centers_unique if self.grid_centers_unique_dict[tuple([gc[0].item(), gc[1].item(),])] >= self.eta_threshold]
                
        self.ideal_grasping_region_points = self.transformed_points_object_frame[self.ideal_grasping_region_indices, :]
        self.ideal_grasping_region_normals = self.normals_object_frame[self.ideal_grasping_region_indices, :]
       
   '''Function to compute the bounding box the points corresponding to the ideal grasping region: '''
   def get_bb_ideal_grasping_region(self):
      # The o3d.geometry.PointCloud() object associated with the ideal grasping region is assigned to self.ideal_grasping_region_object_frame.
      # Computing the axis aligned bounding box: 
      self.ideal_grasping_region_bounding_box = self.ideal_grasping_region_object_frame.get_axis_aligned_bounding_box()
      
      # This is just for visualization:
      self.ideal_grasping_region_bounding_box.color = (1, 0, 0)
    
      # Extracting the coordinates of the vertices of the oriented bounding box:
      self.ideal_grasping_region_bounding_box_vertices = np.asarray(self.ideal_grasping_region_bounding_box.get_box_points())

      # Extracting the center of the oriented bounding box:
      self.ideal_grasping_region_bounding_box_center = np.asarray(self.ideal_grasping_region_bounding_box.get_center())

      # Using the dimensions of the newer bounding box of the ideal grasping region:
      
   '''Function to COMPUTE end effector poses based on the predicted metric values.'''
   # This function needs to be modified and updated: 
   def get_end_effector_poses(self):

      # Saving the sampled end effector poses:
      self.computed_end_effector_poses = []
      self.computed_end_effector_poses_inter = []
      self.sampled_contacts_c1 = []
      self.sampled_contacts_c2 = []
      self.grasp_centers = []
      self.normals_c1 = []
      self.normals_c2 = []
      self.unit_ulist = []

      # Additional lists for experimental purposes:
      self.approach_dir_2_poses = []
      self.approach_dir_2_inter_poses = []
      self.approach_dir_other_poses = []
      self.approach_dir_other_inter_poses = []

      for i,v in enumerate(self.ideal_grasping_region_grid_centers):
          
         # Outer conditional statement to check whether the dimensions along which we are grasping are less than the gripper width tolerance:
         if self.y_dim < self.gripper_width_tolerance:
            # Approach Direction 2:
            self.plane_points_1 = np.asarray([self.transformed_vertices_object_frame[3], self.transformed_vertices_object_frame[6], self.transformed_vertices_object_frame[4], self.transformed_vertices_object_frame[5]])
            # Approach direction 3:
            self.plane_points_2 = np.asarray([self.transformed_vertices_object_frame[3], self.transformed_vertices_object_frame[5], self.transformed_vertices_object_frame[2], self.transformed_vertices_object_frame[0]])
            # Approach direction 5:
            self.plane_points_3 = np.asarray([self.transformed_vertices_object_frame[6], self.transformed_vertices_object_frame[1], self.transformed_vertices_object_frame[7], self.transformed_vertices_object_frame[4]])
            
            # Center points of each of the planes corresponding to the three approach directions:
            self.center_point_plane_1 = np.asarray([self.p_base[0], self.p_base[1], self.plane_points_1[0, 2]])
            self.center_point_plane_2 = np.asarray([self.plane_points_2[0, 0], self.p_base[1], self.p_base[2]])
            self.center_point_plane_3 = np.asarray([self.plane_points_3[0, 0], self.p_base[1], self.p_base[2]])

            # Orientation of the contact reference frames: 
            # Getting an orthogonal reference frame where the z axis is along the normal: 
            # Frame C1:
            self.z_C1 = np.asarray([0, 1, 0])
            self.x_C1 = np.random.randn(3)
            self.x_C1 -= self.x_C1.dot(self.z_C1)*self.z_C1
            self.x_C1 /= la.norm(self.x_C1)
            self.y_C1 = np.cross(self.z_C1, self.x_C1)
            R_C1 = np.zeros([3,3])
            R_C1[:, 0] = self.x_C1
            R_C1[:, 1] = self.y_C1
            R_C1[:, 2] = self.z_C1

            # Frame C2:
            self.z_C2 = np.asarray([0, -1, 0])
            self.x_C2 = np.random.randn(3)
            self.x_C2 -= self.x_C2.dot(self.z_C2)*self.z_C2
            self.x_C2 /= la.norm(self.x_C2)
            self.y_C2 = np.cross(self.z_C2, self.x_C2)
            R_C2 = np.zeros([3,3])
            R_C2[:, 0] = self.x_C2
            R_C2[:, 1] = self.y_C2
            R_C2[:, 2] = self.z_C2

            self.position_C1 = np.asarray([self.ideal_grasping_region_grid_centers[i][0], self.transformed_vertices_object_frame[0,1], self.ideal_grasping_region_grid_centers[i][1]])
            self.position_C2 = np.asarray([self.ideal_grasping_region_grid_centers[i][0], self.transformed_vertices_object_frame[2,1], self.ideal_grasping_region_grid_centers[i][1]])
            self.grasp_center = np.add(self.position_C1, np.dot((self.y_dim)/2, self.z_C1)) 

            self.distance_1, self.unit_u1 = self.get_distance(self.plane_points_1, self.center_point_plane_1, self.grasp_center)
            self.distance_2, self.unit_u2 = self.get_distance(self.plane_points_2, self.center_point_plane_2, self.grasp_center)
            self.distance_3, self.unit_u3 = self.get_distance(self.plane_points_3, self.center_point_plane_3, self.grasp_center)

            # Checking the second approach direction:
            if self.distance_1 < self.gripper_height_tolerance:
               # End Effector Orientation Based on the Franka Panda Convention: 
               '''self.z_EE = -1*self.unit_u1
               self.y_EE = self.z_C1
               self.x_EE = np.cross(self.y_EE, self.z_EE)
               self.R_EE = np.zeros([3,3])
               self.R_EE[:, 0] = self.x_EE
               self.R_EE[:, 1] = self.y_EE
               self.R_EE[:, 2] = self.z_EE'''

               # From Baseline 3:
               self.z_EE = self.unit_u1
               # self.y_EE = np.asarray([0,1,0])
               self.y_EE = np.asarray([0,-1,0])
               # self.y_EE = np.asarray([-1,0,0])
               self.x_EE = np.cross(self.y_EE, self.z_EE)
               self.R_EE = np.zeros([3,3])
               self.R_EE[:, 0] = self.x_EE
               self.R_EE[:, 1] = self.y_EE
               self.R_EE[:, 2] = self.z_EE

               # Intermediate end-effector pose orientation:
               self.R_EE_inter = self.R_EE

               # End Effector Position: 
               self.p_EE = np.add(self.grasp_center, np.dot(-1*(self.g_delta + self.gripper_height_tolerance), self.unit_u1))
               self.p_EE_inter = np.add(self.grasp_center, np.dot(-1*(self.g_delta_inter + self.gripper_height_tolerance), self.unit_u1))    

               # End Effector pose as an element of SE(3) (4x4 transformation matrix):
               self.gripper_pose = np.zeros([4,4])
               self.gripper_pose[0:3, 0:3] = self.R_EE
               self.gripper_pose[0:3, 3] = np.reshape(self.p_EE, [3])
               self.gripper_pose[3,3] = 1
               self.computed_end_effector_poses.append(self.gripper_pose)

               # End Effector pose as an element of SE(3) (4x4 transformation matrix):
               self.gripper_pose_inter = np.zeros([4,4])
               self.gripper_pose_inter[0:3, 0:3] = self.R_EE_inter
               self.gripper_pose_inter[0:3, 3] = np.reshape(self.p_EE_inter, [3])
               self.gripper_pose[3,3] = 1
               self.computed_end_effector_poses_inter.append(self.gripper_pose_inter)

               # Saving the pose of the corresponding object-end_effector contact reference frames:
               self.sampled_pose_c1 = np.zeros([4,4])
               self.sampled_pose_c1[0:3, 0:3] = R_C1
               self.sampled_pose_c1[0:3, 3] = np.reshape(self.position_C1, [3])
               self.sampled_pose_c1[3,3] = 1
               self.sampled_contacts_c1.append(self.sampled_pose_c1)

               self.sampled_pose_c2 = np.zeros([4,4])
               self.sampled_pose_c2[0:3, 0:3] = R_C2
               self.sampled_pose_c2[0:3, 3] = np.reshape(self.position_C2, [3])
               self.sampled_pose_c2[3,3] = 1
               self.sampled_contacts_c2.append(self.sampled_pose_c2)

               # Appending the grasp center and normals for visualization:
               self.grasp_centers.append(self.grasp_center)
               self.normals_c1.append(R_C1[:, 2])
               self.normals_c2.append(R_C2[:, 2])
               self.unit_ulist.append(self.unit_u1)

               # Additional attribute for experimental purposes:
               self.approach_dir_2_poses.append(self.gripper_pose)
               self.approach_dir_2_inter_poses.append(self.gripper_pose_inter)
               
            # Checking the third approach direction:
            if self.distance_2 < self.gripper_height_tolerance: 
               # End Effector Orientation Based on the Franka Panda Convention: 
               '''self.z_EE = self.unit_u2
               self.x_EE = np.asarray([0,0,1])
               self.y_EE = self.z_C1
               self.x_EE = np.cross(self.y_EE, self.z_EE)
               self.R_EE = np.zeros([3,3])
               self.R_EE[:, 0] = self.x_EE
               self.R_EE[:, 1] = self.y_EE
               self.R_EE[:, 2] = self.z_EE'''

               # From Baseline 3:
               self.z_EE = self.unit_u2
               # self.x_EE = np.asarray([0,0,-1])
               self.x_EE = np.asarray([0,0,1])
               self.y_EE = np.cross(self.z_EE, self.x_EE)
               self.R_EE = np.zeros([3,3])
               self.R_EE[:, 0] = self.x_EE
               self.R_EE[:, 1] = self.y_EE
               self.R_EE[:, 2] = self.z_EE

               # Intermediate end-effector pose orientation:
               self.R_EE_inter = self.R_EE

               # End Effector Position: 
               self.p_EE = np.add(self.grasp_center, np.dot(-1*(self.g_delta + self.gripper_height_tolerance), self.unit_u2))
               self.p_EE_inter = np.add(self.grasp_center, np.dot(-1*(self.g_delta_inter + self.gripper_height_tolerance), self.unit_u2)) 

               # End Effector pose as an element of SE(3) (4x4 transformation matrix):
               self.gripper_pose = np.zeros([4,4])
               self.gripper_pose[0:3, 0:3] = self.R_EE
               self.gripper_pose[0:3, 3] = np.reshape(self.p_EE, [3])
               self.gripper_pose[3,3] = 1
               self.computed_end_effector_poses.append(self.gripper_pose)

               # End Effector pose as an element of SE(3) (4x4 transformation matrix):
               self.gripper_pose_inter = np.zeros([4,4])
               self.gripper_pose_inter[0:3, 0:3] = self.R_EE_inter
               self.gripper_pose_inter[0:3, 3] = np.reshape(self.p_EE_inter, [3])
               self.gripper_pose[3,3] = 1
               self.computed_end_effector_poses_inter.append(self.gripper_pose_inter)

               # Saving the pose of the corresponding object-end_effector contact reference frames:
               self.sampled_pose_c1 = np.zeros([4,4])
               self.sampled_pose_c1[0:3, 0:3] = R_C1
               self.sampled_pose_c1[0:3, 3] = np.reshape(self.position_C1, [3])
               self.sampled_pose_c1[3,3] = 1
               self.sampled_contacts_c1.append(self.sampled_pose_c1)

               self.sampled_pose_c2 = np.zeros([4,4])
               self.sampled_pose_c2[0:3, 0:3] = R_C2
               self.sampled_pose_c2[0:3, 3] = np.reshape(self.position_C2, [3])
               self.sampled_pose_c2[3,3] = 1
               self.sampled_contacts_c2.append(self.sampled_pose_c2)

               # Appending the grasp center and normals for visualization:
               self.grasp_centers.append(self.grasp_center)
               self.normals_c1.append(R_C1[:, 2])
               self.normals_c2.append(R_C2[:, 2])
               self.unit_ulist.append(self.unit_u2)

               # Additional attribute for experimental purposes:
               self.approach_dir_other_poses.append(self.gripper_pose)
               self.approach_dir_other_inter_poses.append(self.gripper_pose_inter)

            # Checking the fifth approach direction:
            if self.distance_3 < self.gripper_height_tolerance: 
               # End Effector Orientation Based on the Franka Panda Convention: 
               '''self.z_EE = self.unit_u3
               self.y_EE = self.z_C2
               self.x_EE = np.cross(self.y_EE, self.z_EE)
               self.R_EE = np.zeros([3,3])
               self.R_EE[:, 0] = self.x_EE
               self.R_EE[:, 1] = self.y_EE
               self.R_EE[:, 2] = self.z_EE'''

               # From Baseline 3:
               self.z_EE = self.unit_u3
               # self.x_EE = np.asarray([0,0,-1])
               self.x_EE = np.asarray([0,0,1])
               self.y_EE = np.cross(self.z_EE, self.x_EE)
               self.R_EE = np.zeros([3,3])
               self.R_EE[:, 0] = self.x_EE
               self.R_EE[:, 1] = self.y_EE
               self.R_EE[:, 2] = self.z_EE

               # Intermediate end-effector pose orientation:
               self.R_EE_inter = self.R_EE

               # End Effector Position: 
               self.p_EE = np.add(self.grasp_center, np.dot(-1*(self.g_delta + self.gripper_height_tolerance), self.unit_u3))
               self.p_EE_inter = np.add(self.grasp_center, np.dot(-1*(self.g_delta_inter + self.gripper_height_tolerance), self.unit_u3)) 

               # End Effector pose as an element of SE(3) (4x4 transformation matrix):
               self.gripper_pose = np.zeros([4,4])
               self.gripper_pose[0:3, 0:3] = self.R_EE
               self.gripper_pose[0:3, 3] = np.reshape(self.p_EE, [3])
               self.gripper_pose[3,3] = 1
               self.computed_end_effector_poses.append(self.gripper_pose)

               # End Effector pose as an element of SE(3) (4x4 transformation matrix):
               self.gripper_pose_inter = np.zeros([4,4])
               self.gripper_pose_inter[0:3, 0:3] = self.R_EE_inter
               self.gripper_pose_inter[0:3, 3] = np.reshape(self.p_EE_inter, [3])
               self.gripper_pose[3,3] = 1
               self.computed_end_effector_poses_inter.append(self.gripper_pose_inter)

               # Saving the pose of the corresponding object-end_effector contact reference frames:
               self.sampled_pose_c1 = np.zeros([4,4])
               self.sampled_pose_c1[0:3, 0:3] = R_C1
               self.sampled_pose_c1[0:3, 3] = np.reshape(self.position_C1, [3])
               self.sampled_pose_c1[3,3] = 1
               self.sampled_contacts_c1.append(self.sampled_pose_c1)

               self.sampled_pose_c2 = np.zeros([4,4])
               self.sampled_pose_c2[0:3, 0:3] = R_C2
               self.sampled_pose_c2[0:3, 3] = np.reshape(self.position_C2, [3])
               self.sampled_pose_c2[3,3] = 1
               self.sampled_contacts_c2.append(self.sampled_pose_c2)

               # Appending the grasp center and normals for visualization:
               self.grasp_centers.append(self.grasp_center)
               self.normals_c1.append(R_C1[:, 2])
               self.normals_c2.append(R_C2[:, 2])
               self.unit_ulist.append(self.unit_u3)
               
               # Additional attribute for experimental purposes:
               self.approach_dir_other_poses.append(self.gripper_pose)
               self.approach_dir_other_inter_poses.append(self.gripper_pose_inter)

         # Outer conditional statement to check whether the dimensions along which we are grasping are less than the gripper width tolerance:
         if self.x_dim < self.gripper_width_tolerance:
            # Approach Direction 2:
            self.plane_points_1 = np.asarray([self.transformed_vertices_object_frame[3], self.transformed_vertices_object_frame[6], self.transformed_vertices_object_frame[4], self.transformed_vertices_object_frame[5]])
            # Approach Direction 1:
            self.plane_points_2 = np.asarray([self.transformed_vertices_object_frame[3], self.transformed_vertices_object_frame[0], self.transformed_vertices_object_frame[1], self.transformed_vertices_object_frame[6]])
            # Approach Direction 4:
            self.plane_points_3 = np.asarray([self.transformed_vertices_object_frame[4], self.transformed_vertices_object_frame[7], self.transformed_vertices_object_frame[2], self.transformed_vertices_object_frame[5]])

            self.center_point_plane_1 = np.asarray([self.p_base[0], self.p_base[1], self.plane_points_1[0, 2]])
            self.center_point_plane_2 = np.asarray([self.p_base[0], self.plane_points_2[0, 1], self.p_base[2]])
            self.center_point_plane_3 = np.asarray([self.p_base[0], self.plane_points_3[0, 1], self.p_base[2]]) 

            # Orientation of the contact reference frames: 
            # Getting an orthogonal reference frame where the z axis is along the normal: 
            # Frame C1:
            self.z_C1 = np.asarray([-1, 0, 0])
            self.x_C1 = np.random.randn(3)
            self.x_C1 -= self.x_C1.dot(self.z_C1)*self.z_C1
            self.x_C1 /= la.norm(self.x_C1)
            self.y_C1 = np.cross(self.z_C1, self.x_C1)
            R_C1 = np.zeros([3,3])
            R_C1[:, 0] = self.x_C1
            R_C1[:, 1] = self.y_C1
            R_C1[:, 2] = self.z_C1

            # Frame C2:
            self.z_C2 = np.asarray([1, 0, 0])
            self.x_C2 = np.random.randn(3)
            self.x_C2 -= self.x_C2.dot(self.z_C2)*self.z_C2
            self.x_C2 /= la.norm(self.x_C2)
            self.y_C2 = np.cross(self.z_C2, self.x_C2)
            R_C2 = np.zeros([3,3])
            R_C2[:, 0] = self.x_C2
            R_C2[:, 1] = self.y_C2
            R_C2[:, 2] = self.z_C2

            self.position_C1 = np.asarray([self.transformed_vertices_object_frame[1,0], self.ideal_grasping_region_grid_centers[i][0], self.ideal_grasping_region_grid_centers[i][1]])
            self.position_C2 = np.asarray([self.transformed_vertices_object_frame[0,0], self.ideal_grasping_region_grid_centers[i][0], self.ideal_grasping_region_grid_centers[i][1]])
            self.grasp_center = np.add(self.position_C1, np.dot((self.x_dim)/2, self.z_C1)) 

            self.distance_1, self.unit_u1 = self.get_distance(self.plane_points_1, self.center_point_plane_1, self.grasp_center)
            self.distance_2, self.unit_u2 = self.get_distance(self.plane_points_2, self.center_point_plane_2, self.grasp_center)
            self.distance_3, self.unit_u3 = self.get_distance(self.plane_points_3, self.center_point_plane_3, self.grasp_center)

            # Checking the second approach direction:
            if self.distance_1 < self.gripper_height_tolerance:
               # End Effector Orientation Based on the Franka Panda Convention: 
               '''self.z_EE = -1*self.unit_u1
               self.y_EE = self.z_C2
               self.x_EE = np.cross(self.y_EE, self.z_EE)
               self.R_EE = np.zeros([3,3])
               self.R_EE[:, 0] = self.x_EE
               self.R_EE[:, 1] = self.y_EE
               self.R_EE[:, 2] = self.z_EE'''

               # From Baseline 3:
               self.z_EE = self.unit_u1
               # self.y_EE = np.asarray([0,1,0])
               self.y_EE = np.asarray([0,-1,0])
               # self.y_EE = np.asarray([1,0,0])
               self.x_EE = np.cross(self.y_EE, self.z_EE)
               self.R_EE = np.zeros([3,3])
               self.R_EE[:, 0] = self.x_EE
               self.R_EE[:, 1] = self.y_EE
               self.R_EE[:, 2] = self.z_EE

               # Intermediate end-effector pose orientation:
               self.R_EE_inter = self.R_EE

               # End Effector Position: 
               self.p_EE = np.add(self.grasp_center, np.dot(-1*(self.g_delta + self.gripper_height_tolerance), self.unit_u1)) 
               self.p_EE_inter = np.add(self.grasp_center, np.dot(-1*(self.g_delta_inter + self.gripper_height_tolerance), self.unit_u1))  

               # End Effector pose as an element of SE(3) (4x4 transformation matrix):
               self.gripper_pose = np.zeros([4,4])
               self.gripper_pose[0:3, 0:3] = self.R_EE
               self.gripper_pose[0:3, 3] = np.reshape(self.p_EE, [3])
               self.gripper_pose[3,3] = 1
               self.computed_end_effector_poses.append(self.gripper_pose)

               # End Effector pose as an element of SE(3) (4x4 transformation matrix):
               self.gripper_pose_inter = np.zeros([4,4])
               self.gripper_pose_inter[0:3, 0:3] = self.R_EE_inter
               self.gripper_pose_inter[0:3, 3] = np.reshape(self.p_EE_inter, [3])
               self.gripper_pose[3,3] = 1
               self.computed_end_effector_poses_inter.append(self.gripper_pose_inter)

               # Saving the pose of the corresponding object-end_effector contact reference frames:
               self.sampled_pose_c1 = np.zeros([4,4])
               self.sampled_pose_c1[0:3, 0:3] = R_C1
               self.sampled_pose_c1[0:3, 3] = np.reshape(self.position_C1, [3])
               self.sampled_pose_c1[3,3] = 1
               self.sampled_contacts_c1.append(self.sampled_pose_c1)

               self.sampled_pose_c2 = np.zeros([4,4])
               self.sampled_pose_c2[0:3, 0:3] = R_C2
               self.sampled_pose_c2[0:3, 3] = np.reshape(self.position_C2, [3])
               self.sampled_pose_c2[3,3] = 1
               self.sampled_contacts_c2.append(self.sampled_pose_c2)

               # Appending the grasp center and normals for visualization:
               self.grasp_centers.append(self.grasp_center)
               self.normals_c1.append(R_C1[:, 2])
               self.normals_c2.append(R_C2[:, 2])
               self.unit_ulist.append(self.unit_u1)

               # Additional attribute for experimental purposes:
               self.approach_dir_2_poses.append(self.gripper_pose)
               self.approach_dir_2_inter_poses.append(self.gripper_pose_inter)
               
            # Checking the first approach direction:
            if self.distance_2 < self.gripper_height_tolerance: 
               # End Effector Orientation Based on the Franka Panda Convention: 
               '''self.z_EE = self.unit_u2
               self.y_EE = self.z_C2
               self.x_EE = np.cross(self.y_EE, self.z_EE)
               self.R_EE = np.zeros([3,3])
               self.R_EE[:, 0] = self.x_EE
               self.R_EE[:, 1] = self.y_EE
               self.R_EE[:, 2] = self.z_EE'''

               # From Baseline 3:
               self.z_EE = self.unit_u2
               # self.x_EE = np.asarray([0,0,-1])
               self.x_EE = np.asarray([0,0,1])
               self.y_EE = np.cross(self.z_EE, self.x_EE)
               self.R_EE = np.zeros([3,3])
               self.R_EE[:, 0] = self.x_EE
               self.R_EE[:, 1] = self.y_EE
               self.R_EE[:, 2] = self.z_EE

               # Intermediate end-effector pose orientation:
               self.R_EE_inter = self.R_EE

               # End Effector Position: 
               self.p_EE = np.add(self.grasp_center, np.dot(-1*(self.g_delta + self.gripper_height_tolerance), self.unit_u2))
               self.p_EE_inter = np.add(self.grasp_center, np.dot(-1*(self.g_delta_inter + self.gripper_height_tolerance), self.unit_u2)) 

               # End Effector pose as an element of SE(3) (4x4 transformation matrix):
               self.gripper_pose = np.zeros([4,4])
               self.gripper_pose[0:3, 0:3] = self.R_EE
               self.gripper_pose[0:3, 3] = np.reshape(self.p_EE, [3])
               self.gripper_pose[3,3] = 1
               self.computed_end_effector_poses.append(self.gripper_pose)

               # End Effector pose as an element of SE(3) (4x4 transformation matrix):
               self.gripper_pose_inter = np.zeros([4,4])
               self.gripper_pose_inter[0:3, 0:3] = self.R_EE_inter
               self.gripper_pose_inter[0:3, 3] = np.reshape(self.p_EE_inter, [3])
               self.gripper_pose[3,3] = 1
               self.computed_end_effector_poses_inter.append(self.gripper_pose_inter)

               # Saving the pose of the corresponding object-end_effector contact reference frames:
               self.sampled_pose_c1 = np.zeros([4,4])
               self.sampled_pose_c1[0:3, 0:3] = R_C1
               self.sampled_pose_c1[0:3, 3] = np.reshape(self.position_C1, [3])
               self.sampled_pose_c1[3,3] = 1
               self.sampled_contacts_c1.append(self.sampled_pose_c1)

               self.sampled_pose_c2 = np.zeros([4,4])
               self.sampled_pose_c2[0:3, 0:3] = R_C2
               self.sampled_pose_c2[0:3, 3] = np.reshape(self.position_C2, [3])
               self.sampled_pose_c2[3,3] = 1
               self.sampled_contacts_c2.append(self.sampled_pose_c2)

               # Appending the grasp center and normals for visualization:
               self.grasp_centers.append(self.grasp_center)
               self.normals_c1.append(R_C1[:, 2])
               self.normals_c2.append(R_C2[:, 2])
               self.unit_ulist.append(self.unit_u2)

               # Additional attribute for experimental purposes:
               self.approach_dir_other_poses.append(self.gripper_pose)
               self.approach_dir_other_inter_poses.append(self.gripper_pose_inter)

            # Checking the fourth approach direction:
            if self.distance_3 < self.gripper_height_tolerance: 
               # End Effector Orientation Based on the Franka Panda Convention: 
               '''self.z_EE = self.unit_u3
               self.y_EE = self.z_C2
               self.x_EE = np.cross(self.y_EE, self.z_EE)
               self.R_EE = np.zeros([3,3])
               self.R_EE[:, 0] = self.x_EE
               self.R_EE[:, 1] = self.y_EE
               self.R_EE[:, 2] = self.z_EE'''

               # From Baseline 3:
               self.z_EE = self.unit_u3
               # self.x_EE = np.asarray([0,0,-1])
               self.x_EE = np.asarray([0,0,1])
               self.y_EE = np.cross(self.z_EE, self.x_EE)
               self.R_EE = np.zeros([3,3])
               self.R_EE[:, 0] = self.x_EE
               self.R_EE[:, 1] = self.y_EE
               self.R_EE[:, 2] = self.z_EE

               # Intermediate end-effector pose orientation:
               self.R_EE_inter = self.R_EE

               # End Effector Position: 
               self.p_EE = np.add(self.grasp_center, np.dot(-1*(self.g_delta + self.gripper_height_tolerance), self.unit_u3))
               self.p_EE_inter = np.add(self.grasp_center, np.dot(-1*(self.g_delta_inter + self.gripper_height_tolerance), self.unit_u3)) 

               # End Effector pose as an element of SE(3) (4x4 transformation matrix):
               self.gripper_pose = np.zeros([4,4])
               self.gripper_pose[0:3, 0:3] = self.R_EE
               self.gripper_pose[0:3, 3] = np.reshape(self.p_EE, [3])
               self.gripper_pose[3,3] = 1
               self.computed_end_effector_poses.append(self.gripper_pose)

               # End Effector pose as an element of SE(3) (4x4 transformation matrix):
               self.gripper_pose_inter = np.zeros([4,4])
               self.gripper_pose_inter[0:3, 0:3] = self.R_EE_inter
               self.gripper_pose_inter[0:3, 3] = np.reshape(self.p_EE_inter, [3])
               self.gripper_pose[3,3] = 1
               self.computed_end_effector_poses_inter.append(self.gripper_pose_inter)

               # Saving the pose of the corresponding object-end_effector contact reference frames:
               self.sampled_pose_c1 = np.zeros([4,4])
               self.sampled_pose_c1[0:3, 0:3] = R_C1
               self.sampled_pose_c1[0:3, 3] = np.reshape(self.position_C1, [3])
               self.sampled_pose_c1[3,3] = 1
               self.sampled_contacts_c1.append(self.sampled_pose_c1)

               self.sampled_pose_c2 = np.zeros([4,4])
               self.sampled_pose_c2[0:3, 0:3] = R_C2
               self.sampled_pose_c2[0:3, 3] = np.reshape(self.position_C2, [3])
               self.sampled_pose_c2[3,3] = 1
               self.sampled_contacts_c2.append(self.sampled_pose_c2)

               # Appending the grasp center and normals for visualization:
               self.grasp_centers.append(self.grasp_center)
               self.normals_c1.append(R_C1[:, 2])
               self.normals_c2.append(R_C2[:, 2])
               self.unit_ulist.append(self.unit_u3)

               # Additional attribute for experimental purposes:
               self.approach_dir_other_poses.append(self.gripper_pose)
               self.approach_dir_other_inter_poses.append(self.gripper_pose_inter)
      
      # Transforming the sampled end-effector poses back to the base reference frame.
      self.computed_end_effector_poses_base = []
      self.computed_end_effector_poses_inter_base = []

      # Additional attributes for the purposes of conducting experiments:
      self.approach_dir_2_poses_base = []
      self.approach_dir_2_inter_poses_base = []
      self.approach_dir_other_poses_base = []
      self.approach_dir_other_inter_poses_base = []
 
      for pose in self.computed_end_effector_poses:
         R_pose = pose[0:3, 0:3]
         p_pose = np.reshape(pose[0:3, 3], [3,1])
         self.R_EE_base = np.matmul(self.R_bounding_box, R_pose) 
         self.p_EE_base = self.p_bounding_box + np.dot(self.R_bounding_box, p_pose)
         self.gripper_pose_base = np.zeros([4,4])
         self.gripper_pose_base[0:3, 0:3] = self.R_EE_base
         self.gripper_pose_base[0:3, 3] = np.reshape(self.p_EE_base, [3])
         self.gripper_pose_base[3,3] = 1
         self.computed_end_effector_poses_base.append(self.gripper_pose_base) 

      # The intermediate end-effector poses are also transformed from the object reference frame to the base reference frame. 
      for pose in self.computed_end_effector_poses_inter:
         R_pose_inter = pose[0:3, 0:3]
         p_pose_inter = np.reshape(pose[0:3, 3], [3,1])
         self.R_EE_inter_base = np.matmul(self.R_bounding_box, R_pose_inter) 
         self.p_EE_inter_base = self.p_bounding_box + np.dot(self.R_bounding_box, p_pose_inter)
         self.gripper_pose_inter_base = np.zeros([4,4])
         self.gripper_pose_inter_base[0:3, 0:3] = self.R_EE_inter_base
         self.gripper_pose_inter_base[0:3, 3] = np.reshape(self.p_EE_inter_base, [3])
         self.gripper_pose_inter_base[3,3] = 1
         self.computed_end_effector_poses_inter_base.append(self.gripper_pose_inter_base)

      # Additional loop for the purpose of conducting experiments:
      for pose in self.approach_dir_2_poses:
         R_pose = pose[0:3, 0:3]
         p_pose = np.reshape(pose[0:3, 3], [3,1])
         self.R_EE_base = np.matmul(self.R_bounding_box, R_pose) 
         self.p_EE_base = self.p_bounding_box + np.dot(self.R_bounding_box, p_pose)
         self.gripper_pose_base = np.zeros([4,4])
         self.gripper_pose_base[0:3, 0:3] = self.R_EE_base
         self.gripper_pose_base[0:3, 3] = np.reshape(self.p_EE_base, [3])
         self.gripper_pose_base[3,3] = 1
         self.approach_dir_2_poses_base.append(self.gripper_pose_base)

      for pose in self.approach_dir_2_inter_poses:
         R_pose_inter = pose[0:3, 0:3]
         p_pose_inter = np.reshape(pose[0:3, 3], [3,1])
         self.R_EE_inter_base = np.matmul(self.R_bounding_box, R_pose_inter) 
         self.p_EE_inter_base = self.p_bounding_box + np.dot(self.R_bounding_box, p_pose_inter)
         self.gripper_pose_inter_base = np.zeros([4,4])
         self.gripper_pose_inter_base[0:3, 0:3] = self.R_EE_inter_base
         self.gripper_pose_inter_base[0:3, 3] = np.reshape(self.p_EE_inter_base, [3])
         self.gripper_pose_inter_base[3,3] = 1
         self.approach_dir_2_inter_poses_base.append(self.gripper_pose_inter_base)

      # Additional loop for the purpose of conducting experiments:
      for pose in self.approach_dir_other_poses:
         R_pose = pose[0:3, 0:3]
         p_pose = np.reshape(pose[0:3, 3], [3,1])
         self.R_EE_base = np.matmul(self.R_bounding_box, R_pose) 
         self.p_EE_base = self.p_bounding_box + np.dot(self.R_bounding_box, p_pose)
         self.gripper_pose_base = np.zeros([4,4])
         self.gripper_pose_base[0:3, 0:3] = self.R_EE_base
         self.gripper_pose_base[0:3, 3] = np.reshape(self.p_EE_base, [3])
         self.gripper_pose_base[3,3] = 1
         self.approach_dir_other_poses_base.append(self.gripper_pose_base)

      for pose in self.approach_dir_other_inter_poses:
         R_pose_inter = pose[0:3, 0:3]
         p_pose_inter = np.reshape(pose[0:3, 3], [3,1])
         self.R_EE_inter_base = np.matmul(self.R_bounding_box, R_pose_inter) 
         self.p_EE_inter_base = self.p_bounding_box + np.dot(self.R_bounding_box, p_pose_inter)
         self.gripper_pose_inter_base = np.zeros([4,4])
         self.gripper_pose_inter_base[0:3, 0:3] = self.R_EE_inter_base
         self.gripper_pose_inter_base[0:3, 3] = np.reshape(self.p_EE_inter_base, [3])
         self.gripper_pose_inter_base[3,3] = 1
         self.approach_dir_other_inter_poses_base.append(self.gripper_pose_inter_base)
      

   ''' Function for plotting a CUBE:'''
   def plot_cube(self):
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
      
      for i, v in enumerate(self.faces_vertices):
         for j in range(self.faces_vertices.shape[1]):
               self.faces[i][j, 0] = self.vertices[self.faces_vertices[i,j],0]
               self.faces[i][j, 1] = self.vertices[self.faces_vertices[i,j],1]
               self.faces[i][j, 2] = self.vertices[self.faces_vertices[i,j],2]
               
   '''Function to plot a reference frame:'''
   def plot_reference_frames(self, ax):
      ax.quiver(self.p[0], self.p[1], self.p[2], self.scale_value*self.R[0, 0], self.scale_value*self.R[1, 0], self.scale_value*self.R[2, 0], color = "r", arrow_length_ratio = self.length_value)
      ax.quiver(self.p[0], self.p[1], self.p[2], self.scale_value*self.R[0, 1], self.scale_value*self.R[1, 1], self.scale_value*self.R[2, 1], color = "g", arrow_length_ratio = self.length_value)
      ax.quiver(self.p[0], self.p[1], self.p[2], self.scale_value*self.R[0, 2], self.scale_value*self.R[1, 2], self.scale_value*self.R[2, 2], color = "b", arrow_length_ratio = self.length_value )
      
      return ax
