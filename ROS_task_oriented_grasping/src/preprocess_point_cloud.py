import open3d as o3d
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from collections import Counter


class pointCloud(object):

   def __init__(self): 
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

   '''Function to process the point clouds based on the normal information.
   Input: Downsampled Point Cloud Object
   Output: Point Cloud Object after removing the points corresponding to the flat surfaces/tables'''
   
   def removePlaneSurface(self):
      # Invalidating the existing normals
      self.cloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
      # Estimating new normals for the point cloud
      self.cloud.estimate_normals()
      self.cloud.orient_normals_consistent_tangent_plane(30)
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
         if max_idx[0] == 1:
         #if max_idx[0] == 0:
               point_indices.append(i)
               point_normals.append(self.normals[i])
               
      # Get the points corresponding to the normals pointing upward.
      '''Here we assume that the normals and points are ordered in the same way. That is the index of the normal is same
         as the index for the corresponding point'''
      self.points = np.asarray(self.cloud.points)
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

if __name__ == '__main__':
   cloud_object = pointCloud()
   pcd =  o3d.io.read_point_cloud("cheezItBB.pcd")

   # Downsample it and inspect the normals
   cloud_object.cloud = pcd.voxel_down_sample(voxel_size=10)
   cloud_object.removePlaneSurface()

   # Visualizing the downsampled point cloud. 
   o3d.visualization.draw_geometries([cloud_object.cloud])

   # Specifying parameters for DBSCAN Clustering:
   # Just like the parameters for downsampling even the parameters for DBSCAN Clustering are dependent on the 
   # units used computing and extracting the point cloud data.
   cloud_object.eps = 20
   cloud_object.min_points = 10
   cloud_object.getObjectPointCloud()

   # Visualizing the downsampled point cloud. 
   o3d.visualization.draw_geometries([cloud_object.cloud])
   o3d.visualization.draw_geometries([cloud_object.processed_cloud])

   # Computing the bounding boxes corresponding to the object point cloud: 
   cloud_object.computeBoundingBox()

   # Visualizing the bounding boxes along with the object point cloud: 
   o3d.visualization.draw_geometries([cloud_object.processed_cloud, cloud_object.aligned_bounding_box, cloud_object.oriented_bounding_box])

   # Saving the vertices and the centers of the bounding boxes: 
   np.savetxt('oriented_box_center_cheezIt.csv', cloud_object.oriented_bounding_box_center, delimiter = ",")
   np.savetxt('oriented_box_vertices_cheezIt.csv', cloud_object.oriented_bounding_box_vertices, delimiter = ",")

   np.savetxt('aligned_box_center_cheezIt.csv', cloud_object.aligned_bounding_box_center, delimiter = ",")
   np.savetxt('aligned_box_vertices_cheezIt.csv', cloud_object.aligned_bounding_box_vertices, delimiter = ",")

   # Saving the processed point cloud: 
   o3d.io.write_point_cloud("cheezIt_Processed.ply", cloud_object.processed_cloud)