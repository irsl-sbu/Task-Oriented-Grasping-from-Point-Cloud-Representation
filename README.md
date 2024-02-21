# Task-Oriented-Grasping-from-Point-Cloud-Representation


In this paper, we study the problem of task-oriented grasp synthesis from partial point cloud data using an eye-in-hand camera configuration. In task-oriented grasp synthesis, a grasp has to be selected so that the object is not lost during manipulation, and it is also ensured that adequate force/moment can be applied to perform the task. We formalize the notion of a gross manipulation task as a constant screw motion (or a sequence of constant screw motions) to be applied to the object after grasping. Using this notion of task, and a corresponding grasp quality metric developed in our prior work, we use a neural network to approximate a function for predicting the grasp quality metric on a cuboid shape. We show that by using a bounding box obtained from the partial point cloud of an object, and the grasp quality metric mentioned above, we can generate a good grasping region on the bounding box that can be used to compute an antipodal grasp on the actual object. Our algorithm does not use any manually labeled data or grasping simulator, thus making it very efficient to implement and integrate with screw linear interpolation-based motion planners. We present simulation as well as experimental results that show the effectiveness of our approach.

This is the Python Implementation of neural network-based task-oriented grasp synthesis on object point clouds described in our IROS 2023 paper.
If you find this work useful please cite our work:


Citation:

```
@inproceedings{patankar2023task,
  title={Task-Oriented Grasping with Point Cloud Representation of Objects},
  author={Patankar, Aditya and Phi, Khiem and Mahalingam, Dasharadhan and Chakraborty, Nilanjan and Ramakrishnan, IV},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={6853--6860},
  year={2023},
  organization={IEEE}
}
<<<<<<< HEAD
```

# Installation 

1. Clone the repository

```
git clone git@github.com:irsl-sbu/Task-Oriented-Grasping-from-Point-Cloud-Representation.git
```

2. Create a conda environment using provided .yml file

```
cd Task-Oriented-Grasping-from-Point-Cloud-Representation
conda env create -f tograsp.yml
```

The repository currently contains point clouds (PLY format) captured from multiple camera views using Intel Realsense D415 for a CheezIt box, Domino Sugar box and a Ritz cracker box in the folder ``` partial_point_cloud ```. 

# Usage

Current implementation is for computing an ideal grasping region on point clouds of cuboidal objects for the task of pivoting the cuboidal object about one of its edges. The pivoting motion is a constant screw motion (pure rotation) about one of the edges and is represented using a screw axis. The location of the screw axis is approximated using the edges of the bounding box. 

1. Open a terminal and activate the conda environment

```
conda activate tograsp
```

2. Type the following command to execute and visualize the results on a CheezIt box:
   
``` 
python -u main_pivoting.py --filename partial_point_cloud/cheezit_cracker_box.ply --visualize
```

3. NOTE: Grasp synthesis for pouring using sensor data as well as for partial point clouds obtained from PyBullet to be updated soon. For further inquiries contact:  Aditya Patankar (aditya.patankar@stonybrook.edu)
