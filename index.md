---
title: 3D Shape Completion
---
# Introduction
A fundamental component of robotics is the perception of the external environment. This enables robots to encode information about the space to inform decision-making. One such way to represent spatial information is via point clouds. Point clouds are a collection of points in 3D Euclidean space representing the geometry of an object or space. They are often generated using methods such as LiDAR or photogrammetry. Oftentimes, the point clouds these sensors generate are noisy or incomplete. 

In this project, we investigate the performance of classical and probabilistic machine learning methods under the constraint of partial and noisy observations of point cloud data. We primarily consider two tasks: predicting the object class and completing the full 3D shape given a partial representation. Through this, we aim to provide insights into the effectiveness of different approaches to tasks involving partially observed 3D forms, with potential applications in robotics, autonomous systems, and computer vision.

# Methodology
We explore generative modeling and classification methods on partial 3D point cloud data, specifically focusing on transformations of two shapes: cubes and spheres. 

## Data
In order to evaluate our approaches, we generated a synthetic point cloud dataset consisting of ellipsoids and parallelipipeds under various scalings, rotations, and translations contained within the subset $[-5,5]^{3} \subset \mathbb{R}^3$. 

```{figure} images/dataset_flowchart.png
:name: dataset_flowchart

A summary of the dataset generation process. Each data point consists of a label, its full 3D point cloud representation, and a partial noisy point cloud. 
```

A data point in the dataset is given by its class label, either cube or sphere, the full 3D point cloud representation, and the partial noisy point cloud representation.

To generate a data point, we first pick either a cube or sphere to start with, taking this to be the class label for the data point. Depending on the class label, we then generate the point cloud for a unit cube or unit sphere centered at the origin. Next, a random 3D rotation matrix is generated and applied to the point cloud. Then, each of the three dimensions is scaled by a random factor constrained by the condition that all points lie with the $[-5,5]^3$ region. The point cloud is then translated randomly within the valid region. We take the resultant point cloud as the corresponding full 3D point cloud representation of the data point.

To generate the partial noisy point cloud, we first choose a random unit vector in $\mathbb{R}^3$ to serve as the "viewpoint" of the agent looking at the 3D form. We take the dot product of each point with the viewpoint vector and keep only the points that are associated with the top $k$ values of the dot products to get the points that are "observable" by the model. For each of the remaining $k$ points in the partial point cloud, we perturb each point using independent, zero-mean Gaussian noise. This gives us the final piece of our data set, which is the partial noisy point cloud that would be observed by an agent.

To generate a data set, we repeat the process above until we get as many samples as desired. For our dataset, we generated 200 samples, 100 ellipsoids and 100 parallelipipeds. Each full 3D point cloud contained 10000 points, and each partial point cloud contained 1000 points. For perturbation, we used a multivariate Gaussian distribution with $\mu = 0$ and $\Sigma = 0.01 I_{\mathbb{R}^3}$. 

### Voxels

An issue we an encountered when working with point clouds is how to use point clouds as inputs and outputs. A 3D surface can be considered as a probability distribution over $\mathbb{R}^3$ and its point cloud representation can be viewed as multiple samples from this probability. Taking the point clouds as empirical probability distributions over $\mathbb{R}^3$, we would need to implement our algorithms over an infinite dimensional vector space. To simplify the implementation, we reparametrize our point clouds in terms of voxels. Voxels are a 3D analogue of 2D pixels and allow us to work in a finite dimensional vector space. In particular, we divide the range $[-5,5]$ into $d$ parts, and break up the space $[-5,5]^3$ into $d^3$ voxels. We assign a value of $1$ to a particular voxel if a point in the point cloud lies inside the the voxel, otherwise we assign it a value of $0$. In this way, we get a "discretized" view of the point cloud. 

## Classification of Partial Observations

### K-Nearest Neighbors

### Logistic Regression

## 3D Shape Completion

### Reconstruction

### Completion

# Results
## Classification

### K-Nearest Neighbors

### Logistic Regression

## 3D Shape Completion

### Reconstruction

### Completion
