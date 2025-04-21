---
title: 3D Shape Completion
---
# Introduction
A fundamental component of robotics is the perception of the external environment. This enables robots to encode information about the space to inform decision-making. One such way to represent spatial information is via point clouds. Point clouds are a collection of points in 3D Euclidean space representing the geometry of an object or space. They are often generated using methods such as LiDAR or photogrammetry. Oftentimes, the point clouds these sensors generate are noisy or incomplete. 

In this project, we investigate the performance of classical and probabilistic machine learning methods under the constraint of partial and noisy observations of point cloud data. We primarily consider two tasks: predicting the object class and completing the full 3D shape given a partial representation. Through this, we aim to provide insights into the effectiveness of different approaches to tasks involving partially observed 3D forms, with potential applications in robotics, autonomous systems, and computer vision.

# Methodology
We explore generative modeling and classification methods on partial 3D point cloud data, specifically focusing on transformations of two shapes: cubes and spheres. 

## Data
In order to evaluate our approaches, we generated a synthetic point cloud dataset consisting of ellipsoids and parallelipipeds under various scalings, rotations, and translations contained within the subset $[-5,5]^{3} \subset \mathbb{R}^3$. The dataset generation process is summarized in [](#dataset_flowchart). 

```{figure} images/dataset_flowchart.png
:name: dataset_flowchart
To generate one data point in our dataset, we start by deciding on whether to generate an ellipsoid or parallelipiped. 
```


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
