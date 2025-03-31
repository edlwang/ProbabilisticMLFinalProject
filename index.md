---
title: 3D Point Cloud Completion
abstract: |
    abstract

    abstract

---
# Introduction
A fundamental component of robotics is the perception of the external environment. This enables robots to encode information about the space to inform decision-making. One such way to represent spatial information is via point clouds. Point clouds are a collection of points in 3D euclidean space representing the geometry of an object or space. They are often generated using methods such as LiDAR or photogrammetry. Oftentimes, the point clouds these sensors generate are noisy or incomplete. In this project we aim to generate robot grasp poses/reconstruct point clouds conditioned on partial point cloud observations of household items. Our pipeline is two-fold: predicting the object class from a partial point cloud and reconstructing point clouds conditioned on the predicted object class and partial observations. We then apply previous work of generating robot grasp poses from point clouds.

By integrating classical classification approaches with modern generative models, this project aims to provide insights into robust reconstruction and classification of partially observed 3D point clouds, with potential applications in robotics, autonomous systems, and computer vision.

# Methodology
We explore conditional generative modeling and classification methods on partial 3D point cloud data, specifically focusing on two shapes: cubes and spheres. Given an initial dataset consisting of fully sampled point clouds, we will artificially generate partial observations by randomly removing data points at varying degrees—specifically, retaining 75%, 50%, and 25% of the original points. We also augment this data by applying various rotations and scalings.

## Data
- Generate synthetic partial point clouds from complete shapes to simulate realistic scenarios of incomplete 3D data.
- Structure the resulting partial datasets into tensors of shape (n_points, x, y, z) suitable for machine learning workflows.

## Classification of Partial Observations
- Implement and evaluate multiple classification methods such as K-Nearest Neighbors (KNN), Logistic Regression, and simple Euclidean-distance-based classifiers to assess their effectiveness in predicting shape class given partial observations.
- Estimate posterior class probabilities, formally represented as p(class | partial cloud).

## Point Cloud Completion
- Implement and evaluate multiple classification methods such as K-Nearest Neighbors (KNN), Logistic Regression, and simple Euclidean-distance-based classifiers to assess their effectiveness in predicting shape class given partial observations.
- Estimate posterior class probabilities, formally represented as p(class | partial cloud).
