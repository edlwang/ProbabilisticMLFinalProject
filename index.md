---
title: 3D Shape Completion
bibliography:
    - refs.bib
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

To generate a data set, we repeat the process above until we get as many samples as desired. For our dataset, we generated 200 samples, 100 ellipsoids and 100 parallelipipeds. Each full 3D point cloud contained 10000 points, and each partial point cloud contained 1000 points. For perturbation, we used a multivariate Gaussian distribution with $\mu = 0$ and $\Sigma = 0.01 I_{3}$. 

:::{dropdown} Dataset Generation Code
```{code} python
import matplotlib.pyplot as plt
import numpy as np
import scipy
import random
```
```{code} python
def plot_3d_point_cloud(points, name):
    """
    Plots a 3D point cloud.

    Args:
    - points: A NumPy array of shape (N, 3) containing the 3D coordinates.
    """
    # Check if points is a valid numpy array with shape (N, 3)
    if points.shape[1] != 3:
        raise ValueError("Input array must be of shape (N, 3)")

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract the x, y, and z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Plot the points
    ax.scatter(x, y, z, c='b', marker='o', s=5)  # You can customize color, size, etc.

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    # Show the plot
    plt.show()
    #plt.savefig(name + "Plot.png")
```
```{code} python
def genCubePCS(N):
    """
    To generate a point cloud for a cube: Just generate points randomly from all the 6 faces: If we want uniformity, then we'd want
    to generate points at a certain distance from one another.

    Sides of cube: x plane y plane z plane then same at distance 1 away.

    0: 000 100 110 010
    1 000 100 101 001
    2: 000 001 011 010
    3: 0 + (001)
    4: 1 + (010)
    5: 2 + (100)
    """
    facePoints = np.random.uniform(size = (N,2))
    #value mod 3 is the plane, then the value div by 3 is the close or far.
    face = np.random.randint( 0, 6, N)
    modVal = face%3
    divVal = np.floor(face/3)
    #need 0 on z axis.
    zp = facePoints[modVal == 0, :]
    op = facePoints[modVal == 1, :]
    tp = facePoints[modVal == 2, :]
    fz = face[modVal== 0]
    dz = np.floor(fz/3)[:, np.newaxis]
    fo = face[modVal == 1]
    do = np.floor(fo/3)[:, np.newaxis]
    ft = face[modVal == 2]
    dt = np.floor(ft/3)[:, np.newaxis]
    zpoints = np.concatenate((zp, np.zeros(shape = (zp.shape[0], 1))), axis=-1)
    opoints = np.concatenate((op[:, 0][:, np.newaxis], np.zeros(shape = (op.shape[0], 1)), op[:, 1][:, np.newaxis]), axis=-1)
    tpoints = np.concatenate((np.zeros(shape = (tp.shape[0], 1)), tp), axis=-1)

    addColZ = np.where(np.tile(dz == 1, (1,3)), np.tile(np.array([0,0,1]), (zp.shape[0], 1)), np.zeros((zp.shape[0], 3)))
    addColO = np.where(np.tile(do == 1, (1,3)), np.tile(np.array([0,1,0]), (op.shape[0], 1)), np.zeros((op.shape[0], 3)))
    addColT = np.where(np.tile(dt == 1, (1,3)), np.tile(np.array([1,0,0]), (tp.shape[0], 1)), np.zeros((tp.shape[0], 3)))
    points = np.concatenate([zpoints + addColZ, opoints + addColO, tpoints + addColT], axis=0)
    return points

def genSpherePCS(N, r):
    """

    """
    p = np.random.normal(loc = 0.0, scale = 1.0, size = (N, 3))
    p = p/np.linalg.norm(p, axis=-1)[:, np.newaxis]
    return r*p

def translatePC(points, tVec):
    newPoints = points.copy()
    newPoints += tVec
    return newPoints

def rotatePC(points, axis, angle):
    """
    Performs the rotation defined by the axis of rotation and the angle/degrees
    """
    axis = axis/np.linalg.norm(axis)
    R = scipy.spatial.transform.Rotation.from_rotvec(angle * axis)
    newPoints = R.apply(points)
    return newPoints

def scalePC(points, sVec):
    newPoints = points.copy()
    newPoints *= sVec
    return newPoints

def genViewConstrictedSubCloud(points, n, vVec):
    """
    Generates the subclouds with n points that would be most viewable if viewed from the direction vVec
    This works for small n and point clouds that form the surface of a convex object
    """
    distances = -vVec.T @ points.T
    return points[np.argsort(distances)[:n], :]

def addNoise(points, sigma2):
    """
    Adds Gaussian noise with variance sigma^2*I_3 to each point in the point cloud
    """
    noise = np.random.normal(loc = 0.0, scale = np.sqrt(sigma2), size = points.shape)
    return points + noise
```
```{code} python
def genDataPoint(generator, rotate=True, scale=True, translate=True, visiblePoints = 1000, sigma = 0.01):
    """
    Generates a datapoint starting from the point cloud generated by the generator
    generator () -> point cloud
    The points are then rotated > scaled > translated randomly to give the "true" point cloud
    1000 visible points are sampled from a random view direction and then noised to get the "input" to our dataset
    """
    truePC = generator()
    if rotate:
        randomAxis = np.random.multivariate_normal(mean = np.zeros(3), cov = np.eye(3))
        randomAxis /= np.linalg.norm(randomAxis)
        randomAngle = np.random.uniform(low = 0, high = 2*np.pi)
        truePC = rotatePC(truePC, randomAxis, randomAngle)
    if scale:
        xscale = np.random.rand() * 5/np.max(np.abs(truePC[:,0]))
        yscale = np.random.rand() * 5/np.max(np.abs(truePC[:,1]))
        zscale = np.random.rand() * 5/np.max(np.abs(truePC[:,2]))
        truePC = scalePC(truePC, np.array([xscale, yscale, zscale]))
    if translate:
        xtranslate = np.random.uniform(-5 - np.min(truePC[:,0]), 5 - np.max(truePC[:,0]))
        ytranslate = np.random.uniform(-5 - np.min(truePC[:,1]), 5 - np.max(truePC[:,1]))
        ztranslate = np.random.uniform(-5 - np.min(truePC[:,2]), 5 - np.max(truePC[:,2]))
        truePC = translatePC(truePC, np.array([xtranslate, ytranslate, ztranslate]))
    assert(np.max(np.abs(truePC)) < 5)
    inputPC = truePC.copy()
    viewDirection = np.random.multivariate_normal(mean = np.zeros(3), cov = np.eye(3))
    viewDirection /= np.linalg.norm(viewDirection)
    inputPC = addNoise(genViewConstrictedSubCloud(inputPC, visiblePoints, viewDirection), sigma)
    return (inputPC, truePC)
```
```{code} python
(squareInput, squareTrue) = genDataPoint(lambda : genCubePCS(10000) - np.array([0.5,0.5,0.5]), rotate=False, translate=False, sigma=0)
(sphereInput, sphereTrue) = genDataPoint(lambda : genSpherePCS(10000, 1), rotate=False, translate=False, sigma=0)

plot_3d_point_cloud(squareInput, "Square Input")
plot_3d_point_cloud(squareTrue, "Square True")
plot_3d_point_cloud(sphereInput, "Sphere Input")
plot_3d_point_cloud(sphereTrue, "Sphere True")
```
```{code} python
def genDataset(Nsquare, Nsphere, rotate=True, scale=True, translate=True, visiblePoints = 1000, sigma = 0.01):
    dataset = []
    for i in range(Nsquare):
        dataInput, dataTruth = genDataPoint(lambda : genCubePCS(10000) - np.array([0.5,0.5,0.5]), rotate=rotate, scale=scale, translate=translate, visiblePoints=visiblePoints, sigma=sigma)
        dataset.append((dataInput, dataTruth, "square"))
    for i in range(Nsphere):
        dataInput, dataTruth = genDataPoint(lambda : genSpherePCS(10000, 1), rotate=rotate, scale=scale, translate=translate, visiblePoints=visiblePoints, sigma=sigma)
        dataset.append((dataInput, dataTruth, "sphere"))
    random.shuffle(dataset)
    return dataset
```
```{code} python
dataset = genDataset(100,100)
for i in range(5):
    print(f"Datapoint {i}, label = {dataset[i][2]}")
    plot_3d_point_cloud(dataset[i][0], "Input")
    plot_3d_point_cloud(dataset[i][1], "Truth")
```
```{code} python
def saveDatasetToFile(dataset, filename):
    inputPoints, truePoints, label = zip(*dataset)
    inputPoints = np.array(inputPoints)
    truePoints = np.array(truePoints)
    label = np.array(label)
    np.savez(filename, inputPoints=inputPoints, truePoints=truePoints, label=label)
```
```{code} python
saveDatasetToFile(dataset, 'dataset.npz')
```
:::

### Voxels

An issue we an encountered when working with point clouds was how to use point clouds as inputs and outputs. A 3D surface can be considered as a probability distribution over $\mathbb{R}^3$ and its point cloud representation can be viewed as multiple samples from this probability. Taking the point clouds as empirical probability distributions over $\mathbb{R}^3$, we would need to implement our algorithms over an infinite dimensional vector space. To simplify the implementation, we reparametrize our point clouds in terms of voxels. Voxels are a 3D analogue of 2D pixels and allow us to work in a finite dimensional vector space. In particular, we divide the range $[-5,5]$ into $d$ parts, and break up the space $[-5,5]^3$ into $d^3$ voxels. We assign a value of $1$ to a particular voxel if a point in the point cloud lies inside the the voxel, otherwise we assign it a value of $0$. In this way, we get a "discretized" view of the point cloud. This transforms our problem into the space $\mathbb{R}^{d^3}$, and we can readily apply most classical and probabilistic machine learning techniques. In our implementation, we let $d=50$. 

```{figure} images/point_cloud_voxel_comparison.png
:name: point_cloud_voxel_comparison
The point cloud and voxel representations of the same 3D ellipsoid
```

## Classification of Partial Observations

For this problem, we take the partial and noisy 3D forms and attempt to classify them as either being an ellipsoid or a parallelipiped. We take two approaches to this classification problem. First, we try a classical approach, logistic regression, applied to the voxel representation of the partial 3D form. Second, we try something with a more probabilistic flavor, treating a point cloud as an empirical distribution and applying K-Nearest Neighbors (KNN) classification using a distance measure for probability distributions such as Chamfer or Earthmover's distance. This task is particularly useful as it allows for the implementation of conditional generative models through (THE NAME OF THE TECHNIQUE THIS IS CALLED). 

```{figure} images/classification_example.png
We are trying to learn a function $f$ to distinguish between data points generated initially from a sphere and those generated initially from a cube. 
```

### Logistic Regression

For logistic regression, we take in as input a set of $N$ labeled training examples, where the training examples are the voxel representation of the partial 3D form in $\mathbb{R}^{d \times d \times d}$. We flatten the data into a vector $\mathbb{R}^{d^3}$ and take these to be our features for logistic regression. The features are then standardized (REPLACE/ADD WITH WHAT WE ACTUALLY DO FOR THE PREPROCESSING PIPELINE)

### K-Nearest Neighbors

Let $A$ and $B$ be two sets of points. Let $|A|$ denote the cardinality of the set and $A^{(i)}$ denote the $i$th point. We can define the probability measure $P_A$ associated with the point cloud $A$ as

$$
P_A(S) = \dfrac{1}{|A|} \sum_{i=1}^{|A|} \boldsymbol{1}_{A^{(i)} \in S}\\
$$

and $P_B$ for point cloud $B$ analagously. We then define the distance between point clouds $A$ and $B$ as the distance between probability measures $P_A$ and $P_B$. (CHANGE FORMULA BASED ON ACTUAL IMPLEMENTATION)

$$
d(A,B) = EMD(P, Q)
$$

Using this distance metric, we then apply K-Nearest Neighbor classification to the set of point clouds. (What values of k are we using)

## 3D Shape Completion

For our second task, we start with the voxel representation of the partial 3D form and attempt to complete the rest of the form. To accomplish this, we utilize Variational Auto-Encoders (VAE). Variatonal autoencoders apply variational inference in order to approximate the distribution of the latent variables given the data (encoding) and the data given the latent variables (decoding), learning the data distribution in the process [@kingma2013auto]. We chose this approach since despite having $d^3$ voxels and features, the actual generation of each example in the dataset is governed by a small number of latent variables controlling the effect of scaling, rotation, translation, and noising is applied.

```{figure} images/3D_shape_completion.png
Reconstruction: We train a VAE to first reconstruct voxel representations of the full 3D form from the original 3D form
Completion: We train a VAE to complete the voxel representation of the full 3D shape from the noisy, partial version
```

### Reconstruction

As a precursor to the completion task, we tackle the typical VAE problem, which involves training an encoder that learns the distribution of the latent variables given the full 3D form and a decoder that learns the distribution of the voxels in the full 3D form given the latent variables. (TALK ABOUT ARCHITECTURE, TRAINING, EPOCHS, etc)

### Completion

After learning the distribution of the 3D forms, we then modify our training dataset to take an incomplete 3D forms as input and output the full 3D form. Thus, the encoder learns the distribution of latent variables conditioned on the partial voxel image and the decoder learns the distribution of voxels in the full voxel image. (TALK ABOUT ANY ADDITIONAL CHANGES THAT WERE MADE COMPARED TO RECONSTRUCTION)

# Results
## Classification

### Convolutional Neural Network

#### Architecture

We implemented a 3D Convolutional Neural Network (CNN) for binary classification ('cube' vs 'sphere') of voxelized partial point clouds. The network accepts input voxel grids of size $ 32 \times 32 \times 32 $ (denoted as $D=32$). The architecture comprises three convolutional blocks followed by a fully connected classifier. Each convolutional block consists of a 3D convolutional layer (`Conv3d`) with kernel size $k=3$, stride $s=1$, and padding $p=1$, followed by a ReLU activation (`ReLU`) and a 3D max pooling layer (`MaxPool3d`) with kernel size $k=2$ and stride $s=2$. The number of output channels $C_{out}$ increases through the blocks: Block 1 has $C_{out}=16$, Block 2 has $C_{out}=32$, and Block 3 has $C_{out}=64$. This sequence transforms the input tensor shape from $(B, 1, D, D, D)$ to $(B, 16, D/2, D/2, D/2)$, then to $(B, 32, D/4, D/4, D/4)$, and finally to $(B, 64, D/8, D/8, D/8)$, where $B$ is the batch size. For $D=32$, the final feature map size is $(B, 64, 4, 4, 4)$. This output is flattened into a vector $ x_{flat} \in \mathbb{R}^{4096} $ (since $64 \times 4^3 = 4096$). The classifier then processes this vector through a linear layer mapping $4096 \to 512$ features, followed by ReLU activation, a dropout layer with probability $p_{dropout}=0.5$ for regularization, and a final linear layer mapping $512 \to 2$ output classes. The `nn.CrossEntropyLoss` used incorporates the final log-softmax operation.

#### Training and Results

The CNN was trained using the Adam optimizer with a learning rate $\eta = 1 \times 10^{-4}$ and minimized the Cross-Entropy loss. The dataset was partitioned into 80% training and 20% validation samples. Training proceeded for 50 epochs using a batch size $B=64$. The training and validation loss and accuracy curves are presented below.

```{figure} images/CNN_Loss_Accuracy_Curves.png
:name: cnn_loss_accuracy_curves

Training and validation loss ($L$) and accuracy (Acc) curves for the CNN classifier over 50 epochs.
```

The results show effective learning on the training set, with $L_{train} \to 0$ and $Acc_{train} \to 100\%$. However, the validation performance indicates overfitting. The validation loss $L_{val}$ diverges from $L_{train}$ after approximately 10 epochs, starting to increase while $L_{train}$ continues to decrease. Concurrently, the validation accuracy $Acc_{val}$ plateaus around $75-76\%$, failing to generalize as well as the training performance suggests. This divergence confirms that the model learned training set idiosyncrasies that did not transfer to unseen validation data. The final $Acc_{val}$ after 50 epochs was approximately 76%.

### K-Nearest Neighbors


```{figure} images/KNN_train_confusion_plot.png
Confusion plot for the KNN model on the training dataset
```
Report accuracy, precision, recall, any other metrics we want

```{figure} images/KNN_test_confusion_plot.png
Confusion plot for the KNN model on the test dataset
```
Report accuracy, precision, recall, any other metrics we want

## 3D Shape Completion

### Reconstruction

```{figure} images/reconstruction_train_loss_graph.png
Loss over epochs for training the VAE to do reconstruction
```
Report final train and test loss

```{figure} images/reconstruction_example.png
Qualitative comparison between a couple examples of the full 3D representation and the VAE reconstruction
```

### Completion


```{figure} images/reconstruction_train_loss_graph.png
Loss over epochs for training the VAE to do reconstruction
```
Report final train and test loss

```{figure} images/reconstruction_example.png
Qualitative comparison between a couple examples of the partial, noisy 3D form, full 3D representation, and the VAE completion
```

# Discussion

Classification sucks and VAEs work kinda. 

## Future Work

- Combine generative modeling and classification to do conditional generative modelling
- Add more complex forms in the dataset
- Compare how the percentage of points seen affects the performance (is there a phase transition where the tasks is exceedingly easy above a certain threshold percentage but very difficult below)
- Utilize more sophisticated classification algorithms (ex: feature select from a CNN before classifying)
- Try out different diffusion models and try to work specifically with point clouds (since the number of voxels grows at a very fast rate when increasing precision)
    - Discuss methods we investigated (autoregressive (transformers?), diffusion) but were not able to implement given time constraints.
