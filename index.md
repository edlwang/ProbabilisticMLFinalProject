---
title: 3D Shape Completion
bibliography:
    - refs.bib
---
# Introduction
A fundamental component of robotics is the perception of the external environment. This enables robots to encode information about the space to inform decision-making. One such way to represent spatial information is via point clouds. Point clouds are a collection of points in 3D Euclidean space representing the geometry of an object or space. They are often generated using methods such as LiDAR or photogrammetry. Oftentimes, the point clouds these sensors generate are noisy or incomplete. 

In this project, we investigate the performance of classical and probabilistic machine learning methods under the constraint of partial and noisy observations of point cloud data. We primarily consider two tasks: predicting the object class and completing the full 3D shape given a partial representation. Our motivation for this comes from a robotics paper[@se3diffusionfields], where they use these two steps on the shapeNet dataset before the rest of their method doing diffusion on robot gripper poses.  Through this, we aim to provide insights into the effectiveness of different approaches to tasks involving partially observed 3D forms, with potential applications in robotics, autonomous systems, and computer vision.

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

To generate a data set, we repeat the process above until we get as many samples as desired. Each full 3D point cloud contained 10000 points, and each partial point cloud contained 1000 points. For perturbation, we used a multivariate Gaussian distribution with $\mu = 0$ and $\Sigma = 0.01 I_{3}$. 

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

For this problem, we take the partial and noisy 3D forms and attempt to classify them as either being an ellipsoid or a parallelipiped. We take two approaches to this classification problem. First we applied 3D convolutional neural networks to predict the shape from a partial voxelized representation. We also applied K-Nearest Neighbors (KNN), using a distance measure for probability distributions such as Chamfer or Earthmover's distance, to classify point clouds as an empirical distribution. This task is particularly useful as it allows for the implementation of generative models conditioned on the classifier's predictions. 


### Convolutional Neural Network

#### Architecture

We implemented a 3D Convolutional Neural Network (CNN) for binary classification ('parallelepiped' vs 'sphere') of voxelized partial point clouds. The network accepts input voxel grids of size $32 \times 32 \times 32$ (denoted as $D=32$). The architecture comprises three convolutional blocks followed by a fully connected classifier. Each convolutional block consists of a 3D convolutional layer (`Conv3d`) with kernel size $k=3$, stride $s=1$, and padding $p=1$, followed by a ReLU activation (`ReLU`) and a 3D max pooling layer (`MaxPool3d`) with kernel size $k=2$ and stride $s=2$. The number of output channels $C_{out}$ increases through the blocks: Block 1 has $C_{out}=16$, Block 2 has $C_{out}=32$, and Block 3 has $C_{out}=64$. This sequence transforms the input tensor shape from $(B, 1, D, D, D)$ to $(B, 16, D/2, D/2, D/2)$, then to $(B, 32, D/4, D/4, D/4)$, and finally to $(B, 64, D/8, D/8, D/8)$, where $B$ is the batch size. Each extra output channel is an extra learnable filter that can specialize in a new pattern. As you go deeper and the raw spatial resolution shrinks, the network compensates by adding filters so it can still capture enough information. For $D=32$, the final feature map size is $(B, 64, 4, 4, 4)$. This output is flattened into a vector $x_{flat} \in \mathbb{R}^{4096}$ (since $64 \times 4^3 = 4096$). The classifier then processes this vector through a linear layer mapping $4096 \to 512$ features, followed by ReLU activation, a dropout layer with probability $p_{dropout}=0.5$ for regularization, and a final linear layer mapping $512 \to 2$ output classes. The `nn.CrossEntropyLoss` used incorporates the final log-softmax operation.


:::{dropdown} CNN Architecture and Training Code
```{code} Python

class VoxelCNN(nn.Module):
    def __init__(self, input_dim=(32, 32, 32), num_classes=2):
        super(VoxelCNN, self).__init__()
        d = input_dim[0] # Assuming cubic voxels for simplicity

        # Calculate the size after convolutions and pooling
        # Input: (B, 1, d, d, d) = (B, 1, 32, 32, 32)
        # Conv1 (k=3, s=1, p=1) -> (B, 16, 32, 32, 32)
        # Pool1 (k=2, s=2) -> (B, 16, 16, 16, 16)
        # Conv2 (k=3, s=1, p=1) -> (B, 32, 16, 16, 16)
        # Pool2 (k=2, s=2) -> (B, 32, 8, 8, 8)
        # Conv3 (k=3, s=1, p=1) -> (B, 64, 8, 8, 8)
        # Pool3 (k=2, s=2) -> (B, 64, 4, 4, 4)
        self.final_conv_size = 64 * (d // 8) * (d // 8) * (d // 8) # 64 * 4 * 4 * 4 = 4096 for d=32

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1), # Output: (B, 16, d, d, d)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),              # Output: (B, 16, d/2, d/2, d/2)

            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),# Output: (B, 32, d/2, d/2, d/2)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),              # Output: (B, 32, d/4, d/4, d/4)

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),# Output: (B, 64, d/4, d/4, d/4)
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)               # Output: (B, 64, d/8, d/8, d/8)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.final_conv_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5), # Add dropout for regularization
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, 1, d, d, d)
        x = self.encoder(x)
        # Flatten the output for the linear layer
        x = x.view(x.size(0), -1) # Shape: (batch_size, final_conv_size)
        x = self.classifier(x)   # Shape: (batch_size, num_classes)
        return x
```

```{code} Python
cnn_model = VoxelCNN(input_dim=(voxel_resolution, voxel_resolution, voxel_resolution), num_classes=num_classes_cnn).to(device)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=1e-4) # Might need a smaller learning rate for CNNs
cnn_criterion = nn.CrossEntropyLoss()
```

```{code} Python
train_losses_cnn = []
val_losses_cnn = []
train_accs_cnn = []
val_accs_cnn = []

for epoch in range(cnn_epochs):
    # --- Training Phase ---
    cnn_model.train()
    running_train_loss = 0.0
    train_correct = 0
    train_total = 0
    for inputs, labels in cnn_train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        cnn_optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = cnn_criterion(outputs, labels)
        loss.backward()
        cnn_optimizer.step()

        running_train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    epoch_train_loss = running_train_loss / len(cnn_train_loader.dataset)
    epoch_train_acc = 100 * train_correct / train_total
    train_losses_cnn.append(epoch_train_loss)
    train_accs_cnn.append(epoch_train_acc)

    # --- Validation Phase ---
    cnn_model.eval()
    running_val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in cnn_val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = cnn_model(inputs)
            loss = cnn_criterion(outputs, labels)

            running_val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    epoch_val_loss = running_val_loss / len(cnn_val_loader.dataset)
    epoch_val_acc = 100 * val_correct / val_total
    val_losses_cnn.append(epoch_val_loss)
    val_accs_cnn.append(epoch_val_acc)
```
:::

### K-Nearest Neighbors

We evaluated the K-Nearest Neighbors (KNN) classifier ($k=5$) using various distance metrics between the input partial point clouds and the training samples. We tested three primary distance metrics: Chamfer distance, Masked Chamfer distance (with a threshold of 0.3), and Earth Mover's Distance (EMD). Additionally, we assessed the impact of applying Procrustes analysis to align the point clouds before computing the distance. The classification accuracies on the test set for each configuration are summarized below:

| Distance Metric         | Procrustes Alignment | Test Accuracy |
| :---------------------- | :------------------- | :------------ |
| Chamfer                 | No                   | 0.45          |
| Masked Chamfer ($t=0.3$) | No                   | 0.45          |
| EMD                     | No                   | 0.53          |
| EMD                     | Yes                  | 0.74          |
| Chamfer                 | Yes                  | 0.82          |
| Masked Chamfer ($t=0.3$) | Yes                  | **0.87**      |

**Analysis:**

The results clearly demonstrate the critical importance of rotational and translational alignment for classifying partial point clouds. Without alignment (Procrustes = No), all distance metrics performed poorly, with accuracies barely above random guessing (0.50 for a binary classification task). EMD showed slightly better performance than Chamfer and Masked Chamfer in this baseline case.

Applying Procrustes analysis before calculating the distance significantly improved performance across all metrics. This highlights that the primary challenge in comparing these partial point clouds lies in their arbitrary orientations and positions. Aligning them first allows the distance metrics to capture shape similarity more effectively.

Among the aligned metrics, Masked Chamfer distance achieved the highest accuracy (87%), outperforming both standard Chamfer (82%) and EMD (74%). This suggests that ignoring points that are very far apart after alignment (the masking step) is beneficial for comparing partial observations, potentially filtering out noise or irrelevant parts of the point clouds. The standard Chamfer distance, which considers all points, still performs well, indicating its robustness after alignment. EMD, while significantly improved by alignment, lagged behind the Chamfer-based metrics in this experiment. The superior performance of the Procrustes-aligned Masked Chamfer KNN demonstrates its effectiveness for classifying these specific types of partial 3D shapes.

## 3D Shape Completion

For our second task, we start with the voxel representation of the partial 3D form and attempt to complete the rest of the form. To accomplish this, we utilize Variational Auto-Encoders (VAE). Variatonal autoencoders apply variational inference in order to approximate the distribution of the latent variables given the data (encoding) and the data given the latent variables (decoding), learning the data distribution in the process [@kingma2013auto]. We chose this approach since despite having $d^3$ voxels and features, the actual generation of each example in the dataset is governed by a small number of latent variables controlling the effect of scaling, rotation, translation, and noising is applied.


### Reconstruction

As a precursor to the completion task, we tackle the typical VAE problem, which involves training an encoder that learns the distribution of the latent variables given the full 3D form and a decoder that learns the distribution of the voxels in the full 3D form given the latent variables.

For the architecture, we use a 3D VAE with a CNN Encoder and Decoder. 
Inputs are 32x32x32 voxels, and the encoder has 3 convolutional blocks, with channel size 4, 8 and 16. Between blocks are relu and max pool layers. We then use 2 2 layer MLPs for the mu and log variance respectively, with Relu activations on the first layer and a hidden dimension of 100. 

After sampling from the 128 dim latent space using those parameters, we use a 2 layer decoder MLP with the same specifications as the previous MLPs. Then, we use a convolutional decoder with the same structure as the encoder, but reversed (and with the inverse operations). Then, we end with a sigmoid activation. 

In addition to those things for standard VAEs, we also include a temperature parameter, which affects the variation of the data by influencing the variance. 

We use a Binary crossentropy loss + $\lambda*$KL divergence loss (standard for VAE). We use $\lambda = 10$


:::{dropdown} VAE Code
```{code} Python
class VoxelVAE(nn.Module):
    def __init__(self, latent_dim=128, input_dim=(1, 32, 32, 32)):
        super(VoxelVAE, self).__init__()
        self.latent_dim = latent_dim
       
        k = 3
        pad = math.floor((k - 1)/2)
        #padding = "same"
        padding = pad
        self.d = 4
        d = self.d

        #maybe change to maxpool instead of stride. 
        self.encoder = nn.Sequential(
            nn.Conv3d(1, d, kernel_size=k, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size = k,stride = 2, padding = padding),
            nn.Conv3d(d, 2*d, kernel_size=k, stride=1, padding=padding),
            nn.ReLU(), 
            nn.MaxPool3d(kernel_size = k,stride = 2, padding = padding),
            nn.Conv3d(2*d, 4*d, kernel_size=k, stride=1, padding=padding),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size = k,stride = 2, padding = padding),
            #nn.Conv3d(4*d, 8*d, kernel_size=k, stride=1, padding=padding),
            #nn.ReLU(),
            #nn.MaxPool3d(kernel_size = k,stride = 2, padding = padding),
            #nn.Conv3d(8*d, 8*d, kernel_size = k, stride = 1, padding = padding)
        )
        # TODO: THIS NEEDS TO BE A FUNCTION OF THE INPUT_DIMs?
        self.fc_input_size = 4*d*4*4*4
        hidden_dim = 100
        self.fc_mu = nn.Sequential(
            nn.Linear(self.fc_input_size, hidden_dim), nn.ReLU(),
            #nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        #self.fc_mu = nn.Linear(self.fc_input_size, latent_dim)
        self.fc_logvar = nn.Sequential( 
            nn.Linear(self.fc_input_size, hidden_dim), nn.ReLU(),
            #nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim))
        #self.fc_logvar = nn.Linear(self.fc_input_size, latent_dim)

        # Decoder
        #what on earth does this do. 
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), 
            #nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,self.fc_input_size))
        
        self.decoder = nn.Sequential(
            #nn.ConvTranspose3d(8*d,8*d, kernel_size=k, stride=1, padding=padding), 
            #nn.Upsample(scale_factor = 2),
            #nn.ConvTranspose3d(8*d, 4*d, kernel_size=k, stride=1, padding=padding),
            #nn.ReLU(), 
            nn.Upsample(scale_factor = 2),
            nn.ConvTranspose3d(4*d, 2*d, kernel_size=k, stride=1, padding=padding),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),
            nn.ConvTranspose3d(2*d, d, kernel_size=k, stride=1, padding=padding),
            nn.ReLU(), 
            nn.Upsample(scale_factor = 2),
            nn.ConvTranspose3d(d, 1, kernel_size=k, stride=1, padding=padding),
            nn.Sigmoid()
        )

    def reparametrize(self, mu, logvar, T = 1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * (std*T)

    def forward(self, x, T = 1.0):
        batch_size = x.size(0)
        
        # Encode
        x = self.encoder(x)
        
        x = x.view(batch_size, -1)  # Flatten
        #32 x 256
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        # Reparameterization trick
        z = self.reparametrize(mu, logvar, T)

        # Decode
        x = self.fc_decode(z)
        x = x.view(batch_size, 4*self.d,4,4,4)  # Reshape for transposed convs
       
        x = self.decoder(x)

        return x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
   
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        #print(x.shape)
        #recon_loss = torch.mean(torch.sum((x - recon_x)**2, dim = (1, 2, 3,4))/(torch.sum(x**2, dim = (1,2,3,4))+1e-8))
        #nn.functional.mse_loss(recon_x, x, reduction = "mean")
        #check this. 
        #print((1+ logvar- mu.pow(2) - logvar.exp()).shape)
        #why does this work here. 
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
       
        # why this loss, maybe needs
        #print("recon: ", recon_loss)
        #print("Kl div: ", kl_div)
        return recon_loss + 10*kl_div,recon_loss, kl_div
    def relative_mse_loss(self, x, x_recon, eps=1e-8):
      
        # print(x_recon)
        #print(x.shape)
        #problem with these dimensions. 
        #print(torch.sum(x ** 2, dim=-1))
        #print(torch.sum(x ** 2, dim=-1).shape)

        return torch.mean(torch.sum((x - x_recon) ** 2, dim=-1) / (torch.sum(x ** 2, dim=-1) + eps))
def train_vae(model, dataloader, valdataloader, optimizer, epochs=100, device='cpu'):
    model.to(device)
    model.train()
    print("cudnn benchmark is enabled:", torch.backends.cudnn.benchmark)
    torch.backends.cudnn.benchmark = True

    train_losses = [] # Store training loss per epoch
    val_losses = []   # Store validation loss per epoch
    train_kls = []    # Store training KL divergence per epoch
    val_kls = []      # Store validation KL divergence per epoch

    for epoch in range(epochs):
        total_loss = 0
        batchCount = 0
        klLoss = 0
        for (x,y) in dataloader:
            batchCount+=1

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss, recon, kl = model.loss_function(recon_x, y, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            klLoss += kl.item() # Use .item() to get scalar value

        # Calculate average loss for the epoch
        epoch_train_loss = total_loss / len(dataloader.dataset)
        epoch_train_kl = klLoss / len(dataloader.dataset)
        train_losses.append(epoch_train_loss)
        train_kls.append(epoch_train_kl)

        # Validation loop
        model.eval()
        val_loss = 0.0
        valklLoss = 0.0
        with torch.no_grad():
            for (x,y) in valdataloader:

                x = x.to(device)
                y = y.to(device)

                recon_x, mu, logvar = model(x)
                valLoss, valRecon, valKL = model.loss_function(recon_x, y, mu, logvar)
                valklLoss += valKL.item() # Use .item()
                val_loss += valLoss.item() # Use .item()

        # Calculate average validation loss for the epoch
        epoch_val_loss = val_loss / len(valdataloader.dataset)
        epoch_val_kl = valklLoss / len(valdataloader.dataset)
        val_losses.append(epoch_val_loss)
        val_kls.append(epoch_val_kl)

        print(f"Epoch {epoch+1}, Train: {epoch_train_loss:.4f} Val: {epoch_val_loss:.4f}")
        print(f"Train KL loss: {epoch_train_kl:.4f} Val: {epoch_val_kl:.4f}")
        model.train()

    return train_losses, val_losses, train_kls, val_kls # Return the history lists

# No completion, just learn spheres and cubes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VoxelVAE(input_dim=(1, d, d, d))
#model.to(device)
#randomize beforehand please to vary which data is in train test and val. 
#shuffle before splitting
rng = np.random.default_rng()
#changes inplace
combinedData = np.stack([inputVoxels, trueVoxels], axis=-1)
rng.shuffle(combinedData, axis=0)
trainInput, other = np.split(combinedData, [int(0.8 * inputVoxels.shape[0])])

trainx = trainInput[:, :, :, :, 0]
#trainx = trainInput[:, :, :, :,  1]
trainy = trainInput[:, :, :, :,  1]

valInput, testInput = np.split(other, [other.shape[0]//2])
valx = valInput[:, :, :, :, 0]
#valx = valInput[:, :, :, :, 1]
valy = valInput[:, :, :, :, 1]



dataloader = DataLoader(TensorDataset(torch.tensor(trainx).float().unsqueeze(1), torch.tensor(trainy).float().unsqueeze(1)), batch_size=100, shuffle=True, pin_memory = True, num_workers= 8) # unsqueeze is needed to make dimensions correct, but why do we have the (1,d,d,d) instead of (d,d,d)
valdataloader = DataLoader(TensorDataset(torch.tensor(valx).float().unsqueeze(1), torch.tensor(valy).float().unsqueeze(1)), batch_size = 100, shuffle = True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```
:::


### Completion

After learning the distribution of the 3D forms, we then modify our training dataset to take an incomplete 3D forms as input and output the full 3D form. Thus, the encoder learns the distribution of latent variables conditioned on the partial voxel image and the decoder learns the distribution of voxels in the full voxel image. The architecture remains the same, but the input to the VAE is changed. 

# Results
## Classification

### Convolutional Neural Network

#### Training and Results

The CNN was trained using the Adam optimizer with a learning rate $\eta = 1 \times 10^{-4}$ and minimized the Cross-Entropy loss. The dataset was partitioned into 80% training and 20% validation samples. Training proceeded for 50 epochs using a batch size $B=64$. The training and validation loss and accuracy curves are presented below.

```{figure} images/cnn_loss_accuracy_curves.png
:name: cnn_loss_accuracy_curves

Training and validation loss ($L$) and accuracy (Acc) curves for the CNN classifier over 50 epochs.
```

The results show effective learning on the training set, with $L_{train} \to 0$ and $Acc_{train} \to 100\%$. However, the validation performance indicates overfitting. The validation loss $L_{val}$ diverges from $L_{train}$ after approximately 10 epochs, starting to increase while $L_{train}$ continues to decrease. Concurrently, the validation accuracy $Acc_{val}$ plateaus around $75-76\%$, failing to generalize as well as the training performance suggests. This divergence confirms that the model learned training set idiosyncrasies that did not transfer to unseen validation data. The final $Acc_{val}$ after 50 epochs was approximately 76%.


### K-Nearest Neighbors

We evaluated the K-Nearest Neighbors (KNN) classifier ($k=5$) using various distance metrics between the input partial point clouds and the training samples. We tested three primary distance metrics: Chamfer distance, Masked Chamfer distance (with a threshold of 0.3), and Earth Mover's Distance (EMD). Additionally, we assessed the impact of applying Procrustes analysis to align the point clouds before computing the distance. The classification accuracies on the test set for each configuration are summarized below:

| Distance Metric         | Procrustes Alignment | Test Accuracy |
| :---------------------- | :------------------- | :------------ |
| Chamfer                 | No                   | 0.45          |
| Masked Chamfer ($t=0.3$) | No                   | 0.45          |
| EMD                     | No                   | 0.53          |
| EMD                     | Yes                  | 0.74          |
| Chamfer                 | Yes                  | 0.82          |
| Masked Chamfer ($t=0.3$) | Yes                  | **0.87**      |


The results demonstrate the critical importance of rotational and translational alignment for classifying partial point clouds. Without alignment (Procrustes = No), all distance metrics performed poorly, with accuracies barely above random guessing (0.50 for a binary classification task). EMD showed slightly better performance than Chamfer and Masked Chamfer in this baseline case.

Applying Procrustes analysis before calculating the distance significantly improved performance across all metrics. This highlights that the primary challenge in comparing these partial point clouds lies in their arbitrary orientations and positions. Aligning them first allows the distance metrics to capture shape similarity more effectively.

Among the aligned metrics, Masked Chamfer distance achieved the highest accuracy (87%), outperforming both standard Chamfer (82%) and EMD (74%). This suggests that ignoring points that are very far apart after alignment (the masking step) is beneficial for comparing partial observations, potentially filtering out noise or irrelevant parts of the point clouds. The standard Chamfer distance, which considers all points, still performs well, indicating its robustness after alignment. EMD, while significantly improved by alignment, lagged behind the Chamfer-based metrics in this experiment. The superior performance of the Procrustes-aligned Masked Chamfer KNN demonstrates its effectiveness for classifying these specific types of partial 3D shapes.

## 3D Shape Completion

  

### Reconstruction

  

```{figure} images/reconLossC.png

Loss over epochs for training the VAE to do reconstruction

```
```{figure} images/reconImages.png
{figure} images/reconImages2.png

Qualitative comparison between a couple examples of the full 3D representation and the VAE reconstruction

```

The training and validation loss curves for the VAE reconstruction task demonstrate successful convergence. Both losses decrease rapidly initially and then plateau, indicating that the model learned to effectively reconstruct the target voxel shapes from the input latent representation. The small, stable gap between the training and validation loss suggests minimal overfitting.

Qualitatively, the reconstructions shown in the figure are visually very similar to the original ground truth voxels. The VAE effectively captures the overall shape and extent of the objects. Minor differences are observable, particularly a tendency for the reconstructed shapes to appear slightly more "rounded" or smoother than the originals. This could potentially stem from the continuous nature of the latent space, the prevalence of spherical/ellipsoidal shapes in the training data influencing the learned prior, or the inherent smoothing effect of the convolutional decoder. Despite these subtle variations, the high fidelity of the reconstructions confirms the VAE's capability in learning a compressed representation and accurately decoding it back to the voxel space.

It is important to note that these visualizations employ a threshold (0.3); only voxels with a predicted occupancy probability above this value are displayed. Varying this threshold would alter the visual appearance by reflecting different confidence levels in the reconstruction.



### Completion

  
  

```{figure} images/lossCurveC.png

Loss over epochs for training the VAE to do reconstruction

```


  

```{figure} images/vaeImages.png

Qualitative comparison between a couple examples of the partial, noisy 3D form, full 3D representation, and the VAE completion

```

The VAE was subsequently trained on the completion task, using partial voxel grids as input and the corresponding complete voxel grids as the target output. The loss curves  show that the training loss steadily decreases, while the validation loss decreases initially but begins to diverge upwards after approximately 75 epochs. This indicates that the model starts to overfit to the training data beyond this point, although it still learns the general task of completion.

Qualitative results are presented in the above figure. The generated completions successfully infer the overall shape and orientation from the partial, noisy input. For instance, the elongated nature of the partial input is reflected in the completed ellipsoidal form. The model produces plausible completions that respect the input conditioning. However, similar to the reconstruction task, the completed shapes tend to be smoother and more rounded compared to the ground truth, particularly noticeable when the target is a parallelepiped (though the example shows an ellipsoid). This might reflect ambiguity inherent in completing shapes from partial data, properties of the VAE's latent space, or the smoothing effect of the decoder. The difference between the two generated samples ('Generated 1' and 'Generated 2') for the same input highlights the stochastic nature of the VAE, sampling different plausible completions from the learned latent distribution.

<!-- As with reconstruction, the visualizations use a threshold of 0.3 for displaying occupied voxels. -->

# Conclusion

In this paper, we demonstrate methods to effectively deal with noisy and incomplete 3D representations, in both classification and completion tasks. We achieve reasonable performance in classification, with best methods reaching nearly 80% accuracy. In the reconstruction and completion tasks, we demonstrate the viability of using VAEs in this completion tasks, with reasonable qualitative completions of point clouds. While our analysis is limited to only ellipsoids and parallelipipeds, we demonstrate a proof of concept for further classification and completion of additional more complex forms.


## Future Work

There are many directions that would make sense as a follow up to this project. The most obvious approach is to add more complex forms. While our approach works for ellipsoids and parallelipeds, it is unclear how well it will generate to forms that are more complicated. Thus, a good place to start is the ShapeNet [@shapenet] dataset which includes 3D representations of many everyday objects such as laptops, benches, chairs, etc. This will allow us to get a better evaluation of our methods on more realistic data. 

Next, we can try more sophisticated generative models for the completion of point clouds. In particular, diffusion models have been the standard for image generation tasks [@diffusionimagesurvey]. Given the similarity of generation of 2D pixel images and 3D voxel shapes, this is a reasonable alternative approach to VAEs. Additionally, other architectures such as autoregressive generative models, like Point Transformers [@pointtransformer], can work directly with point clouds, circumventing the loss of information and large computational costs of sparse surfaces that result from working with voxels. 

Finally, while the classification and generative modeling tasks seem fairly separate, they can be combined with certain architectures, such as diffusion, in order to make conditional generative models through classifier guidance [@song2020score]. 


