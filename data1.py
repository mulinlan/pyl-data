import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fix random seed for reproducibility
np.random.seed(5)

# Generate dataset
x_train = np.random.random((1000, 12))
y_train = np.random.randint(2, size=(1000, 1))

# Select first two features for 2D plotting
x_train_2d = x_train[:, :2]

# Select first three features for 3D plotting
x_train_3d = x_train[:, :3]

# Generate 2D scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x_train_2d[y_train.flatten() == 0, 0], x_train_2d[y_train.flatten() == 0, 1], label='Class 0', alpha=0.6)
plt.scatter(x_train_2d[y_train.flatten() == 1, 0], x_train_2d[y_train.flatten() == 1, 1], label='Class 1', alpha=0.6)
plt.title("2D Scatter Plot")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Generate 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_train_3d[y_train.flatten() == 0, 0], x_train_3d[y_train.flatten() == 0, 1],
           x_train_3d[y_train.flatten() == 0, 2], label='Class 0', alpha=0.6)
ax.scatter(x_train_3d[y_train.flatten() == 1, 0], x_train_3d[y_train.flatten() == 1, 1],
           x_train_3d[y_train.flatten() == 1, 2], label='Class 1', alpha=0.6)
ax.set_title("3D Scatter Plot")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
ax.legend()
plt.show()
