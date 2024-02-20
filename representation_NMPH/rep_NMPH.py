"""
now I have 6 10000x1000 matrices, each of them is a representation of 10000 images with 1000 features.
matrix path="/gpfs/milgram/project/turk-browne/projects/SoftHebb/representation_activations_{0/10/20/30/40/49}.npy"
6 matrices are from 6 different time points.
I want to select the same 100 images from each matrix
For these 100 images, calculate their pairwise correlations so that I have 6 100x100 matrices
create a function taking two 100x100 matrices.
This function will calculate the difference between the two matrices and use this as the y axis of a scatter NMPH plot.
the x axis is the value of the first matrix. now create a NMPH scatter plot based on x and y axis.
"""

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

# Load matrices
matrices = []
available_time_points = 600
for ii in range(0, available_time_points, 10):
    matrix_path = (f"/gpfs/milgram/scratch60/turk-browne/kp578/softHebb/result"
                   f"/representation_{ii}.npy")
    matrix = np.load(matrix_path)
    matrices.append(matrix)

# Select the same 100 images from each matrix
selected_indices = np.random.choice(10000, size=100, replace=False)

selected_matrices = [matrix[selected_indices, :] for matrix in matrices]

# Calculate pairwise correlations for each time point
correlation_matrices = [np.corrcoef(matrix.T) for matrix in selected_matrices]


# Define the function to calculate the difference between two matrices
def calculate_difference(matrix1, matrix2):
    return matrix1 - matrix2


# Create a scatter plot based on x and y axis
def create_scatter_plot(matrix1, matrix2, title=None):
    # Take the upper triangle of the matrices
    upper_triangle_indices = np.triu_indices(matrix1.shape[0], k=1)

    differences = calculate_difference(matrix2, matrix1)

    plt.figure(figsize=(10, 8))
    plt.scatter(matrix1[upper_triangle_indices].flatten(),
                differences[upper_triangle_indices].flatten(),
                alpha=0.5, s=1)
    plt.title(title)
    plt.xlabel('Matrix 1 Values')
    plt.ylabel('Matrix Differences')
    plt.show()


# Create scatter plots for each time point
for i in range(len(correlation_matrices) - 1):
    create_scatter_plot(correlation_matrices[i], correlation_matrices[i + 1],
                        title=f'Time Point {i} vs. Time Point {i + 1}')


def trash():
    # Store activations for the selected indices
    selected_activations = model.representation[:,
                           selected_indices].cpu().numpy()  # model.representation.shape: [1000, 24576]
    representation_activations.append(selected_activations)

    # Convert the list of representation activations to a numpy array
    representation_activations = np.concatenate(representation_activations, axis=0)

    # Now you can save or further analyze the representation_activations as needed
    np.save(f'./result/representation_activations_{epoch}.npy', representation_activations)