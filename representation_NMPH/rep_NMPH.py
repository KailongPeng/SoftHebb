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
for i in range(6):
    if i == 5:
        matrix_path = f"/gpfs/milgram/project/turk-browne/projects/SoftHebb/representation_activations_{49}.npy"
    else:
        matrix_path = f"/gpfs/milgram/project/turk-browne/projects/SoftHebb/representation_activations_{i * 10}.npy"
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
def create_scatter_plot(matrix1, matrix2):
    differences = calculate_difference(matrix1, matrix2)

    plt.figure(figsize=(10, 8))
    plt.scatter(matrix1.flatten(), differences.flatten(), alpha=0.5)
    plt.title('Scatter Plot of Matrix Values vs. Differences')
    plt.xlabel('Matrix 1 Values')
    plt.ylabel('Matrix Differences')
    plt.show()


# Create scatter plots for each time point
# for i in range(5):
i = 1
create_scatter_plot(correlation_matrices[i], correlation_matrices[i + 1])
differences = calculate_difference(correlation_matrices[i], correlation_matrices[i + 1])