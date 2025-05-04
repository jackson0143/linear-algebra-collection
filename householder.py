import numpy as np

'''
Householder reflection is a linear algebra technique used to make a matrix more simple to work with by zeroing out elements
below the diagonal.
It is commonly used in QR decomposition and other matrix factorization methods.

Input is an nxn matrix, output is the Householder matrix Q1.
'''
def householder_reflect(matrix):
    # input validating
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input should be a numpy array.")
    if len(matrix.shape) != 2:
        raise ValueError("Input must be a 2D matrix.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square (nxn).")
    
    m, n = matrix.shape
 
    
    # Grab first column of matrix a1 and compute the norm
    a1 = matrix[:, 0]
    norm_a1 = np.linalg.norm(a1)

    # Construct Q1a1 and choose a sign for the value to ensure numerical stability
    sign = -1 if a1[0] > 0 else 1
    q1a1 = np.zeros_like(a1)
    q1a1[0] = sign * norm_a1
    
    # Formula for v1 is v1 = a1 - t1
    # where t1 is Q1a1

    v1 = a1 - q1a1
   
    # u1 is the normalized vector of v1 
    # u1 = v1 / ||v1||
    u1 = v1 / np.linalg.norm(v1)
    
    # Construct the Householder matrix Q1
    # Q1 = I - 2uu^T
    Q1 = np.identity(m) - 2 * np.outer(u1, u1)
    return Q1
matrix = np.array([[8,10], [6,20]])

print(householder_reflect(matrix))