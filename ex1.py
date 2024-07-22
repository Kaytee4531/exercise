#length of vector
import numpy as np
def compute_vector_length(vector):
    sq_len_of_vector = 0
    for i in range(len(vector)):
        sq_len_of_vector += vector[i]**2
    len_of_vector = np.sqrt(sq_len_of_vector)
    return len_of_vector

#dot product
def compute_dot_product(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    return dot_product

#multiply vector and matrix
def matrix_multi_vector(matrix, vector):
    vector_ans = np.dot(matrix, vector)
    return vector_ans

#multiplu matrix and matrix
def matrix_multi_matrix(matrix1, matrix2):
    matrix_ans = np.dot(matrix1, matrix2)
    return matrix_ans

#matrix inverse
def inverse_matrix(matrix):
    inverse = np.linalg.inv(matrix)
    return inverse