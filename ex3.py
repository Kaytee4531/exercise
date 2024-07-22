#cosine similarity
import numpy as np
from numpy import dot
from numpy.linalg import norm
def compute_cosine(v1, v2):
    cosine = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return cosine