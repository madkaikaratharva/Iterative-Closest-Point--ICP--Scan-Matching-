import numpy as np

def mean_dist(points1, points2):
    """
    Returns the mean Euclidean distance between corresponding points in the two point clouds.
    """
    distances = []
    for i in range(points1.shape[0]):
        difference = points1[i] - points2[i]
        distance = np.linalg.norm(difference)
        distances.append(distance)
    
    return np.mean(np.array(distances))