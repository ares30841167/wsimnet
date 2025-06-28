import numpy as np
from scipy.spatial import distance


# Calculate the cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


# Calculate the euclidean distance between two vectors
def euclidean_distance(x, y):
    return distance.euclidean(x, y)


# Calculate the similarity score based on the L2 norm like elastic search
def elastic_like_euclidean_similarity_score(x, y):
    return 1 / (1 + np.linalg.norm(y - x)**2)


# Calculate the similarity score based on the cosine like elastic search
def elastic_like_cosine_similarity_score(query, vector):
    dot_product = np.dot(query, vector)
    norm_query = np.linalg.norm(query)
    norm_vector = np.linalg.norm(vector)
    
    cosine_sim = dot_product / (norm_query * norm_vector)
    
    return (1 + cosine_sim) / 2
