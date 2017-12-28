import scipy

def euclidean(u, v):    
    """Eculidean distance between 1d np.arrays `u` and `v`, which must 
    have the same dimensionality. Returns a float."""
    # Use scipy's method:
    return scipy.spatial.distance.euclidean(u, v)
    # Or define it yourself:
    # return vector_length(u - v)

def cosine(u, v):        
    """Cosine distance between 1d np.arrays `u` and `v`, which must have 
    the same dimensionality. Returns a float."""
    # Use scipy's method:
    return scipy.spatial.distance.cosine(u, v)
    # Or define it yourself:
    # return 1.0 - (np.dot(u, v) / (vector_length(u) * vector_length(v)))

def matching(u, v):    
    """Matching coefficient between the 1d np.array vectors `u` and `v`, 
    which must have the same dimensionality. Returns a float."""
    # The scipy implementation is for binary vectors only. 
    # This version is more general.
    return np.sum(np.minimum(u, v))

def jaccard(u, v):    
    """Jaccard distance between the 1d np.arrays `u` and `v`, which must 
    have the same dimensionality. Returns a float."""
    # The scipy implementation is for binary vectors only. 
    # This version is more general.
    return 1.0 - (matching(u, v) / np.sum(np.maximum(u, v)))