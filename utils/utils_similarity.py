from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(query_embedding, image_embeddings):
    """
    Returns the cosine similarity between the query embedding and the image embeddings.
    Args:
        query_embedding (numpy.ndarray): The query embedding.
        image_embeddings (dict): A dictionary containing the image embeddings.
    
    Returns:
        dict: A dictionary containing the cosine similarity between the query embedding and the image embeddings. 
    """
    similarities = {}
    for image_name, image_embedding in image_embeddings.items():
        similarity = cosine_similarity(query_embedding, image_embedding)
        similarities[image_name] = similarity[0][0]
    return similarities