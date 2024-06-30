from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def compute_similarity_torch(query_embedding, image_embeddings):
    """
    Returns the cosine similarity between the query embedding and the image embeddings.
    Args:
        query_embedding (numpy.ndarray): The query embedding.
        image_embeddings (dict): A dictionary containing the image embeddings.
    
    Returns:
        dict: A dictionary containing the cosine similarity between the query embedding and the image embeddings. 
    """
    # Stack all image embeddings into a single matrix
    image_names = list(image_embeddings.keys())
    image_embedding_matrix = np.vstack([image_embeddings[name] for name in image_names])

    # Compute cosine similarity between the query embedding and all image embeddings in a batch
    similarities = cosine_similarity(query_embedding, image_embedding_matrix)
    
    # Map similarities to the corresponding image names
    similarities_dict = {image_names[i]: similarities[0][i] for i in range(len(image_names))}
    
    return similarities_dict
