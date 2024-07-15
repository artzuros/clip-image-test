from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import torch
import clip
# from transformers import CLIPModel, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# preprocess = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

def compute_similarity(query_embedding, image_embeddings):
    """
    Returns the cosine similarity between the query embedding and the image embeddings.
    Args:
        query_embedding (numpy.ndarray): The query embedding.
        image_embeddings (dict): A dictionary containing the image embeddings.
    
    Returns:
        dict: A dictionary containing the cosine similarity between the query embedding and the image embeddings. 
    """
    image_names = list(image_embeddings.keys())
    image_embedding_matrix = np.vstack([image_embeddings[name] for name in image_names])

    similarities = cosine_similarity(query_embedding, image_embedding_matrix)
    similarities_dict = {image_names[i]: similarities[0][i] for i in range(len(image_names))}
    
    return similarities_dict

def compute_similarity_image(uploaded_image, image_embeddings):
    """
    Returns the cosine similarity between the uploaded image embedding and the image embeddings.
    Args:
        uploaded_image_embedding (numpy.ndarray): The uploaded image embedding.
        image_embeddings (dict): A dictionary containing the image embeddings.
    
    Returns:
        dict: A dictionary containing the cosine similarity between the uploaded image embedding and the image embeddings. 
    """
    image = preprocess(Image.open(uploaded_image)).unsqueeze(0).to(device)
    # image = uploaded_image
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().numpy()

    #calculate cosine similarity and return the sorted similarities
    image_names = list(image_embeddings.keys())
    image_embedding_matrix = np.vstack([image_embeddings[name] for name in image_names])
    similarities = cosine_similarity(image_features, image_embedding_matrix)
    similarities_dict = {image_names[i]: similarities[0][i] for i in range(len(image_names))}
    sorted_similarities = sorted(similarities_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities
