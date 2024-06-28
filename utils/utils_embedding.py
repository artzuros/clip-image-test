import clip
import pickle, os
import streamlit as st
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def load_embeddings(image_folder, embeddings_path):
    if os.path.isdir(image_folder):
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                st.session_state.embeddings = pickle.load(f)
            st.success("Embeddings loaded successfully.")
        else:
            st.error("Embeddings not found. Please generate them first.")
    else:
        st.error("Invalid folder path.")

# def load_embeddings(file_path):
#     """
#     Loads the embeddings from the given file path.
#     Args:
#         file_path (str): The file path to load the embeddings from.

#     Returns:
#         dict: A dictionary containing the embeddings.
#     """
#     with open(file_path, 'rb') as f:
#         embeddings = pickle.load(f)
#     return embeddings

def get_text_embedding(text):
    """
    Returns the CLIP embedding for the given text.
    Args:
        text (str): The text to get the embedding for.
        
    Returns:
        numpy.ndarray: The CLIP embedding for the given text.
    """
    text_inputs = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

import os
from PIL import Image
import torch

def generate_clip_embeddings(image_folder):
    """
    Generates CLIP embeddings for the images in the given folder and its subfolders.
    Args:
        image_folder (str): The folder containing the images.
        
    Returns:
        dict: A dictionary containing the CLIP embeddings for the images.
    """
    embeddings = {}

    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(root, file)
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                embeddings[image_path] = image_features.cpu().numpy()
    
    return embeddings
