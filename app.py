import streamlit as st
import torch
import clip
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

embeddings = load_embeddings('D:/C_Drive/Desktop/CS/clip-image-test/image_embeddings.pkl')

def get_text_embedding(text):
    text_inputs = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy()

def compute_similarity(query_embedding, image_embeddings):
    similarities = {}
    for image_name, image_embedding in image_embeddings.items():
        similarity = cosine_similarity(query_embedding, image_embedding)
        similarities[image_name] = similarity[0][0]
    return similarities

st.title("CLIP Image Search")

query = st.text_input("Enter your search query:")

if query:
    query_embedding = get_text_embedding(query)
    similarities = compute_similarity(query_embedding, embeddings)
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    st.write("Top matching images:")
    for image_name, similarity in sorted_similarities[:5]:  # Display top 5 matches
        image = Image.open(f"test/{image_name}")
        st.image(image, caption=f"Image: {image_name}, Similarity: {similarity:.4f}")

