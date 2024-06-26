import streamlit as st
import torch
from PIL import Image
import clip
import pickle
import os

from utils.utils_embedding import load_or_generate_embeddings, get_text_embedding, generate_clip_embeddings
from utils.utils_similarity import compute_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'sorted_similarities' not in st.session_state:
    st.session_state.sorted_similarities = None
if 'selected_image_name' not in st.session_state:
    st.session_state.selected_image_name = None

image_folder = st.text_input("Enter the FULL path of the images folder:")
embeddings_path = os.path.join(image_folder, 'image_embeddings.pkl')

if st.button("Load Embeddings"):
    load_or_generate_embeddings(image_folder, embeddings_path)

if st.button("Generate Embeddings"):
    if os.path.isdir(image_folder):
        st.session_state.embeddings = generate_clip_embeddings(image_folder)
        with open(embeddings_path, 'wb') as f:
            pickle.dump(st.session_state.embeddings, f)
        st.success(f"Embeddings generated and saved successfully in {embeddings_path}")
    else:
        st.error("Invalid folder path.")

query = st.text_input("Enter your search query:")

def perform_search():
    if query and st.session_state.embeddings:
        query_embedding = get_text_embedding(query)
        similarities = compute_similarity(query_embedding, st.session_state.embeddings)
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        st.session_state.sorted_similarities = sorted_similarities

if st.button("Search"):
    perform_search()

if st.session_state.sorted_similarities:
    st.write("Top matching images:")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        cols = st.columns(4)
        for idx, (image_name, similarity) in enumerate(st.session_state.sorted_similarities[:20]):
            col = cols[idx % 4]
            image = Image.open(os.path.join(image_folder, image_name))
            col.image(image, use_column_width=True)
            if col.button(f"Select", key=f"select_{image_name}"):
                st.session_state.selected_image_name = image_name

    if 'selected_image_name' in st.session_state:
        selected_image_name = st.session_state.selected_image_name
        with col2:
            image = Image.open(os.path.join(image_folder, selected_image_name))
            st.image(image, caption=selected_image_name, use_column_width=True)
            st.write(f"Image Path: {os.path.join(image_folder, selected_image_name)}")
            similarity = dict(st.session_state.sorted_similarities).get(selected_image_name, "N/A")
            st.write(f"Similarity: {similarity}")
