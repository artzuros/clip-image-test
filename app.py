import streamlit as st
import torch
from PIL import Image
import clip

from utils.utils_embedding import load_embeddings, get_text_embedding
from utils.utils_similarity import compute_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

st.title("CLIP Image Search")

query = st.text_input("Enter your search query:")
embedding_path = st.text_input("Enter the FULL path of the embeddings file:")

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'sorted_similarities' not in st.session_state:
    st.session_state.sorted_similarities = None

def perform_search():
    embeddings = load_embeddings(embedding_path)
    if query and embeddings:
        query_embedding = get_text_embedding(query)
        similarities = compute_similarity(query_embedding, embeddings)
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        st.session_state.embeddings = embeddings
        st.session_state.sorted_similarities = sorted_similarities

if st.button("Search"):
    perform_search()

if st.session_state.sorted_similarities:
    st.write("Top matching images:")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Image Grid")
        cols = st.columns(4)
        for idx, (image_name, similarity) in enumerate(st.session_state.sorted_similarities[:20]):
            col = cols[idx % 4]
            image = Image.open(f"test/{image_name}")
            col.image(image, caption=f"{image_name}\nSimilarity: {similarity:.4f}", use_column_width=True)
            if col.button(f"Select {image_name}", key=f"select_{image_name}"):
                st.session_state.selected_image_name = image_name

if 'selected_image_name' in st.session_state:
    selected_image_name = st.session_state.selected_image_name
    with col2:
        st.write("Selected Image")
        image = Image.open(f"test/{selected_image_name}")
        st.image(image, caption=selected_image_name, use_column_width=True)
