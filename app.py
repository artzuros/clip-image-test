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





import clip
import pickle
import os
import streamlit as st
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def load_or_generate_embeddings(image_folder, embeddings_path):
    if os.path.isdir(image_folder):
        image_names = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                existing_embeddings = pickle.load(f)
            
            existing_image_names = set(existing_embeddings.keys())
            current_image_names = set(image_names)
            
            # Check for new or missing images
            if existing_image_names != current_image_names:
                st.warning("Discrepancy found between existing embeddings and current images. Updating embeddings.")
                new_embeddings = generate_clip_embeddings(image_folder)
                existing_embeddings.update(new_embeddings)
                
                with open(embeddings_path, 'wb') as f:
                    pickle.dump(existing_embeddings, f)
                st.session_state.embeddings = existing_embeddings
                st.success("Embeddings updated successfully.")
            else:
                st.session_state.embeddings = existing_embeddings
                st.success("Embeddings loaded successfully.")
        else:
            st.warning("Embeddings not found. Generating new embeddings.")
            embeddings = generate_clip_embeddings(image_folder)
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
            st.session_state.embeddings = embeddings
            st.success("Embeddings generated successfully.")
    else:
        st.error("Invalid folder path.")

def load_embeddings(file_path):
    """
    Loads the embeddings from the given file path.
    Args:
        file_path (str): The file path to load the embeddings from.

    Returns:
        dict: A dictionary containing the embeddings.
    """
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

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

def generate_clip_embeddings(image_folder):
    """
    Generates CLIP embeddings for the images in the given folder.
    Args:
        image_folder (str): The folder containing the images.
        
    Returns:
        dict: A dictionary containing the CLIP embeddings for the images.
    """
    embeddings = {}
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        embeddings[image_name] = image_features.cpu().numpy()
    
    return embeddings