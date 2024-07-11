import streamlit as st
import datetime, csv, os
from utils.utils_embedding import get_text_embedding
from utils.utils_similarity import compute_similarity
@st.cache_data
def load_embeddings_cached(image_folder, embeddings_path):
    return load_embeddings(image_folder, embeddings_path)

@st.cache_data
def generate_clip_embeddings_cached(image_folder):
    return generate_clip_embeddings(image_folder)

def perform_search():
    """Performs the search based on the query and the embeddings."""
    if 'query' in st.session_state and st.session_state.query and st.session_state.embeddings:
        query_embedding = get_text_embedding(st.session_state.query)
        similarities = compute_similarity(query_embedding, st.session_state.embeddings)
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        st.session_state.sorted_similarities = sorted_similarities
def perform_search():
    """ Performs the search based on the query and the embeddings.
    
    Args:
        query (string): The search query.
        embeddings (dict): The image embeddings.
        
    Returns:
        dict: A dictionary containing the cosine similarity between the query and the image embeddings.
    """
    if 'query' in st.session_state and st.session_state.query and st.session_state.embeddings:
        query_embedding = get_text_embedding(st.session_state.query)
        similarities = compute_similarity(query_embedding, st.session_state.embeddings)
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        st.session_state.sorted_similarities = sorted_similarities

def select_image(image_name):
    if image_name not in st.session_state.selected_images:
        st.session_state.selected_images.append(image_name)

def deselect_image(image_name):
    if image_name in st.session_state.selected_images:
        st.session_state.selected_images.remove(image_name)

def select_all_images(start_index, end_index):
    for image_name, _ in st.session_state.sorted_similarities[start_index:end_index]:
        if image_name not in st.session_state.selected_images:
            st.session_state.selected_images.append(image_name)

def deselect_all_images():
    st.session_state.selected_images = []
    
def export_to_csv(image_folder):
    """ Exports the selected images to a CSV file.
    
    Args:
        image_folder (string): The folder containing the images.
        
    Returns:
        string: The name of the CSV file.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'selected_images_{timestamp}.csv'
    with open(os.path.join(image_folder, csv_filename), 'w', newline='') as csvfile:
        fieldnames = ['query', 'image_path', 'similarity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for image_name in st.session_state.selected_images:
            similarity = dict(st.session_state.sorted_similarities).get(image_name, "N/A")
            writer.writerow({'query': st.session_state.query, 'image_path': os.path.join(image_folder, image_name), 'similarity': similarity})
    st.success(f"Selected images exported to {csv_filename}")