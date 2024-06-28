import streamlit as st
import torch
from PIL import Image
import clip
import pickle
import os
import csv
import datetime
from utils.utils_embedding import load_embeddings, get_text_embedding, generate_clip_embeddings
from utils.utils_similarity import compute_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'sorted_similarities' not in st.session_state:
    st.session_state.sorted_similarities = None
if 'selected_images' not in st.session_state:
    st.session_state.selected_images = []

def load_images(image_folder):
    image_paths = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def perform_search():
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

def select_all_images():
    st.session_state.selected_images = [image_name for image_name, _ in st.session_state.sorted_similarities]

def deselect_all_images():
    st.session_state.selected_images = []

def export_to_csv():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'selected_images_{timestamp}.csv'
    with open(os.path.join(image_folder, csv_filename), 'w', newline='') as csvfile:
        fieldnames = ['query', 'image_path', 'similarity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for image_name in st.session_state.selected_images:
            similarity = dict(st.session_state.sorted_similarities).get(image_name, "N/A")
            writer.writerow({'query': st.session_state.query, 'image_path': image_name, 'similarity': similarity})
    st.success(f"Selected images exported to {csv_filename}")

if 'query' not in st.session_state:
    st.session_state.query = ""

image_folder = st.text_input("Enter the FULL path of the images folder:")
embeddings_path = os.path.join(image_folder, 'image_embeddings.pkl')

if st.button("Load Embeddings"):
    image_paths = load_images(image_folder)
    st.session_state.embeddings = load_embeddings(image_paths, embeddings_path)

if st.button("Generate Embeddings"):
    if os.path.isdir(image_folder):
        st.session_state.embeddings = generate_clip_embeddings(image_folder)
        with open(embeddings_path, 'wb') as f:
            pickle.dump(st.session_state.embeddings, f)
        st.success(f"Embeddings generated and saved successfully in {embeddings_path}")
    else:
        st.error("Invalid folder path.")

query_input = st.text_input("Enter your search query:", value=st.session_state.query, key="query_input")

if st.session_state.query != query_input:
    st.session_state.query = query_input
    perform_search()

if st.button("Search"):
    perform_search()

if st.session_state.sorted_similarities:
    st.write("Top matching images:")
    
    total_images_to_show = st.slider("Total images to show", 20, 200, 20)
    total_pages = (total_images_to_show - 1) // 20 + 1
    page_number = st.number_input("Page number", min_value=1, max_value=total_pages, value=1, step=1)
    
    start_index = (page_number - 1) * 20
    end_index = start_index + 20

    if st.button("Select All Images"):
        select_all_images()
    if st.button("Deselect All Images"):
        deselect_all_images()

    cols = st.columns(4)
    for idx, (image_name, similarity) in enumerate(st.session_state.sorted_similarities[start_index:end_index]):
        col = cols[idx % 4]
        image = Image.open(image_name)
        col.image(image, use_column_width=True)
        col.write(f"{image_name} ({similarity:.3f})")
        if col.button(f"Select", key=f"select_{image_name}"):
            select_image(image_name)
        if col.button(f"Deselect", key=f"deselect_{image_name}"):
            deselect_image(image_name)

if st.session_state.selected_images:
    st.write("Selected images to export:")
    for image_name in st.session_state.selected_images:
        st.write(image_name)

    if st.button("Export to CSV"):
        export_to_csv()
