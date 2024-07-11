import streamlit as st
import torch
from PIL import Image
import clip
import pickle
import os
from utils.utils_embedding import load_embeddings, generate_clip_embeddings, update_embeddings
from utils.utils import perform_search, select_all_images, deselect_all_images, export_to_csv
from utils.utils_similarity import compute_similarity_image
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

@st.cache_data
def load_embeddings_cached(image_folder, embeddings_path):
    return load_embeddings(image_folder, embeddings_path)

@st.cache_data
def generate_clip_embeddings_cached(image_folder):
    return generate_clip_embeddings(image_folder)


if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'sorted_similarities' not in st.session_state:
    st.session_state.sorted_similarities = None
if 'selected_images' not in st.session_state:
    st.session_state.selected_images = []


if 'query' not in st.session_state:
    st.session_state.query = ""

if 'image' not in st.session_state:
    st.session_state.image = None

image_folder = st.text_input("Enter the FULL path of the images folder:")
embeddings_path = os.path.join(image_folder, 'image_embeddings.pkl')

col_load, col_generate = st.columns(2)
with col_load:
    if st.button("Load Embeddings"):
        load_embeddings_cached(image_folder, embeddings_path)
        
with col_load:
    if st.button("Update Embeddings"):
        update_embeddings(image_folder, embeddings_path)
        
with col_generate:
    if st.button("Generate Embeddings"):
        if os.path.isdir(image_folder):
            st.session_state.embeddings = generate_clip_embeddings_cached(image_folder)
            with open(embeddings_path, 'wb') as f:
                pickle.dump(st.session_state.embeddings, f)
            st.success(f"Embeddings generated and saved successfully in {embeddings_path}")
        else:
            st.error("Invalid folder path.")
# Select between text or image search
search_type = st.radio("Select search type:", ["Text", "Image"])
if search_type == "Text":
    query_input = st.text_input("Enter your search query:", value=st.session_state.query, key="query_input")

    if st.session_state.query != query_input:
        st.session_state.query = query_input
        perform_search()

    if st.button("Search"):
        perform_search()

elif search_type == "Image":
    uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # image = Image.open(uploaded_image)
        # transform = transforms.ToTensor()
        # tensor_image = transform(image)
        # tensor_image = tensor_image.float()  # Convert to float tensor
        # tensor_image /= 255.0 
        # st.image(tensor_image, caption="Uploaded Image", use_column_width=True)
        st.session_state.image = uploaded_image
        st.session_state.sorted_similarities = compute_similarity_image(st.session_state.image, st.session_state.embeddings)

if st.session_state.sorted_similarities:
    st.write("Top matching images:")
    
    total_images_to_show = st.slider("Total images to show", 20, 200, 20, 20)
    total_pages = (total_images_to_show - 1) // 20 + 1
    page_number = st.number_input("Page number", min_value=1, max_value=total_pages, value=1, step=1)
    
    start_index = (page_number - 1) * 20
    end_index = start_index + 20

    col_select_all, col_deselect_all = st.columns(2)

    with col_select_all:
        if st.button(f"Select All in Page"):
            select_all_images(start_index, end_index)
    with col_deselect_all:
        if st.button(f"Deselect All"):
            deselect_all_images()

    num_columns = 3
    col_width = 200
    cols = st.columns(num_columns)
    # print(st.session_state.sorted_similarities)
    for idx, (image_name, similarity) in enumerate(st.session_state.sorted_similarities[start_index:end_index]):
        col_idx = idx % num_columns
        with cols[col_idx]:
            image = Image.open(os.path.join(image_folder, image_name))
            st.image(image, caption=f"Similarity: {similarity:.3f}", use_column_width=True)
            
            checkbox_key = f"select_{image_name}"
            checked = st.checkbox("Select", key=checkbox_key, value=image_name in st.session_state.selected_images)
            if checked:
                if image_name not in st.session_state.selected_images:
                    st.session_state.selected_images.append(image_name)
            else:
                if image_name in st.session_state.selected_images:
                    st.session_state.selected_images.remove(image_name)
    
    if st.session_state.selected_images:
        if st.button("Confirm Selection"):
            for idx, (image_name, similarity) in enumerate(st.session_state.sorted_similarities[start_index:end_index]):
                checkbox_key = f"select_{image_name}"
                if st.session_state[checkbox_key] and image_name not in st.session_state.selected_images:
                    st.session_state.selected_images.append(image_name)
                elif not st.session_state[checkbox_key] and image_name in st.session_state.selected_images:
                    st.session_state.selected_images.remove(image_name)
    
    if st.session_state.selected_images:
        st.write("Selected images to export:")
        for image_name in st.session_state.selected_images:
            st.write(image_name)
        if st.button("Export to CSV"):
            export_to_csv(image_folder)