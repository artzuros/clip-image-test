import streamlit as st
import torch, pickle, os
import clip

from utils.utils_embedding import generate_clip_embeddings

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

st.title("CLIP Embedding Generation")

image_folder = st.text_input("Enter the FULL path of the images folder:")

if st.button("Generate and Save Embeddings"):
    if os.path.isdir(image_folder):
        embeddings = generate_clip_embeddings(image_folder)
        embeddings_path = os.path.join(image_folder, 'image_embeddings.pkl')
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        st.success(f"Embeddings generated and saved successfully in {embeddings_path}")
    else:
        st.error("Invalid folder path")