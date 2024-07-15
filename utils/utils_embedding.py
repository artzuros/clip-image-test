import clip
import pickle, os
import streamlit as st
import torch
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm
# from transformers import CLIPModel, AutoProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# preprocess = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

def process_batch(image_paths):
    """ Processes a batch of images and returns their embeddings.
    
    Args:
        image_paths (list): A list of image paths.
        
    Returns:
        dict: A dictionary containing the embeddings of the images.
    """
    batch_embeddings = {}
    images = [preprocess(Image.open(image_path)).unsqueeze(0).to(device) for image_path in image_paths]
    images = torch.cat(images)
    with torch.no_grad():
        image_features = model.encode_image(images)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    for i, image_path in enumerate(image_paths):
        batch_embeddings[image_path] = image_features[i].cpu().numpy()
    return batch_embeddings

def load_embeddings(image_folder, embeddings_path):
    """Loads the embeddings from the given path.
    
    Args:
        image_folder (string): The folder containing the images.
        embeddings_path (string): The path to the embeddings file.
        
    Returns:
        dict: A dictionary containing the embeddings.
    """
    if os.path.isdir(image_folder):
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                st.session_state.embeddings = pickle.load(f)
            st.success("Embeddings loaded successfully.")
        else:
            st.error("Embeddings not found. Please generate them first.")
    else:
        st.error("Invalid folder path.")

def update_embeddings(image_folder, embeddings_path, batch_size=32, n_jobs=-1):
    """Updates the embeddings with new images from the given folder.

    Args:
        image_folder (string): The folder containing the images.
        embeddings_path (string): The path to the embeddings file.
        batch_size (int, optional): The size of the batch to process. Defaults to 32.
        n_jobs (int, optional): The number of jobs to run in parallel. -1 means using all processors. Defaults to -1.        
    """
    if os.path.isdir(image_folder):
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)

            existing_images = set(embeddings.keys())
            new_images = []
            for root, _, files in os.walk(image_folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        image_path = os.path.join(root, file)
                        if image_path not in existing_images:
                            new_images.append(image_path)
            st.write("Found", len(new_images), "new images.")

            batches = [new_images[i:i + batch_size] for i in range(0, len(new_images), batch_size)]
            
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_batch)(batch) for batch in tqdm(batches)
            )

            for batch_embeddings in results:
                embeddings.update(batch_embeddings)

            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
            with open(embeddings_path, 'rb') as f:
                st.session_state.embeddings = pickle.load(f)
                print("Embeddings updated and loaded successfully.")
            st.success(f"Embeddings updated and loaded successfully from {embeddings_path}")
        else:
            st.error("Embeddings not found. Please generate them first.")
    else:
        st.error("Invalid folder path.")

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

def get_image_embedding(image):
    """
    Returns the CLIP embedding for the given image.
    Args:
        image (PIL.Image): The image to get the embedding for.
        
    Returns:
        numpy.ndarray: The CLIP embedding for the given image.
    """
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()


def generate_clip_embeddings(image_folder, batch_size=32, n_jobs=-1):
    """
    Generates CLIP embeddings for the images in the given folder and its subfolders.
    Args:
        image_folder (str): The folder containing the images.
        batch_size (int): Number of images to process in a batch.
        n_jobs (int): Number of jobs to run in parallel. -1 means using all processors.
        
    Returns:
        dict: A dictionary containing the CLIP embeddings for the images.
    """
    embeddings = {}
    image_paths = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch) for batch in tqdm(batches)
    )

    for batch_embeddings in results:
        embeddings.update(batch_embeddings)
    print("Embeddings generated successfully.")
    return embeddings