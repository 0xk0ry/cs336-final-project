import streamlit as st
import os
import torch
import clip
import torchvision.transforms as T
import argparse
import numpy as np
from operator import itemgetter
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import re
from models import CIRPlus, CLIPGradCAM
from typing import List, Tuple
from torch.utils.data import DataLoader
from data_utils import squarepad_transform, targetpad_transform, WikiartDataset
from combiner import Combiner
from PIL import Image
from utils import collate_fn, element_wise_sum, device
from clip.model import CLIP
import torch.nn.functional as F
from offline_stage import load_index_features
from config import *
import json
import time
import cv2
import faiss
import annoy

st.set_page_config(
    layout="wide",
)

class UI:
    def __init__(self):
        st.sidebar.title("Wikiart Image Retrieval")
        st.sidebar.write(f"Total images: {len(unique_metadata['titles'])}")
        st.sidebar.write(f"Total artist: {len(unique_metadata['artists'])}")
        st.sidebar.write(f"Total art styles: {len(unique_metadata['art_styles'])}")
        st.sidebar.write(f"Total genres: {len(unique_metadata['genres'])}")
        st.sidebar.markdown('---')
        # Initialize session states
        if 'image_query' not in st.session_state:
            st.session_state.image_query = ''
        if 'ui_key' not in st.session_state:
            st.session_state.ui_key = 0
            
        self.text_query = st.sidebar.text_input("Enter your search query:", on_change=self.handle_query_change)
        self.indexing_type = st.sidebar.selectbox("Choose Indexing", ["None", "FAISS", "ANNOY"], index=0)
        self.top_k = st.sidebar.number_input("Choose top-k for re-ranking", value=3)
        if st.session_state.image_query != '':
            st.sidebar.button("Clear image Selection", on_click=self.handle_clear_image)
        # File uploader
        uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state.image_query = uploaded_file.name
            image.save(os.path.join('wikiart/images', uploaded_file.name))
                        
        if st.session_state.image_query != '':
            st.sidebar.markdown("**Selected Image:**")
            st.sidebar.image(os.path.join('wikiart/images', st.session_state.image_query), caption=st.session_state.image_query)
        
        # Metadata filtering
        self.include_artists = st.sidebar.multiselect("Include Artists", unique_metadata['artists'])
        self.exclude_artists = st.sidebar.multiselect("Exclude Artists", unique_metadata['artists'])
        self.include_titles = st.sidebar.text_input("Title Include (Separate using ,)")
        self.exclude_titles = st.sidebar.text_input("Title Exclude (Separate using ,)")
        self.include_years = st.sidebar.multiselect("Include Years", unique_metadata['years'])
        self.exclude_years = st.sidebar.multiselect("Exclude Years", unique_metadata['years'])
        self.include_art_styles = st.sidebar.multiselect("Include Art Styles", unique_metadata['art_styles'])
        self.exclude_art_styles = st.sidebar.multiselect("Exclude Art Styles", unique_metadata['art_styles'])
        self.include_genres = st.sidebar.multiselect("Include Genres", unique_metadata['genres'])
        self.exclude_genres = st.sidebar.multiselect("Exclude Genres", unique_metadata['genres'])
    
    
    def handle_clear_image(self):
        st.session_state.image_query = ''
        
    def handle_query_change(self):
        reset_show_count()
        
def handle_image_query(image_name):
    st.session_state.image_query = image_name
    reset_show_count()
     
def reset_show_count():
    st.session_state.show_count = 12
    st.session_state.show_more_clicked = False
    st.session_state.selected_image_metadata = None
    
def handle_show_more(show_count):
    st.session_state.show_count += show_count
    st.session_state.show_more_clicked = True
   
def handle_image_selection(text_query, image_metadata, col):
    st.session_state.selected_image_metadata = image_metadata
    st.session_state.show_count_metadata = 12
    st.session_state.show_more_clicked_metadata = False
    
def handle_show_more_metadata(show_count):
    st.session_state.show_count_metadata += show_count
    st.session_state.show_more_clicked_metadata = True
     
def display_metadata(text_query, image_metadata, col):
    with col:
        image_name = image_metadata['image_name']
        
        col1, col2 = st.columns([0.65,0.35])
        with col1:
            target_image_path = os.path.join('wikiart/images', image_name)
            st.image(target_image_path, caption=image_name)
        with col2:
            st.write(f"**Title:** {image_metadata['title']}")
            st.write(f"**Year:** {image_metadata['year']}")
            st.write(f"**Artist:** {image_metadata['artist']}")
            st.write(f"**Art Style:** {image_metadata['art_style']}")
            st.write(f"**Genre:** {image_metadata['genre']}")
            st.button("Use as input", on_click=handle_image_query, args=(image_name,))
        
        if st.session_state.show_more_clicked_metadata == False:
            image_path = os.path.join('wikiart/images', image_name)
            text_features, image_features = get_features(text_query, image_path)
            
            sorted_indexes = get_predictions(
                image_features,
                text_features,
                clip_model,
                preprocess,
                index_features,
                index_names,
                combining_function
            )
            
            # st.session_state.sorted_index_metadata = filter_index(sorted_indexes[0].tolist())
            st.session_state.sorted_index_metadata = filter_index(sorted_indexes)
        col1, col2, col3 = st.columns(3)
        columns = [col1, col2, col3]

        count = 0
        i = 0
        while count < st.session_state.show_count_metadata and i < len(st.session_state.sorted_index_metadata):
            image = images_metadata[st.session_state.sorted_index_metadata[i]]
            image_path = os.path.join('wikiart/images', image['image_name'])
            columns[count % 3].image(image_path)
            columns[count % 3].button(f"{image['title']}", on_click=handle_image_selection, args=(text_query, image, col),key={image['image_name']})
            i+=1
            count+=1
            # Show more button
        st.markdown('---')
        if st.session_state.show_count_metadata + 6 < len(st.session_state.sorted_index_metadata):
            st.button('Show more', key="show_more_metadata", on_click=handle_show_more_metadata, args=(6,)) 

def display_image_gallery(text_query, sorted_indexes, col1, col2, show_count):
    if len(sorted_indexes) <= 0:
        return None
    
    # Initialize show count state
    if 'show_count' not in st.session_state:
        st.session_state.show_count = show_count

    # Show only the show_count amount of images
    with col1:
        col11, col12, col13 = st.columns(3)
        columns = [col11, col12, col13]
        count = 0
        with col1:
            for i, index in enumerate(sorted_indexes[:st.session_state.show_count]):
                image_path = os.path.join('wikiart/images', images_metadata[index]['image_name'])
                columns[count % 3].image(image_path)
                if columns[count % 3].button(f"{images_metadata[index]['title']}", key=f"{image_path}_{i}"):
                    st.session_state.selected_image_metadata = images_metadata[index]
                count += 1
            col1.markdown('---')

        # Show more button
        if st.session_state.show_count + show_count < len(sorted_indexes):
            st.button('Show more', key="show_more", on_click=handle_show_more, args=(show_count//2,)) 

def get_average_features(sorted_indexes):
    features_tensor = torch.stack([index_features[index] for index in sorted_indexes])
    return features_tensor.mean(dim=0)

def get_predictions(image_features, text_features, clip_model: CLIP, preprocess, 
                   index_features: torch.tensor, index_names: List[str], 
                   combining_function: callable, k: int = 0, index_type: str = None):
    if text_features != None and image_features != None:
        final_features = combining_function(text_features, image_features)
    elif text_features != None:
        final_features = text_features
    else:
        final_features = image_features
    
    if k <= 0 or index_type == "None":
        # Normalize index features
        index_features = F.normalize(index_features, dim=-1).float()
        # Calculate similarities
        distances = 1 - final_features @ index_features.T
        
        indices = torch.argsort(distances, dim=-1).cpu().tolist()[0]
        return indices
    else:
        if index_type == 'FAISS':
            distances, indices = faiss_index.search(final_features.cpu().detach().numpy(), k)
            indices = indices[0]
        else:
            indices = annoy_index.get_nns_by_vector(final_features.squeeze(0), k)
        return indices
    
def get_features(text_query, image_path = None):
    if text_query == None:
        text_query = ''
    text_tokens = clip.tokenize([text_query]).to(device)
        
    with torch.no_grad():
        if text_query != None:
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if image_path != None:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            image_features = clip_model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        else:
            image_features = torch.zeros_like(index_features[0]).unsqueeze(0).to(device)
    return text_features, image_features
    
def filter_index(sorted_indexes):
    include_artists = set(ui.include_artists) if ui.include_artists else None
    exclude_artists = set(ui.exclude_artists) if ui.exclude_artists else None
    include_titles = [word.strip().lower() for word in ui.include_titles.split(',')] if ui.include_titles else []
    exclude_titles = [word.strip().lower() for word in ui.exclude_titles.split(',')] if ui.exclude_titles else []
    include_years = set(ui.include_years) if ui.include_years else None
    exclude_years = set(ui.exclude_years) if ui.exclude_years else None
    include_art_styles = set(ui.include_art_styles) if ui.include_art_styles else None
    exclude_art_styles = set(ui.exclude_art_styles) if ui.exclude_art_styles else None
    include_genres = set(ui.include_genres) if ui.include_genres else None
    exclude_genres = set(ui.exclude_genres) if ui.exclude_genres else None

    def passes_filters(idx):
        meta = images_metadata[idx]
        title_words = [w.strip().lower() for w in meta['title'].split(',')]
        
        if include_artists and meta['artist'] not in include_artists:
            return False
        if exclude_artists and meta['artist'] in exclude_artists:
            return False
        if include_titles and not any(t in w for t in include_titles for w in title_words):
            return False
        if exclude_titles and any(t in w for t in exclude_titles for w in title_words):
            return False
        if include_years and meta['year'] not in include_years:
            return False
        if exclude_years and meta['year'] in exclude_years:
            return False
        if include_art_styles and meta['art_style'] not in include_art_styles:
            return False
        if exclude_art_styles and meta['art_style'] in exclude_art_styles:
            return False
        if include_genres and meta['genre'] not in include_genres:
            return False
        if exclude_genres and meta['genre'] in exclude_genres:
            return False
        return True

    return [i for i in sorted_indexes if passes_filters(i)]

def main():
    col1, gap, col2 = st.columns([1, 0.05, 1])
    if (ui.text_query or st.session_state.image_query) and not st.session_state.show_more_clicked:
        
        st.session_state.use_image_as_input_clicked = False
        if st.session_state.image_query != '':
            image_path = os.path.join('wikiart\images', st.session_state.image_query)
        else:
            image_path = None
        text_features, image_features = get_features(ui.text_query, image_path)
        st.session_state.text_features = text_features
        st.session_state.image_features = image_features
        start_time = time.time()

        # First query
        sorted_indexes = get_predictions(
            image_features,
            text_features,
            clip_model,
            preprocess,
            index_features,
            index_names,
            combining_function,
            ui.top_k,
            ui.indexing_type
        )
        # Get average features
        avg_features =  get_average_features(sorted_indexes[:ui.top_k if ui.top_k > 0 else min(3, len(sorted_indexes))])
        # Second query with averaged features
        sorted_indexes = get_predictions(
            avg_features.unsqueeze(0),
            text_features,
            clip_model,
            preprocess,
            index_features,
            index_names,
            combining_function
        )
        avg_features =  get_average_features(sorted_indexes)
        st.session_state.sorted_index = filter_index(sorted_indexes)
        st.session_state.search_time = time.time() - start_time
        col1.write(f"Time taken for retrieval: {st.session_state.search_time:.5f} seconds")
        display_image_gallery(ui.text_query, st.session_state.sorted_index, col1, col2, 12)
    elif (ui.text_query or st.session_state.image_query):
        col1.write(f"Time taken for predictions: {st.session_state.search_time:.5f} seconds")
        display_image_gallery(ui.text_query, st.session_state.sorted_index, col1, col2, 12)

    if 'selected_image_metadata' in st.session_state and st.session_state.selected_image_metadata != None:
        display_metadata(ui.text_query, st.session_state.selected_image_metadata, col2)

@st.cache_resource
def load_model():
    # Import CLIP
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = CIRPlus(MODEL_NAME, transform=TRANSFORM, device=device)
    clip_model, clip_preprocess = model.clip, model.preprocess
    if MODEL_PATH:
        try:
            print('Streamlit trying to load the fine-tuned CLIP model')
            model.load_ckpt(MODEL_PATH, False)
            print('CLIP model loaded successfully')
            print('Streamlit trying to load the Combiner model')
        except:
            raise Exception('Fine-tuned CLIP model load failed')
        
    clip_model, clip_preprocess = model.clip, model.preprocess

    clip_model.eval()
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim

    # Import preprocess
    if TRANSFORM == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(TARGET_RATIO, input_dim)
    elif TRANSFORM == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess
        
    # Import combiner
    if COMBINING_FUNCTION == 'sum':
        model.load_combiner(COMBINING_FUNCTION)
    else:
        model.load_combiner(COMBINING_FUNCTION, COMBINER_PATH, PROJECTION_DIM, HIDDEN_DIM)
    combining_function = model.combining_function
    return clip_model, preprocess, combining_function

# if __name__ == "__main__":
if 'sorted_index' not in st.session_state:
    st.session_state.sorted_index = None
if 'sorted_index_metadata' not in st.session_state:
    st.session_state.sorted_index_metadata = None
if 'search_time' not in st.session_state:
    st.session_state.search_time = None
if 'show_more_clicked' not in st.session_state:
    st.session_state.show_more_clicked = False
if 'show_more_clicked_metadata' not in st.session_state:
    st.session_state.show_more_clicked_metadata = False
if 'show_count_metadata' not in st.session_state:
    st.session_state.show_count_metadata = 12
clip_model, preprocess, combining_function = load_model()
images_metadata, index_features, index_names = load_index_features('wikiart/image_data.pickle', 'wikiart/image_feature.pickle')
faiss_index = faiss.read_index("wikiart/faiss_index.idx")
annoy_index = annoy.AnnoyIndex(640, 'angular')
annoy_index.load("wikiart/annoy_index.ann")
unique_metadata = json.load(open('wikiart/unique_metadata.json', 'r'))
metadata_list = json.load(open('wikiart/metadata_dictionary.json', 'r'))
ui = UI()
main()