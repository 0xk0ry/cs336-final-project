from offline_stage import load_index_features
from config import *
from inference import predictions
from data_utils import squarepad_transform, targetpad_transform, WikiartDataset
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
from spellchecker import SpellChecker
import re
from models import CIRPlus
from typing import List, Tuple
from torch.utils.data import DataLoader
from data_utils import squarepad_transform, targetpad_transform, WikiartDataset
from combiner import Combiner
from PIL import Image
from utils import collate_fn, element_wise_sum, device
from clip.model import CLIP
import torch.nn.functional as F
import ast 
from inference import predictions
from offline_stage import load_index_features
from config import *
import json
index_features, index_names = load_index_features('wikiart-landscape/index_features.pickle')
print(len(index_features[0]), index_names[0])

from annoy import AnnoyIndex

# Step 1: Initialize the Annoy index
vector_dim = 640
index = AnnoyIndex(vector_dim, 'angular')

for i, vector in enumerate(index_features):
    index.add_item(i, vector)
# Step 3: Build the index
index.build(n_trees=10)

# # Step 4: Save the index
# index.save('test_index.ann')

# Step 5: Load and query
index = AnnoyIndex(vector_dim, 'angular')
index.load('test_index.ann')
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

clip_model, preprocess, combining_function = load_model()
text_query = 'french house'
image_query = os.path.join('wikiart-landscape', 'images', 'albrecht-durer_view-of-the-arco-valley-in-the-tyrol-1495.jpg')
query_vector = predictions(image_query, text_query, clip_model, preprocess, combining_function)
# Normalize the index features
query_vector = F.normalize(query_vector, dim=-1).float().squeeze()
print(query_vector.shape)
k = 5
nearest_neighbors = index.get_nns_by_vector(query_vector, k, include_distances=True)

print("Nearest Neighbors IDs:", nearest_neighbors[0])
print("Distances:", nearest_neighbors[1])