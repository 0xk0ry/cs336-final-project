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
from models import CIRPlus
from typing import List, Tuple
from torch.utils.data import DataLoader
from data_utils import squarepad_transform, targetpad_transform, WikiartDataset
from combiner import Combiner
from PIL import Image
from utils import collate_fn, element_wise_sum, device
from clip.model import CLIP
import pickle
import torch.nn.functional as F

def predictions(image_query, text_query, clip_model, preprocess, combining_function):
    # print("Compute CIRR validation predictions for a single query")

    # Load and preprocess the image
    if image_query != '':
        input_img = Image.open(image_query)
        input_img = preprocess(input_img).unsqueeze(0).to(device)
        print('loaded image')
    else:
        input_img = torch.zeros((1, 3, clip_model.visual.input_resolution, clip_model.visual.input_resolution)).to(device)
    # Tokenize the text query
    text_inputs = clip.tokenize([text_query]).to(device)

    # Compute the predicted features
    with torch.no_grad():
        image_features = clip_model.encode_image(input_img)
        text_features = clip_model.encode_text(text_inputs)
        predicted_features = combining_function(image_features, text_features)

    return predicted_features


def get_predictions(image_query: torch.Tensor, text_query: str, clip_model: CLIP, preprocess, index_features: torch.tensor,
                             index_names: List[str], combining_function: callable) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param clip_model: CLIP model
    :param index_features: validation index features
    :param index_names: validation index names
    :param combining_function: function which takes as input (image_features, text_features) and outputs the combined
                            features
    :return: the computed validation metrics
    """
    # Generate predictions
    predicted_features = \
        predictions(image_query, text_query, clip_model, preprocess, combining_function)

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # # Delete the reference image from the results
    # reference_mask = torch.tensor(
    #     sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    # sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
    #                                                                 sorted_index_names.shape[1] - 1)
    
    return sorted_index_names

def extract_index_features(dataset, clip_model):
    feature_dim = clip_model.visual.output_dim
    # Use drop_last=True to avoid incomplete or empty batches
    classic_val_loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    index_names = []

    # Check if the dataset is empty
    if len(dataset) == 0:
        print("Dataset is empty.")
        return index_features, index_names

    for names, images in tqdm(classic_val_loader):
        if len(names) == 0 or len(images) == 0:
            print("Skipping empty batch.")
            continue

        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)
    
    return index_features, index_names

def inference(combining_function: callable, clip_model: CLIP, preprocess: callable, image_query: str, text_query: str):

    # Define the validation datasets and extract the index features
    wikiart_dataset = WikiartDataset('wikiart-landscape', preprocess)
    index_features, index_names = extract_index_features(wikiart_dataset, clip_model)
    return get_predictions(image_query, text_query, clip_model, preprocess, index_features, index_names,
                                    combining_function)
    return 0
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image and text inference using CLIP model")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--text_query", type=str, help="Text query for inference")
    parser.add_argument("--db_path", type=str, default="database", help="Path to the image database")
    parser.add_argument("--clip_model_name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--model_path", type=Path, default=None, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--combining_function", type=str, required=True,
                        help="Which combining function use, should be in ['combiner', 'sum']")
    parser.add_argument("--combiner_path", type=Path, default=None, help="path to trained Combiner")
    parser.add_argument("--projection_dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden_dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--target_ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")

    args = parser.parse_args()

    # Import CLIP
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
    model = CIRPlus(args.clip_model_name, transform=args.transform, device=device)
    
    #! add this
    if args.model_path:
        try:
            print('Trying to load the fine-tuned CLIP model')
            model.load_ckpt(args.model_path, False)
            print('CLIP model loaded successfully')
            print('Trying to load the Combiner model')
        except:
            raise Exception('Fine-tuned CLIP model load failed')
        
    clip_model, clip_preprocess = model.clip, model.preprocess

    clip_model.eval()
    input_dim = clip_model.visual.input_resolution
    feature_dim = clip_model.visual.output_dim
    
    # Import combiner
    if args.combining_function == 'sum':
        model.load_combiner(args.combining_function)
    else:
        model.load_combiner(args.combining_function, args.combiner_path, args.projection_dim, args.hidden_dim)

    # Import preprocess
    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.transform == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        print('CLIP default preprocess pipeline is used')
        preprocess = clip_preprocess
        
    image_query = args.image_path
    text_query = args.text_query
    sorted_index_names = inference(model.combining_function, clip_model, preprocess, image_query, text_query)
    print(sorted_index_names[:5])
    # print(sorted_index_names)
    
    # results = infer(args.image_path, args.text_query, args.db_path)
    
    # for img_file, sim in results:
    #     print(f"Image: {img_file}, Similarity: {sim:.4f}")