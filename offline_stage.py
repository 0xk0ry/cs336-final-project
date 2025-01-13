from inference import extract_index_features
import pickle
import torch
from models import CIRPlus
from data_utils import squarepad_transform, targetpad_transform, WikiartDataset
from config import *
import json
import os


def save_image_data(clip_model, preprocess, file_path):
    # Define the validation datasets and extract the index features
    wikiart_dataset = WikiartDataset('wikiart-og', preprocess)
    index_features, index_names = extract_index_features(wikiart_dataset, clip_model)
    json_obj = json.load(open('wikiart-og/metadata.json', 'r'))
    images_metadata = []
    unique_artists = set()
    unique_titles = set()
    unique_years = set()
    unique_art_styles = set()
    unique_genres = set()

    for i, (image_feature, image_name) in enumerate(zip(index_features, index_names)):
        for item in json_obj:
            if item['image_path'] == image_name:
                image_metadata = next(item for item in json_obj if item['image_path'] == image_name)
                if image_metadata is not None:
                    images_metadata.append({
                        'index': i,
                        'image_name': image_name,
                        'title': image_metadata['title'],
                        'year': image_metadata['year'],
                        'artist': image_metadata['artist'],
                        'art_style': image_metadata['art_style'],
                        'genre': image_metadata['genre']
                    })
                    unique_artists.add(image_metadata['artist'])
                    unique_titles.add(image_metadata['title'])
                    unique_years.add(image_metadata['year'])
                    unique_art_styles.add(image_metadata['art_style'])
                    unique_genres.add(image_metadata['genre'])

    with open(os.path.join(file_path, 'image_data.pickle'), 'wb') as f:
        pickle.dump(images_metadata, f)
    with open(os.path.join(file_path, 'image_feature.pickle'), 'wb') as f:
        pickle.dump((index_features, index_names), f)

    unique_metadata = {
        'artists': sorted(list(unique_artists)),
        'titles': sorted(list(unique_titles)),
        'years': sorted(list(unique_years)),
        'art_styles': sorted(list(unique_art_styles)),
        'genres': sorted(list(unique_genres))
    }

    with open(os.path.join(file_path, 'unique_metadata.json'), 'w') as f:
        json.dump(unique_metadata, f, indent=4)

def load_index_features(metadata_path, feature_path):
  with open(metadata_path, 'rb') as f:
    image_metadata = pickle.load(f)
  with open(feature_path, 'rb') as f:
    image_feature = pickle.load(f)
    index_features, index_names = image_feature
  return image_metadata, index_features, index_names
  
if __name__ == '__main__':
  # Import CLIP
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model = CIRPlus(MODEL_NAME, transform=TRANSFORM, device=device)
  clip_model, clip_preprocess = model.clip, model.preprocess
  
  if MODEL_PATH:
    try:
      print('Trying to load the fine-tuned CLIP model')
      model.load_ckpt(MODEL_PATH, False)
      print('CLIP model loaded successfully')
      print('Trying to load the Combiner model')
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
  save_image_data(clip_model, preprocess, 'wikiart-og')
  
  
