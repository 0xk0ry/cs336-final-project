from inference import extract_index_features
import pickle
import torch
from models import CIRPlus
from data_utils import squarepad_transform, targetpad_transform, WikiartDataset
from config import *
import json
import os
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv

def save_image_data(clip_model, preprocess, file_path):
    # Define the validation datasets and extract the index features
    wikiart_dataset = WikiartDataset(file_path, preprocess)
    index_features, index_names = extract_index_features(wikiart_dataset, clip_model)
    json_obj = json.load(open(os.path.join(file_path, 'metadata.json'), 'r'))
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

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    list_metadata = []

    for entry in unique_metadata.values():
        for value in entry:
            words = value.replace("-", " ").replace("_", " ").split(" ")
            for word in words:
                normalized_word = word.lower()
                lemmatized_word = lemmatizer.lemmatize(normalized_word)
                if lemmatized_word.isalpha() and len(lemmatized_word) > 1 and lemmatized_word not in stop_words:
                    list_metadata.append(lemmatized_word)

    list_metadata = list(set(list_metadata))
    list_metadata_alphabetically = sorted(list_metadata)
    with open(os.path.join(file_path, 'metadata_dictionary.json'), 'w') as f:
        json.dump(list_metadata_alphabetically, f)

def create_metadata_json(images_folder):
    annotations_folder = os.path.join(images_folder, "annotations")
    output_json = os.path.join(images_folder, "metadata.json")
    metadata_list = []

    # Iterate through all CSV files in the annotations folder
    for csv_file in os.listdir(annotations_folder):
        if csv_file.endswith('.csv'):
            genre = os.path.splitext(csv_file)[0]  # Get the genre from the file name
            csv_path = os.path.join(annotations_folder, csv_file)
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_path = row['filename'] + '.jpg'
                    full_image_path = os.path.join(images_folder, "images", image_path)
                    
                    # Check if the image file exists
                    if os.path.exists(full_image_path):
                        metadata = {
                            "image_path": image_path,
                            "url": row.get('url', 'None') or 'None',
                            "title": row.get('title', 'None') or 'None',
                            "year": row.get('year', 'None') or 'None',
                            "artist": row.get('artist', 'None') or 'None',
                            "art_style": row.get('art_style', 'None') or 'None',
                            "genre": genre
                        }
                        metadata_list.append(metadata)

    # Save the metadata list as a JSON file
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=4)
        
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
  nltk.download('stopwords')
  nltk.download('wordnet')
  create_metadata_json('wikiart')
  save_image_data(clip_model, preprocess, 'wikiart')
  
  
