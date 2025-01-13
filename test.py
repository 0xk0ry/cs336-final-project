import os
import csv
import json

import os
import csv
import json

def create_metadata_json(annotations_folder, images_folder, output_json):
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
                    full_image_path = os.path.join(images_folder, image_path)
                    
                    # Check if the image file exists
                    if os.path.exists(full_image_path):
                        metadata = {
                            "image_path": image_path,
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

# Define the annotations folder, images folder, and output JSON file path
annotations_folder = 'wikiart-og/annotations'
images_folder = 'wikiart-og/images'
output_json = 'wikiart-og/metadata.json'

# Create the metadata JSON file
# create_metadata_json(annotations_folder, images_folder, output_json)
# import os
# import shutil

# def move_images_to_root(images_folder):
#     # Iterate through all genre folders in the images folder
#     for genre_folder in os.listdir(images_folder):
#         genre_path = os.path.join(images_folder, genre_folder)
#         if os.path.isdir(genre_path):
#             # Iterate through all images in the genre folder
#             for image_file in os.listdir(genre_path):
#                 old_image_path = os.path.join(genre_path, image_file)
#                 new_image_path = os.path.join(images_folder, image_file)
                
#                 # Move the image to the new location
#                 if os.path.exists(old_image_path):
#                     shutil.move(old_image_path, new_image_path)

# # Define the images folder path
# images_folder = 'wikiart-og/images'

# # Move the images to the root of the images folder
# move_images_to_root(images_folder)


with open("./validation.txt", "r") as f:
  lines = [line.strip() for line in f if line.strip()]

headers = lines[0].split(",")
values = lines[1].split()

for h, v in zip(headers, values):
  print(f"{h}: {v}")