# Class Project

## Team Members
| Name                  | Email                  | GitHub                                  |
|-----------------------|------------------------|-----------------------------------------|
| Lê Bình Nguyên         | 22520969@gm.uit.edu.vn  | [lbngyn](https://github.com/lbngyn)     |
| Phan Huỳnh Ngọc Trâm   | 22521500@gm.uit.edu.vn  | [Aph3li0s](https://github.com/Aph3li0s) |
| Phạm Thạch Thanh Trúc  | 22521551@gm.uit.edu.vn  | [0xk0ry](https://github.com/0xk0ry)     |

## Overview
This project is a retrieval system that combines image and text features for advanced search capabilities. It includes offline processing to extract features and metadata, as well as an interactive web-based demo.

## Features
- **Zero-shot Image Retrieval:** Search for images using text queries without prior fine-tuning.
- **Composed Image Retrieval:** Combine text and image inputs for refined search results.
- **Metadata Filtering:** Apply various filters to narrow down search results.

## Getting Started
### Prerequisites
- Python 3.9
- CUDA 12.4

### Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/0xk0ry/cs336-final-project
   cd cs336-final-project
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Checkpoints**
   - Download the pre-trained model checkpoints from the [release section](https://github.com/0xk0ry/cs336-final-project/releases/tag/release).

4. **Download Images**
   - Download the images from the [Google Drive](https://drive.google.com/drive/folders/18OAlaadpgcgoof7QuWCpDd3bB1uKPo-S?usp=sharing) and place them in the designated directory.

5. **Run the Application**
   - Launch the Streamlit demo:
     ```bash
     streamlit run demo.py
     ```

## Project Components
### Scripts
#### `offline_stage.py`
- **Purpose:** Extract image features and metadata for the retrieval system.
- **Output:**
  - `metadata.json`: Contains extracted metadata.
  - `image_data.pickle`: Serialized image data.
  - `image_feature.pickle`: Pre-computed image features.
- Note: The provided `unique_metadata.json` is derived from `metadata.json` to ensure unique values.

#### `demo.py`
- **Purpose:** Web-based deployment for the retrieval system.
- **Capabilities:**
  - Zero-shot image retrieval with text queries.
  - Composed image retrieval using text and image inputs.
  - Filtering options to refine search results.

## Usage Example
1. Launch the demo using the command:
   ```bash
   streamlit run demo.py
   ```

2. Use the interface to upload an image, input a text query, or both.

3. Apply filters as needed to customize search results.