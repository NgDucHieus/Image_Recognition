import os
import shutil
import platform
import subprocess
import json
import time
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import requests
import logging
from typing import List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="image_classification.log",
)

# Determine device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load CLIP model and processor
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model.eval()

# Define construction-related labels (can be loaded from JSON)
try:
    with open("labels.json", "r") as f:
        construction_labels = json.load(f)
except FileNotFoundError:
    construction_labels = [
        "digital multimeter, black color, gray color",
        "digital multimeter, green color, dark green,hardness tester, Insize hardness gauge,",
        "tape measure, ruler, caliper measuring steel thickness",
        "Top view of corrugated metal roof, Aerial view of metal roof",
        "Structural Joints, Connection Details, Steel Connections",
        "general view of the construction site, Underside view of corrugated metal roof",
    ]
    logging.warning("labels.json not found, using default construction_labels.")

# Paths
source_folder = "Test"  # Replace with your source folder path
destination_base_folder = "Result"  # Replace with your destination folder path
confidence_threshold = 0.7  # Minimum confidence for classification

# Create destination subfolders (including Skipped)
for label in construction_labels + ["Skipped"]:
    os.makedirs(os.path.join(destination_base_folder, label), exist_ok=True)

# Generate text embeddings for labels (centroids)
text_inputs = processor(text=construction_labels, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs).cpu().numpy()

# Initialize KNN classifier with cosine similarity
knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
knn.fit(text_features, construction_labels)

# Function to extract features from a single image
def extract_clip_features(image_path: str, processor, model) -> Optional[np.ndarray]:
    """Extract CLIP features from an image (local file or URL)."""
    try:
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path, stream=True, timeout=10)
            response.raise_for_status()
            image = Image.open(response.raw)
        else:
            image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs).squeeze().cpu().numpy()
        return features
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return None

# Function to extract features from a batch of images
def extract_batch_features(image_paths: List[str], processor, model) -> np.ndarray:
    """Extract CLIP features from a batch of images."""
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            if path.startswith(("http://", "https://")):
                response = requests.get(path, stream=True, timeout=10)
                response.raise_for_status()
                images.append(Image.open(response.raw))
            else:
                images.append(Image.open(path))
            valid_paths.append(path)
        except Exception as e:
            logging.warning(f"Skipping {path}: {e}")
    
    if not images:
        return None
    
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs).cpu().numpy()
    return features if len(features) > 0 else None

# Function to open a folder based on the operating system
def open_folder(folder_path: str):
    """Open the specified folder in the default file explorer."""
    try:
        if platform.system() == "Windows":
            os.startfile(folder_path)  # Windows
        elif platform.system() == "Darwin":
            subprocess.run(["open", folder_path])  # macOS
        elif platform.system() == "Linux":
            subprocess.run(["xdg-open", folder_path])  # Linux
        else:
            logging.warning(f"Unsupported OS: {platform.system()}. Cannot open folder.")
            return
        logging.info(f"Opened folder: {folder_path}")
    except Exception as e:
        logging.error(f"Failed to open folder {folder_path}: {e}")

# Process images in batches and open folders
def classify_and_move_images(batch_size: int = 10):
    """Classify and copy images, then open the destination folder, with processing time."""
    start_time = time.time()  # Bắt đầu đo thời gian tổng cộng
    
    image_files = [
        os.path.join(source_folder, f) for f in os.listdir(source_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]
    
    if not image_files:
        logging.info("No images found in source folder.")
        return
    
    logging.info(f"Found {len(image_files)} images to process.")
    
    for i in range(0, len(image_files), batch_size):
        batch_start_time = time.time()  # Bắt đầu đo thời gian cho batch
        
        batch_paths = image_files[i:i + batch_size]
        logging.info(f"Processing batch {i // batch_size + 1} with {len(batch_paths)} images.")
        
        # Extract features for the batch
        batch_features = extract_batch_features(batch_paths, processor, model)
        if batch_features is None:
            continue
        
        # Classify each image in the batch
        predictions = knn.predict(batch_features)
        probabilities = knn.predict_proba(batch_features)
        
        for path, features, pred_label, probs in zip(batch_paths, batch_features, predictions, probabilities):
            confidence = max(probs)
            filename = os.path.basename(path)
            
            if confidence >= confidence_threshold:
                destination_path = os.path.join(destination_base_folder, pred_label, filename)
                try:
                    shutil.copy(path, destination_path)  # Sử dụng copy thay vì move
                    logging.info(
                        f"Classified {filename} as '{pred_label}' "
                        f"(confidence: {confidence:.4f}), copied to {destination_path}"
                    )
                except Exception as e:
                    logging.error(f"Error copying {filename}: {e}")
            else:
                # Copy to Skipped folder if below threshold
                destination_path = os.path.join(destination_base_folder, "Skipped", filename)
                try:
                    shutil.copy(path, destination_path)  # Sử dụng copy thay vì move
                    logging.info(
                        f"Skipped {filename}: low confidence ({confidence:.4f}) "
                        f"for '{pred_label}', copied to {destination_path}"
                    )
                except Exception as e:
                    logging.error(f"Error copying {filename} to Skipped: {e}")
        
        batch_end_time = time.time()  # Kết thúc đo thời gian batch
        batch_duration = batch_end_time - batch_start_time
        logging.info(f"Batch {i // batch_size + 1} completed in {batch_duration:.2f} seconds.")
    
    # Open the base destination folder after classification
    open_folder(destination_base_folder)
    
    end_time = time.time()  # Kết thúc đo thời gian tổng cộng
    total_duration = end_time - start_time
    logging.info(f"Total processing time: {total_duration:.2f} seconds.")

# Run the classification process
if __name__ == "__main__":
    logging.info("Starting image classification process...")
    classify_and_move_images(batch_size=10)  # Adjust batch_size as needed
    logging.info("Image classification completed!")