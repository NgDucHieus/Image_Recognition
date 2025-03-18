import os, shutil, platform, subprocess, json, time
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch, numpy as np, requests, logging
import torch.nn.functional as F
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="image_classification.log")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); logging.info(f"Using device: {device}")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(device); model.eval()

# --- Example-Based Classification Setup ---
example_images_folder = "ExampleImages" # Folder containing subfolders for each category
destination_base_folder = "Result_ExampleBased" # New result folder for example-based classification
confidence_threshold = 0.3 # Adjust threshold as needed

# Load category folder names from labels.json (or default)
try:
    with open("labels.json", "r") as f:
        category_folders = json.load(f) # Now labels.json should contain folder names as category names
except FileNotFoundError:
    category_folders = [
        "DigitalMultimeter",
        "HardnessTester",
        "TapeMeasure",
        "CorrugatedRoof",
        "SteelJoints",
        "ConstructionSite"
    ] # Default category folder names
    logging.warning("labels.json not found, using default category folders.")

categories = category_folders # Using folder names as categories directly

# Create destination subfolders (including Skipped) in the new result folder
for category in categories + ["Skipped"]:
    os.makedirs(os.path.join(destination_base_folder, category), exist_ok=True)


# --- Define extract_clip_features function FIRST ---
def extract_clip_features(image_path: str, processor, model) -> Optional[torch.Tensor]:
    try:
        image = Image.open(requests.get(image_path, stream=True, timeout=10).raw) if image_path.startswith(("http://", "https://")) else Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs); features = F.normalize(features, p=2, dim=-1)
        return features.squeeze()
    except Exception as e: logging.error(f"Error processing {image_path}: {e}"); return None


# --- THEN define load_example_image_features function (which calls extract_clip_features) ---
def load_example_image_features(example_images_folder, categories, processor, model) -> dict:
    """Loads example images from folders and calculates average CLIP features for each category."""
    category_prototype_features = {}
    for category_folder_name in categories:
        category_path = os.path.join(example_images_folder, category_folder_name)
        if not os.path.isdir(category_path):
            logging.warning(f"Category folder '{category_path}' not found.")
            continue

        example_image_paths = [os.path.join(category_path, f) for f in os.listdir(category_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]
        if not example_image_paths:
            logging.warning(f"No example images found in '{category_path}'.")
            continue

        example_features_list = []
        for image_path in example_image_paths:
            features = extract_clip_features(image_path, processor, model) # Reuse extract_clip_features
            if features is not None:
                example_features_list.append(features)

        if example_features_list:
            # Average features to get category prototype
            category_features = torch.stack(example_features_list).mean(dim=0)
            category_prototype_features[category_folder_name] = F.normalize(category_features, p=2, dim=-1).cpu() # Normalize and move to CPU for prototype

        else:
            logging.warning(f"No features extracted for example images in '{category_path}'.")

    return category_prototype_features


def extract_batch_features(image_paths: List[str], processor, model) -> Optional[torch.Tensor]:
    images, valid_paths = [], []
    for path in image_paths:
        try:
            image = Image.open(requests.get(path, stream=True, timeout=10).raw) if path.startswith(("http://", "https://")) else Image.open(path)
            images.append(image); valid_paths.append(path)
        except Exception as e: logging.warning(f"Skipping {path}: {e}")
    if not images: return None
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad(): features = model.get_image_features(**inputs); features = F.normalize(features, p=2, dim=-1)
    return features if len(features) > 0 else None


def open_folder(folder_path: str):
    try:
        if platform.system() == "Windows": os.startfile(folder_path)
        elif platform.system() == "Darwin": subprocess.run(["open", folder_path])
        elif platform.system() == "Linux": subprocess.run(["xdg-open", folder_path])
        else: logging.warning(f"Unsupported OS: {platform.system()}. Cannot open folder."); return
        logging.info(f"Opened folder: {folder_path}")
    except Exception as e: logging.error(f"Failed to open folder {folder_path}: {e}")


def classify_and_move_images(batch_size: int = 10):
    start_time = time.time()
    source_folder = "Test" # Define source folder here, or make it a global variable if needed
    image_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]
    if not image_files: logging.info("No images found in source folder."); return
    logging.info(f"Found {len(image_files)} images to process.")

    # Load prototype features for each category (moved here as it depends on extract_clip_features)
    category_prototype_embeddings = load_example_image_features(example_images_folder, categories, processor, model)
    logging.info(f"Loaded prototype features for categories: {list(category_prototype_embeddings.keys())}")


    for i in range(0, len(image_files), batch_size):
        batch_start_time = time.time()
        batch_paths = image_files[i:i + batch_size]; logging.info(f"Processing batch {i // batch_size + 1} with {len(batch_paths)} images.")
        batch_features = extract_batch_features(batch_paths, processor, model);
        if batch_features is None: continue

        for path, image_feature in zip(batch_paths, batch_features):
            max_similarity = -1.0 # Initialize with a very low value
            predicted_category = "Skipped" # Default to Skipped
            confidence_value = 0.0

            for category_name, prototype_feature in category_prototype_embeddings.items():
                similarity_score = F.cosine_similarity(image_feature.cpu(), prototype_feature.unsqueeze(0)).item() # Compare to prototype
                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    predicted_category = category_name
                    confidence_value = similarity_score

            filename = os.path.basename(path)
            destination_folder = predicted_category if confidence_value >= confidence_threshold else "Skipped"
            destination_path = os.path.join(destination_base_folder, destination_folder, filename)

            try:
                shutil.copy(path, destination_path)
                log_msg = f"Classified {filename} as '{predicted_category}' (confidence: {confidence_value:.4f}), copied to {destination_path}" if destination_folder != "Skipped" else f"Skipped {filename}: low confidence ({confidence_value:.4f}) for '{predicted_category}', copied to {destination_path}"
                logging.info(log_msg)
            except Exception as e: logging.error(f"Error copying {filename}: {e}")

        batch_duration = time.time() - batch_start_time; logging.info(f"Batch {i // batch_size + 1} completed in {batch_duration:.2f} seconds.")

    open_folder(destination_base_folder)
    total_duration = time.time() - start_time; logging.info(f"Total processing time: {total_duration:.2f} seconds.")


if __name__ == "__main__":
    logging.info("Starting example-based image classification process...")
    classify_and_move_images(batch_size=10)
    logging.info("Image classification completed!")