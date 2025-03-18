import os, shutil, platform, subprocess, json, time
from typing import List, Optional
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch, numpy as np, requests, logging
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename="image_classification.log")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); logging.info(f"Using device: {device}")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(device); model.eval()

try:
    with open("labels.json", "r") as f: construction_labels = json.load(f)
except FileNotFoundError:
    construction_labels = ["digital multimeter, black color, gray color", "digital multimeter, green color, dark green,hardness tester, Insize hardness gauge,", "tape measure, ruler, caliper measuring steel thickness", "Top view of corrugated metal roof, Aerial view of metal roof", "Structural Joints, Connection Details, Steel Connections", "general view of the construction site, Underside view of corrugated metal roof"]
    logging.warning("labels.json not found, using default labels.")

source_folder, destination_base_folder, confidence_threshold = "Test", "Result", 0.25
for label in construction_labels + ["Skipped"]: os.makedirs(os.path.join(destination_base_folder, label), exist_ok=True)

text_inputs = processor(text=construction_labels, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs); text_features = F.normalize(text_features, p=2, dim=-1)

def extract_clip_features(image_path: str, processor, model) -> Optional[torch.Tensor]:
    try:
        image = Image.open(requests.get(image_path, stream=True, timeout=10).raw) if image_path.startswith(("http://", "https://")) else Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            features = model.get_image_features(**inputs); features = F.normalize(features, p=2, dim=-1)
        return features.squeeze()
    except Exception as e: logging.error(f"Error processing {image_path}: {e}"); return None

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
    image_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]
    if not image_files: logging.info("No images found in source folder."); return
    logging.info(f"Found {len(image_files)} images to process.")

    for i in range(0, len(image_files), batch_size):
        batch_start_time = time.time()
        batch_paths = image_files[i:i + batch_size]; logging.info(f"Processing batch {i // batch_size + 1} with {len(batch_paths)} images.")
        batch_features = extract_batch_features(batch_paths, processor, model);
        if batch_features is None: continue

        for path, image_feature in zip(batch_paths, batch_features):
            similarity_scores = F.cosine_similarity(image_feature, text_features)
            confidence, predicted_label_index = torch.max(similarity_scores.cpu(), dim=0)
            predicted_label, confidence_value, filename = construction_labels[predicted_label_index], confidence.item(), os.path.basename(path)
            destination_folder = predicted_label if confidence_value >= confidence_threshold else "Skipped"
            destination_path = os.path.join(destination_base_folder, destination_folder, filename)
            try:
                shutil.copy(path, destination_path)
                log_msg = f"Classified {filename} as '{predicted_label}' (confidence: {confidence_value:.4f}), copied to {destination_path}" if destination_folder != "Skipped" else f"Skipped {filename}: low confidence ({confidence_value:.4f}) for '{predicted_label}', copied to {destination_path}"
                logging.info(log_msg)
            except Exception as e: logging.error(f"Error copying {filename}: {e}")

        batch_duration = time.time() - batch_start_time; logging.info(f"Batch {i // batch_size + 1} completed in {batch_duration:.2f} seconds.")

    open_folder(destination_base_folder)
    total_duration = time.time() - start_time; logging.info(f"Total processing time: {total_duration:.2f} seconds.")

if __name__ == "__main__":
    logging.info("Starting image classification process..."); classify_and_move_images(batch_size=10); logging.info("Image classification completed!")