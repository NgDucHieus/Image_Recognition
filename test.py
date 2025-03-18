import cv2
import numpy as np
import os
import uuid
from PIL import Image
import sys

# Configurable thresholds for each category
THRESHOLDS = {
    "roof_lines": 70,
    "ruler_long_lines": 5,
    "ruler_pixels": 2000,
    "green_pixels": 5000,
    "connection_lines": 10,
    "thickness_area": 5000,  # Area for thickness measurement device detection
    "thickness_aspect_ratio_min": 0.5, # Aspect ratio min for thickness device
    "thickness_aspect_ratio_max": 2.0 # Aspect ratio max for thickness device
}

# Classification functions for each of the 6 categories
def is_roof(img):
    """Classifies 'Mái tôn' (Roof) images."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # ...existing code...
# Corrected line using keyword arguments for HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
# ...existing code...
    num_lines = len(lines) if lines is not None else 0
    return num_lines > THRESHOLDS["roof_lines"]

def is_overall_image(img):
    """Classifies 'Ảnh tổng thể' (Overall Image). This is a fallback, so logic can be adjusted if needed."""
    # For now, classify as overall if it doesn't fit other categories.
    # More specific features for overall images could be added if needed (e.g., image size, complexity).
    return True # Default to overall if no other category is matched

def is_connection_detail(img):
    """Classifies 'Ảnh chi tiết liên kết' (Connection Detail) images."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=5)
    num_short_lines = len(lines) if lines is not None else 0
    return num_short_lines > THRESHOLDS["connection_lines"]

def is_ruler_measurement(img):
    """Classifies 'Ảnh đo đạc bằng thước thép' (Ruler Measurement) images."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    num_long_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if np.sqrt((x2 - x1)**2 + (y2 - y1)**2) > 100:
                num_long_lines += 1
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    red_pixels = cv2.countNonZero(mask_red)
    return num_long_lines > THRESHOLDS["ruler_long_lines"] and red_pixels > THRESHOLDS["ruler_pixels"]

def is_thickness_measurement(img):
    """Classifies 'Đo chiều dày thép' (Thickness Measurement) images."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > THRESHOLDS["thickness_area"]:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if THRESHOLDS["thickness_aspect_ratio_min"] <= aspect_ratio <= THRESHOLDS["thickness_aspect_ratio_max"]:
                return True

    return False

def is_hardness_test(img):
    """Classifies 'Đo độ cứng thép' (Hardness Test) images."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask_green)
    return green_pixels > THRESHOLDS["green_pixels"]


def classify_image(image_path):
    """Classifies an image into one of 6 categories and returns category and image."""
    img = cv2.imread(image_path)
    if img is None:
        return "Không xác định (Lỗi đọc ảnh)", None

    if is_roof(img):
        return "Mái tôn", img
    if is_ruler_measurement(img):
        return "Ảnh đo đạc bằng thước thép", img
    if is_hardness_test(img):
        return "Đo độ cứng thép", img
    if is_connection_detail(img):
        return "Ảnh chi tiết liên kết", img
    if is_thickness_measurement(img):
        return "Đo chiều dày thép", img

    return "Ảnh tổng thể", img  # Fallback to "Ảnh tổng thể" if no other category matches

# Main execution
if __name__ == "__main__":
    image_folder = os.path.abspath("Test")  # Folder "Test" in script directory
    if not os.path.exists(image_folder):
        print(f"Error: Image folder '{image_folder}' does not exist.")
        exit(1)

    image_files = os.listdir(image_folder)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Working directory: {os.getcwd()}")

    for image_file in image_files:
        if not image_file.lower().endswith(valid_extensions):
            print(f"Skipping non-image file: {image_file}")
            continue

        image_path = os.path.join(image_folder, image_file)
        category, img = classify_image(image_path)
        print(f"Ảnh: {image_file}, Phân loại: {category}")

        if img is not None:
            category_folder = os.path.join(image_folder, category)
            os.makedirs(category_folder, exist_ok=True)

            save_filename = f"{uuid.uuid4().hex}.jpg"
            save_path = os.path.join(category_folder, save_filename)

            try:
                success = cv2.imwrite(save_path, img)
                if success:
                    print(f"Image saved successfully to: {save_path}")
                else:
                    print(f"Failed to save image to: {save_path}")
                    print(f" - Directory writable? {os.access(category_folder, os.W_OK)}")
                    print(f" - File already exists? {os.path.exists(save_path)}")
                    print(f" - Image shape: {img.shape}")
                    print(f" - Disk free space (bytes): {os.statvfs(category_folder).f_bavail * os.statvfs(category_folder).f_frsize if hasattr(os, 'statvfs') else 'Unknown (Windows)'}")

                    # Fallback with PIL
                    print("Attempting fallback save with PIL...")
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    pil_img.save(save_path, "JPEG")
                    if os.path.exists(save_path):
                        print(f"Fallback save successful with PIL to: {save_path}")
                    else:
                        print(f"Fallback save with PIL also failed.")

            except Exception as e:
                print(f"Error saving {save_path}: {str(e)}")

    print("\n--- Classification Complete ---")
    print("Notes:")
    print("- Adjust thresholds in THRESHOLDS for better accuracy.")
    print("- Ensure 'Test' folder exists in the same directory as the script and contains images.")
    print("- Categories: Mái tôn, Ảnh tổng thể, Ảnh chi tiết liên kết, Ảnh đo đạc bằng thước thép, Đo chiều dày thép, Đo độ cứng thép")