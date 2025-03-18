import os
import shutil
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image
import requests
import torch  # Import torch here

def classify_construction_image(image_path, candidate_labels):
    """
    Phân loại ảnh công trình xây dựng sử dụng mô hình VLM (CLIP).

    Args:
        image_path (str): Đường dẫn đến file ảnh hoặc URL ảnh.
        candidate_labels (list): Danh sách các nhãn phân loại tiềm năng (dạng text).

    Returns:
        str: Nhãn phân loại dự đoán có độ tin cậy cao nhất.
        dict: Dictionary chứa xác suất cho từng nhãn phân loại.
    """
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = AutoModelForZeroShotImageClassification.from_pretrained("openai/clip-vit-large-patch14")

    try:
        if image_path.startswith('http://') or image_path.startswith('https://'):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            image = Image.open(image_path)
    except Exception as e:
        print(f"Lỗi khi mở ảnh: {e}")
        return None, None

    inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    predicted_class_idx = probs.argmax(-1).item()
    predicted_class = candidate_labels[predicted_class_idx]

    probabilities_dict = {}
    probabilities_list = probs.tolist()[0]
    for i, label in enumerate(candidate_labels):
        probabilities_dict[label] = probabilities_list[i]

    return predicted_class, probabilities_dict

if __name__ == "__main__":

    # 1. Cấu hình thư mục và nhãn
    source_folder = "Test"  # <---- Thay đổi đường dẫn thư mục ảnh nguồn của bạn
    destination_base_folder = "ResultResult" # <---- Thay đổi đường dẫn thư mục gốc chứa các thư mục con đã phân loại
    construction_labels = [
        "digital multimeter, black color, gray colorcolor", #đo chiều dày thép
        "digital multimeter, green color, dark green", #đo độ cứng thép
        "tape measure, ruler", #ảnh đo đạc bằng thước thép #ảnh tổng thể
        "Top view of corrugated metal roof, Aerial view of metal roof, Galvanized metal roof top view, Bird’s-eye view of steel roof", #mái tôn
        "Structural Joints, Connection Details, Steel Connections", #ảnh chi tiết liên kết
        "general view of the construction site, construction site, construction project, Underside view of corrugated metal roof, Steel roof frame from below",
    ]

    # 2. Tạo thư mục đích nếu chưa tồn tại
    for label in construction_labels:
        os.makedirs(os.path.join(destination_base_folder, label), exist_ok=True)

    # 3. Duyệt qua các file ảnh trong thư mục nguồn
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')): # Lọc các file ảnh
            image_path = os.path.join(source_folder, filename)
            print(f"Đang phân loại ảnh: {filename}")

            predicted_label, probabilities = classify_construction_image(image_path, construction_labels)

            if predicted_label:
                destination_folder = os.path.join(destination_base_folder, predicted_label)
                destination_path = os.path.join(destination_folder, filename)

                try:
                    shutil.move(image_path, destination_path) # Sử dụng shutil.move để di chuyển ảnh
                    print(f"  -> Phân loại là: {predicted_label} (độ tin cậy: {probabilities[predicted_label]:.4f}), đã di chuyển đến: {destination_folder}")
                except Exception as e:
                    print(f"  -> Lỗi khi di chuyển ảnh {filename} đến thư mục {destination_folder}: {e}")
            else:
                print(f"  -> Không thể phân loại ảnh {filename}.")
        else:
            print(f"Bỏ qua file không phải ảnh: {filename}")

    print("Hoàn thành phân loại ảnh.")