import sys
import os
import shutil
import platform
import subprocess
import json
import logging
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLineEdit, QFileDialog, QProgressBar, QLabel,
    QFrame, QHBoxLayout, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath
from PySide6.QtGui import QIcon
from PySide6.QtCore import QSize
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="image_classification.log",
)


class ImageClassificationWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, source_folder, destination_folder):
        super().__init__()
        self.source_folder = source_folder
        self.destination_folder = destination_folder
        self.stopped = False

    def run(self):
        try:
            self.status.emit("Initializing...")
            self.progress.emit(5)

            # Import heavy modules inside the thread to keep UI responsive
            import torch
            import numpy as np
            from PIL import Image

            try:
                from transformers import AutoProcessor, AutoModel
                from sklearn.neighbors import KNeighborsClassifier
            except ImportError:
                self.error.emit(
                    "Required packages not installed. Please install transformers and scikit-learn.")
                return

            self.status.emit("Loading AI model...")
            self.progress.emit(10)

            # Determine device (CPU or GPU)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {device}")

            # Load CLIP model and processor
            try:
                processor = AutoProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32")
                model = AutoModel.from_pretrained(
                    "openai/clip-vit-base-patch32").to(device)
                model.eval()
                self.progress.emit(30)
            except Exception as e:
                self.error.emit(f"Failed to load the AI model: {str(e)}")
                return

            # Define construction-related labels
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
                logging.warning(
                    "labels.json not found, using default construction_labels.")

            confidence_threshold = 0.7  # Minimum confidence for classification

            # Create destination subfolders (including Skipped)
            self.status.emit("Creating output folders...")
            self.progress.emit(35)

            for label in construction_labels + ["Skipped"]:
                label_folder = os.path.join(self.destination_folder, label)
                os.makedirs(label_folder, exist_ok=True)

            # Generate text embeddings for labels (centroids)
            self.status.emit("Processing label embeddings...")
            self.progress.emit(40)

            text_inputs = processor(
                text=construction_labels, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                text_features = model.get_text_features(
                    **text_inputs).cpu().numpy()

            # Initialize KNN classifier with cosine similarity
            knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
            knn.fit(text_features, construction_labels)

            # Get image files
            self.status.emit("Finding images...")
            image_files = [
                os.path.join(self.source_folder, f) for f in os.listdir(self.source_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
            ]

            if not image_files:
                self.status.emit("No images found!")
                self.error.emit("No images found in the source folder.")
                return

            total_images = len(image_files)
            self.status.emit(f"Processing {total_images} images...")

            # Process images one by one to update progress accurately
            for i, image_path in enumerate(image_files):
                if self.stopped:
                    return

                filename = os.path.basename(image_path)
                self.status.emit(
                    f"Processing {filename} ({i+1}/{total_images})")

                try:
                    # Process image
                    image = Image.open(image_path)
                    inputs = processor(
                        images=image, return_tensors="pt", padding=True).to(device)

                    with torch.no_grad():
                        features = model.get_image_features(
                            **inputs).cpu().numpy()

                    # Classify image
                    pred_label = knn.predict(features)[0]
                    probabilities = knn.predict_proba(features)[0]
                    confidence = max(probabilities)

                    # Copy to appropriate folder based on confidence
                    if confidence >= confidence_threshold:
                        destination_path = os.path.join(
                            self.destination_folder, pred_label, filename)
                        logging.info(
                            f"Classified {filename} as '{pred_label}' (confidence: {confidence:.4f})")
                    else:
                        destination_path = os.path.join(
                            self.destination_folder, "Skipped", filename)
                        logging.info(
                            f"Skipped {filename}: low confidence ({confidence:.4f})")

                    shutil.copy(image_path, destination_path)

                except Exception as e:
                    logging.error(f"Error processing {filename}: {str(e)}")

                # Update progress
                # 40%-95% range for processing
                progress_value = 40 + int((i + 1) / total_images * 55)
                self.progress.emit(progress_value)

            self.status.emit("Classification complete!")
            self.progress.emit(100)
            self.finished.emit()

        except Exception as e:
            self.error.emit(f"An unexpected error occurred: {str(e)}")
            logging.exception("Unexpected error in worker thread")

    def stop(self):
        self.stopped = True


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
            logging.warning(
                f"Unsupported OS: {platform.system()}. Cannot open folder.")
            return
        logging.info(f"Opened folder: {folder_path}")
    except Exception as e:
        logging.error(f"Failed to open folder {folder_path}: {e}")


class RoundedFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("roundedFrame")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 12, 12)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#ffffff"))
        painter.drawPath(path)


class ImageClassificationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.worker = None

    def initUI(self):
        self.setWindowTitle("Image Classification Tool")
        self.setFixedSize(600, 450)

        # Gradient background
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                           stop:0 #f5f7fa, stop:1 #e9eef6);
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)

        # Create a rounded main container
        container = RoundedFrame()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(25, 25, 25, 25)
        container_layout.setSpacing(20)

        # Title
        title_label = QLabel("Image Classification Tool")
        title_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #333333;")
        container_layout.addWidget(title_label)

        # Source folder selection
        source_label = QLabel("Source Folder:")
        source_label.setStyleSheet("font-weight: bold; color: #555555;")
        container_layout.addWidget(source_label)

        source_frame = QFrame()
        source_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #dcdcdc;
                border-radius: 8px;
                background-color: #ffffff;
            }
        """)

        source_layout = QHBoxLayout(source_frame)
        source_layout.setContentsMargins(12, 12, 12, 12)

        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("Select source folder...")
        self.source_edit.setReadOnly(True)
        self.source_edit.setMinimumHeight(14)
        self.source_edit.setStyleSheet(
            "border: none; font-size: 14px; background: transparent; color: black;")

        source_browse = QPushButton()
        source_browse.setIcon(QIcon.fromTheme("folder"))
        source_browse.setIconSize(QSize(18, 18))  # Kích thước icon phù hợp
        source_browse.setFixedSize(30, 30)  # Giữ nút nhỏ gọn, chỉ vừa với icon
        source_browse.setStyleSheet(
            "background: #4a86e8; border: none; border-radius:5px;margin-bottom: 10px;margin-right: 10px;")
        source_browse.clicked.connect(self.select_source_folder)

        source_layout.addWidget(self.source_edit)
        source_layout.addWidget(source_browse)
        container_layout.addWidget(source_frame)

        # Destination folder selection
        dest_label = QLabel("Destination Folder:")
        dest_label.setStyleSheet("font-weight: bold; color: #555555;")
        container_layout.addWidget(dest_label)

        dest_frame = QFrame()
        dest_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #dcdcdc;
                border-radius: 8px;
                background-color: #ffffff;
            }
        """)

        dest_layout = QHBoxLayout(dest_frame)
        dest_layout.setContentsMargins(12, 12, 12, 12)

        self.dest_edit = QLineEdit()
        self.dest_edit.setPlaceholderText("Select destination folder...")
        self.dest_edit.setReadOnly(True)
        self.dest_edit.setMinimumHeight(14)
        self.dest_edit.setStyleSheet(
            "border: none; font-size: 14px; background: transparent; color: black;")

        dest_browse = QPushButton()
        dest_browse.setIcon(QIcon.fromTheme("folder"))
        dest_browse.setIconSize(QSize(18, 18))  # Kích thước icon phù hợp
        dest_browse.setFixedSize(30, 30)  # Giữ nút nhỏ gọn, chỉ vừa với icon
        dest_browse.setStyleSheet(
            "background: #4a86e8; border: none; border-radius:5px; margin-bottom: 10px; margin-left: 10px;")  # Xóa nền và viền
        dest_browse.clicked.connect(self.select_dest_folder)

        dest_layout.addWidget(self.dest_edit)
        dest_layout.addWidget(dest_browse)
        container_layout.addWidget(dest_frame)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #555555; font-size: 14px;")
        container_layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 5px;
                background-color: #e0e0e0;
                height: 10px;
                text-align: center;
            }
            QProgressBar::chunk {
                border-radius: 5px;
                background-color: #4a86e8;
            }
        """)
        container_layout.addWidget(self.progress_bar)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        self.start_button = QPushButton("Start Classification")
        self.start_button.setEnabled(False)
        self.start_button.setStyleSheet(
            "background-color: #5cb85c; color: white; padding: 10px; border-radius: 8px; font-weight: bold;")
        self.start_button.clicked.connect(self.start_classification)

        self.open_button = QPushButton("Open Results")
        self.open_button.setEnabled(False)
        self.open_button.setStyleSheet(
            "background-color: #f0ad4e; color: white; padding: 10px; border-radius: 8px; font-weight: bold;")
        self.open_button.clicked.connect(self.open_results)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.open_button)
        container_layout.addLayout(button_layout)

        main_layout.addWidget(container)
        self.setLayout(main_layout)

    def select_source_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if folder:
            self.source_edit.setText(folder)
            self.check_ready()

    def select_dest_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Destination Folder")
        if folder:
            self.dest_edit.setText(folder)
            self.check_ready()

    def check_ready(self):
        if self.source_edit.text() and self.dest_edit.text():
            self.start_button.setEnabled(True)
        else:
            self.start_button.setEnabled(False)

    def start_classification(self):
        source_folder = self.source_edit.text()
        dest_folder = self.dest_edit.text()

        if not os.path.exists(source_folder):
            QMessageBox.warning(self, "Error", "Source folder does not exist.")
            return

        # Check if source folder has images
        has_images = False
        for file in os.listdir(source_folder):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                has_images = True
                break

        if not has_images:
            QMessageBox.warning(self, "No Images Found",
                                "No image files found in the source folder.")
            return

        # Disable UI elements during processing
        self.start_button.setEnabled(False)
        self.source_edit.setEnabled(False)
        self.dest_edit.setEnabled(False)

        # Create and start worker thread
        self.worker = ImageClassificationWorker(source_folder, dest_folder)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_classification_finished)
        self.worker.error.connect(self.on_classification_error)
        self.worker.start()

    def on_classification_finished(self):
        self.source_edit.setEnabled(True)
        self.dest_edit.setEnabled(True)
        self.start_button.setEnabled(True)
        self.open_button.setEnabled(True)
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Success")
        msg_box.setText("Phân loại ảnh thành công!")
        msg_box.setStyleSheet("""
    QMessageBox { background-color: white; }
    QLabel { color: black; font-size: 14px; }  /* Đảm bảo chữ có màu đen */
    QPushButton { background-color: #4CAF50; color: white; padding: 5px; border-radius: 5px; }
""")
        msg_box.exec()

    def on_classification_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.source_edit.setEnabled(True)
        self.dest_edit.setEnabled(True)
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Ready")

    def open_results(self):
        dest_folder = self.dest_edit.text()
        if os.path.exists(dest_folder):
            open_folder(dest_folder)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Thư mục lưu trữ không tồn tại!")
            msg.setWindowTitle("Error")
            # Đảm bảo màu chữ là đen
            msg.setStyleSheet("QLabel { color: black; }")
            msg.exec()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ImageClassificationApp()
    window.show()
    sys.exit(app.exec())
