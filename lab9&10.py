import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from retinaface import RetinaFace

# Define your dataset directory
base_dir = "DriverAttentionDataset"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
categories = os.listdir(train_dir)

class DriverAttentionMonitor:
    """
    An enhanced Driver Attention Monitoring system using RetinaFace.
    """
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size  # You can use this if you later classify faces

    def _detect_faces_retinaface(self, image):
        """
        Uses RetinaFace to detect faces in the input image.
        Returns a list of bounding boxes: [x, y, w, h]
        """
        try:
            faces = RetinaFace.detect_faces(image)
            results = []
            for key in faces:
                x1, y1, x2, y2 = faces[key]['facial_area']
                w, h = x2 - x1, y2 - y1
                results.append([x1, y1, w, h])
            return np.array(results)
        except Exception as e:
            print("❌ RetinaFace detection failed:", e)
            return []

    def _enhance_brightness(self, img, alpha=1.5, beta=10):
        """Increases brightness and contrast."""
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def _select_driver_face(self, faces):
        """Selects the largest detected face (assumed to be the driver)."""
        if len(faces) == 0:
            return []
        return [max(faces, key=lambda f: f[2] * f[3])]

    def process_image(self, image_path):
        """
        Processes an image, detects the driver face, retries with brightness enhancement if needed.
        """
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Failed to load image:", image_path)
            return None, None, "Error"

        # First attempt
        driver_faces = self._detect_faces_retinaface(img)
        driver_face = self._select_driver_face(driver_faces)
        method = "RetinaFace"
        image_for_display = img

        # Retry with brightness enhancement if failed
        if len(driver_face) == 0:
            print("⚠️ Initial detection failed. Trying again with brightness enhancement...")
            bright_img = self._enhance_brightness(img)
            driver_faces = self._detect_faces_retinaface(bright_img)
            driver_face = self._select_driver_face(driver_faces)
            if len(driver_face) > 0:
                method += " + Brightness"
                image_for_display = bright_img

        return image_for_display, driver_face, method

def demonstrate_system_on_split_data(data_directory, categories_list):
    """
    Demonstrates face detection on a randomly selected image.
    """
    monitor = DriverAttentionMonitor()

    category = random.choice(categories_list)
    category_path = os.path.join(data_directory, category)
    image_file = random.choice(os.listdir(category_path))
    image_path = os.path.join(category_path, image_file)

    print(f"\n📂 Dataset: {os.path.basename(data_directory)}")
    print(f"📸 Image: {image_file} (Category: {category})")

    processed_image, driver_face, method = monitor.process_image(image_path)

    if processed_image is None:
        print("❌ Skipping visualization due to error.")
        return

    # Display results
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

    if len(driver_face) > 0:
        x, y, w, h = driver_face[0]
        rect = plt.Rectangle((x, y), w, h, fill=False, color='lime', linewidth=3)
        plt.gca().add_patch(rect)
        plt.text(x, y - 10, f'Driver ({method})', color='lime', fontsize=12, fontweight='bold')
        print(f"✅ Driver face detected using: {method}")
    else:
        print("❌ No driver face detected after retries.")

    plt.title(f"Driver Attention Monitoring - {method}")
    plt.axis('off')
    plt.show()

# 🔁 Run the demo on training set
demonstrate_system_on_split_data(train_dir, categories)
# --- Test the system on five specific categories (one image each) ---
selected_categories = ['drowsy', 'distracted', 'phone', 'cigeratte', 'natural']

print("\n🔍 Testing on one image from each of the following categories:")
print(", ".join(selected_categories))

for cat in selected_categories:
    if cat in categories:
        demonstrate_system_on_split_data(train_dir, [cat])
    else:
        print(f"⚠️ Category '{cat}' not found in your dataset.")

