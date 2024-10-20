import os
import cv2
import pytesseract
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from src.utils import download_image, parse_string
from src.constants import entity_unit_map, allowed_units
from src.sanity import sanity_check
from pathlib import Path
import urllib.request
import time

# Setup paths for Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Adjust path as needed

# Preprocess image to improve OCR accuracy
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph

# Perform text detection using Tesseract OCR on the preprocessed image
def detect_text_tesseract(image):
    preprocessed_image = preprocess_image(image)
    text = pytesseract.image_to_string(preprocessed_image)
    return text

# Download and process image, then detect text
def process_image(image_url, save_folder='images/'):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    image_path = download_image(image_url, save_folder=save_folder)

    if image_path is None:
        print(f"Skipping image: {image_url} (failed to download)")
        return ""

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return ""

    detected_text = detect_text_tesseract(image)
    os.remove(image_path)
    return detected_text


# Main function to process all images in the dataset
def run_predictions_and_save(input_file, output_file):
    df = pd.read_csv(input_file)
    df['prediction'] = df.apply(lambda row: predict_entity_value(row), axis=1)
    df[['index', 'prediction']].to_csv(output_file, index=False)

# Predict entity value based on image text
def predict_entity_value(row):
    image_url = row['image_link']
    detected_value = process_image(image_url)
    return detected_value if detected_value else ""

# Main execution point
def main():
    train_file = 'dataset/train.csv'
    output_file = 'output_predictions.csv'
    run_predictions_and_save(train_file, output_file)

if __name__ == "__main__":
    main()