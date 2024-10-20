import pandas as pd
import os
import cv2
import easyocr
import re
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from src.utils import download_images, parse_string
from src.constants import allowed_units
from src.sanity import sanity_check

# Step 1: Load the training and test datasets
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

# Step 2: Create the folder if it doesn't exist, then download images
def create_folder_if_needed(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")

# Create folders for train and test images
create_folder_if_needed('train_images')
create_folder_if_needed('test_images')

# Download images using utils.py
download_images(train_df['image_link'], download_folder='train_images')
download_images(test_df['image_link'], download_folder='test_images')

# Step 3: Initialize EasyOCR reader and OpenCV parameters
reader = easyocr.Reader(['en'])  # Initialize EasyOCR with English language support

# Function to preprocess the image (using OpenCV)
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

# Function to extract entity values using EasyOCR
def extract_entity_from_image(image_path):
    if not os.path.exists(image_path):
        return ""

    processed_image = preprocess_image(image_path)
    results = reader.readtext(processed_image)
    extracted_text = ' '.join([text[1] for text in results])

    # Delete the image after processing
    try:
        os.remove(image_path)
        print(f"Deleted image: {image_path}")
    except Exception as e:
        print(f"Error deleting image: {image_path}. Error: {str(e)}")

    return extracted_text

# Function to parse and extract specific entity values from text
def extract_value_from_text(text, entity_name):
    patterns = {
        'item_weight': r'(\d+(\.\d+)?\s*(gram|kilogram|mg|ounce|pound))',
        'height': r'(\d+(\.\d+)?\s*(centimetre|inch|foot|metre|yard))',
        'width': r'(\d+(\.\d+)?\s*(centimetre|inch|foot|metre|yard))',
        'voltage': r'(\d+(\.\d+)?\s*(volt|kilovolt|millivolt))',
        'wattage': r'(\d+(\.\d+)?\s*(watt|kilowatt))'
    }
    
    pattern = patterns.get(entity_name, "")
    match = re.search(pattern, text)
    
    if match:
        return match.group(0)
    return ""

# Step 4: Define a function to make predictions on the test data
def predict(test_df):
    predictions = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_path = os.path.join('test_images', os.path.basename(row['image_link']))
        extracted_text = extract_entity_from_image(image_path)
        entity_name = row['entity_name']
        prediction = extract_value_from_text(extracted_text, entity_name)
        predictions.append(prediction if prediction else "")
    
    return predictions

# Step 5: Make predictions on the test data
test_predictions = predict(test_df)

# Step 6: Prepare the submission file
test_df['prediction'] = test_predictions
test_df[['index', 'prediction']].to_csv('test_out.csv', index=False)

# Step 7: Perform sanity check before submission
sanity_check('dataset/test.csv', 'test_out.csv')

# Step 8: F1 Score Calculation - Comparing predictions with ground truth
y_true = train_df['entity_value']  # Ground truth from train.csv
y_pred = predict(train_df)  # Predictions on train images

# Function to preprocess and normalize the predictions for comparison
def preprocess_for_f1(text):
    try:
        number, unit = parse_string(text)
        return f"{number} {unit}"  # Normalize to "x unit" format
    except Exception:
        return ""  # Return empty if the string is not in valid format

# Normalize ground truth and predictions
y_true = y_true.apply(preprocess_for_f1)
y_pred = pd.Series(y_pred).apply(preprocess_for_f1)

# Calculate Precision, Recall, and F1 Score
precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)

# Print scores
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# End of script
