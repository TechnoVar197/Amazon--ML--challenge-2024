import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras import backend as K
from src.utils import download_images, parse_string
from src.constants import allowed_units
from src.sanity import sanity_check

# Step 1: Load the test dataset
test_df = pd.read_csv('dataset/train.csv')

# Step 2: Define the CNN model
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')  # Output a single numeric value
    ])
    return model

# Step 3: Custom F1-score metric
def f1_metric(y_true, y_pred):
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Step 4: Preprocess the image for prediction (resizing, normalization)
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    image = cv2.resize(image, (128, 128))  # Resize to a standard size
    image = image.astype('float32') / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Step 5: Function to validate and format prediction with units
def validate_prediction(value, entity_name):
    if value is None or np.isnan(value):
        return ""  # Return blank if the prediction is invalid

    # Retrieve allowed units based on the entity name
    possible_units = allowed_units.intersection(entity_unit_map[entity_name])
    if len(possible_units) == 0:
        return ""  # No valid units for this entity

    unit = list(possible_units)[0]  # Choose the first allowed unit
    return f"{value:.2f} {unit}"  # Format the output with two decimal places and unit

# Step 6: Define a function to download images, process them, and predict entity values
def predict_entity_values(test_df, model):
    predictions = []
    
    # Download and predict for each row
    for _, row in test_df.iterrows():
        image_link = row['image_link']
        entity_name = row['entity_name']

        # Download the image
        download_images([image_link], download_folder='test_images', allow_multiprocessing=False)

        # Preprocess the image
        image_filename = os.path.basename(image_link)
        image_path = os.path.join('test_images', image_filename)
        if os.path.exists(image_path):
            image = preprocess_image(image_path)

            # Predict the numeric value using the model
            predicted_value = model.predict(image).flatten()[0]

            # Validate and format the prediction
            formatted_prediction = validate_prediction(predicted_value, entity_name)
            predictions.append(formatted_prediction)

            # Remove the image after processing
            os.remove(image_path)
        else:
            predictions.append("")  # Append empty string if the image does not exist

    return predictions

# Step 7: Prepare the dataset and run predictions
if __name__ == "__main__":
    # Image shape for the CNN model
    img_rows, img_cols = 128, 128
    input_shape = (img_rows, img_cols, 1)  # Grayscale image with 1 channel

    # Create and compile the model
    model = create_model(input_shape)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=[f1_metric])

    # Load pre-trained weights if available (or you can train the model before prediction)
    # model.load_weights('path_to_your_pretrained_weights.h5')

    # Predict entity values for the test data
    test_predictions = predict_entity_values(test_df, model)

    # Save predictions in the correct format
    test_df['prediction'] = test_predictions
    test_df[['index', 'prediction']].to_csv('test_out.csv', index=False)

    # Perform sanity check before submission
    sanity_check('dataset/test.csv', 'test_out.csv')

    print("Predictions saved and sanity check passed.")
