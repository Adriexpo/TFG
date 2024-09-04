import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model when the module is imported
model_path = os.path.join(os.path.dirname(__file__), 'card_condition_regressor.keras')
model = tf.keras.models.load_model(model_path)

# Function to make predictions on new images and update the JSON file
def predict_image(img_path, json_file_path):
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    
    img = image.load_img(img_path, target_size=(128, 128))  # Ensure this matches the image size used during training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Remember to rescale as done during training
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    
    classes = ['Mint', 'Excellent', 'Good', 'Played', 'Poor']
    predicted_condition = classes[class_idx]
    
    # Update the JSON data with the predicted condition
    image_filename = os.path.basename(img_path)

    print(f"Analyzing image: {image_filename}")
    
    # Check if json_data is a dictionary or a list
    if isinstance(json_data, dict):
        log_entries = json_data.get("Log Entries", [])
    elif isinstance(json_data, list):
        log_entries = json_data
    else:
        raise ValueError("Unexpected JSON structure")

    match_found = False
    for image_info in log_entries:
        # Check if 'Processing Image' matches the filename
        if 'Processing Image' in image_info:
            if image_info['Processing Image'] == image_filename:
                image_info['Condition'] = predicted_condition
                match_found = True
                print(f"Match found for image: {image_filename}. Condition updated to: {predicted_condition}")
                break

    if not match_found:
        print(f"No match found for image: {image_filename} in the JSON file.")

    # Save the updated JSON data back to the file
    with open(json_file_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    return predicted_condition
