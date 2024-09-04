import os
import pytesseract
from PIL import Image
import json
import csv
import cv2
import numpy as np
import time

from set_prediction import find_closest_set
from name_prediction import find_closest_name

# Record the start time
start_time = time.time()

# Set Tesseract environment variable
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Load the JSON data
json_file_path = os.path.join(current_directory, 'BoundingBoxes_config_json.json')
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

# Define crops directory
images_directory = os.path.join(current_directory, 'Training_Images')
crops_directory = os.path.join(current_directory, 'Training_Images/Training_Regions')

# Create the crops directory if it doesn't exist.
os.makedirs(crops_directory, exist_ok=True)

# Counters.
processed_images = 0
over_threshold = 0
correctly_predicted = 0
wrongly_predicted = 0
incomplete_data_count = 0

# Open the output files in write mode
correct_output_file_path = os.path.join(current_directory, 'output.txt')
wrong_output_file_path = os.path.join(current_directory, 'wrong.txt')

log_correct = open('correct_manual.txt', 'w')
log_incorrect = open('wrong_manual.txt', 'w')

# Function to find image in subdirectories
def find_image_in_subdirectories(root_directory, filename):
    for dirpath, _, filenames in os.walk(root_directory):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

# Loop through each image in the JSON
for image_id, image_info in json_data.items():
    image_filename = image_info['filename']
    image_path = find_image_in_subdirectories(images_directory, image_filename)
    
    # Check if the image was found
    if image_path is None:
        print(f"Warning: Image file not found: {image_filename}")
        log_incorrect.write(f"\nProcessing image: {image_filename}\n")
        log_incorrect.write(f"Image file could not be found in any subdirectory of {images_directory}.\n")
        continue

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Warning: Cannot open image at path: {image_path}")
        log_incorrect.write(f"\nProcessing image: {image_filename}\n")
        log_incorrect.write(f"Image file could not be opened or found in the specified path {image_path}.\n")
        continue
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get regions
    regions = image_info['regions']
    
    # Initialize variables to store extracted data
    extracted_name = None
    extracted_set = None
    extracted_id = None
    
    # Loop through each region
    for region_info in regions:
        # Extract region attributes
        shape_attributes = region_info['shape_attributes']
        if shape_attributes['name'] == 'rect':
            x, y, width, height = shape_attributes['x'], shape_attributes['y'], \
                                    shape_attributes['width'], shape_attributes['height']
        elif shape_attributes['name'] == 'polygon':
            all_x, all_y = shape_attributes['all_points_x'], shape_attributes['all_points_y']
            x, y = min(all_x), min(all_y)
            width, height = max(all_x) - min(all_x), max(all_y) - min(all_y)
        else:
            continue  # Skip if shape not recognized
        
        box_name = region_info['region_attributes']['Boxes']
        
        # Crop the region from the image
        region_image = Image.open(image_path).crop((x, y, x + width, y + height))
        
        # Extract expected text from the filename
        name_expected_text, set_expected_text = image_id.split(";")
        set_expected_text = set_expected_text.split(".")[0]  # Stop reading at the first "."
        
        # Specify the engine mode using the config argument
        config = '--psm 8 --oem 2'  # Set engine mode to LSTM_ONLY (OEM_LSTM_ONLY)
        
        # Use Tesseract OCR to extract text from the region with specified config
        if "Set" in box_name or "Name" in box_name or "ID" in box_name: 
            # Convert region image to grayscale
            region_image_gray = cv2.cvtColor(np.array(region_image), cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            region_image_blur = cv2.GaussianBlur(region_image_gray, (5, 5), 0)
            
            # Use adaptive thresholding to enhance contrast
            _, region_image_thresh = cv2.threshold(region_image_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Convert the processed image back to PIL format
            region_image = Image.fromarray(region_image_thresh)
            
            # Save the cropped and processed image in the crops directory
            crop_filename = f"{image_id}_{box_name}.png"  # Modify the filename as needed
            crop_path = os.path.join(crops_directory, crop_filename)
            region_image.save(crop_path)
            
            # Extract text from the processed region
            extracted_text = pytesseract.image_to_string(region_image, lang='spa', config=config).strip()
            
            if "Set" in box_name:
                extracted_set = extracted_text
            elif "Name" in box_name:
                extracted_name = extracted_text
            elif "ID" in box_name:
                extracted_id = extracted_text

    processed_images += 1
    
    if extracted_name and extracted_set and extracted_id: # If all necessary data has been correctly extracted, create a tuple with said extracted data.
        
        card_tuple = (extracted_name, extracted_set, extracted_id)
        
        # Use the closest set to determine the CSV file
        closest_set = find_closest_set(extracted_set)

        if closest_set: # If a similarity can be determined within the group of valid sets, we open the CSV file associated with it.
            csv_filename = f"{closest_set}.csv"
            csv_folder_path = os.path.join(current_directory, 'Card_Set_Info')
            csv_file_path = os.path.join(csv_folder_path, csv_filename)
            
            if os.path.exists(csv_file_path):   #If the CSV file is found
                # Load CSV file
                with open(csv_file_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    csv_data = list(csv_reader)
                
                # Find closest match for Name in the loaded CSV data
                closest_name, similarity_score = find_closest_name(csv_data, extracted_name)
                
                # Determine a similarity threshold.
                similarity_threshold = 0.7

                # Extract the expected name from the image filename
                expected_name = image_filename.split(";")[0]

                # Output predicted card
                if closest_name:    #If a close name was found
                    if closest_name.lower() == expected_name.lower(): # If it matches the name of the card it is correctly predicted.
                        correctly_predicted += 1
                        log_correct.write(f"\nProcessing Image: {image_filename}                  Number: {processed_images}\n")
                        log_correct.write(f"Extracted Tuple: {card_tuple}\n")
                        
                        log_correct.write(f"\nExtracted Name: {extracted_name}\n")
                        log_correct.write(f"Expected Name: {expected_name}\n")

                        log_correct.write(f"\nSimilarity Score between Extracted and Expected Name: {similarity_score:.2f}\n")
                        if similarity_score >= similarity_threshold:
                            log_correct.write(f"Similarity is above the threshold of {similarity_threshold}.\n")
                            over_threshold += 1
                        else:
                            log_correct.write(f"Similarity is below the threshold of {similarity_threshold}.\n")

                        log_correct.write(f"\nPredicted set is: {closest_set}\n")
                        log_correct.write(f"The set associated with this card's code is: {closest_set}\n")

                        log_correct.write(f"Predicted card is: {closest_name}\n")
                        log_correct.write("-----------------------------------------------------------------------------------------\n\n")
                        
                    else:
                        wrongly_predicted += 1 # Otherwise, it means that the name prediction was wrong.
                        log_incorrect.write(f"\nProcessing Image: {image_filename}                  Number: {processed_images}\n")
                        log_incorrect.write(f"Extracted Tuple: {card_tuple}\n")
                        
                        log_incorrect.write(f"\nExtracted Name: {extracted_name}\n")
                        log_incorrect.write(f"Expected Name: {expected_name}\n")

                        log_incorrect.write(f"\nSimilarity Score between Extracted and Expected Name: {similarity_score:.2f}\n")
                        if similarity_score >= similarity_threshold:
                            log_incorrect.write(f"Similarity is above the threshold of {similarity_threshold}.\n")
                            over_threshold += 1
                        else:
                            log_incorrect.write(f"Similarity is below the threshold of {similarity_threshold}.\n")

                        log_incorrect.write(f"\nPredicted set is: {closest_set}\n")
                        log_incorrect.write(f"The set associated with this card's code is: {closest_set}\n")

                        log_incorrect.write(f"Predicted card is: {closest_name}\n")
                        log_incorrect.write("-----------------------------------------------------------------------------------------\n\n")
                else:
                    wrongly_predicted += 1
                    log_incorrect.write(f"\nProcessing image: {image_path}\n")
                    log_incorrect.write(f"Extracted Tuple: {card_tuple}\n")
                    log_incorrect.write(f"The extracted set was: {extracted_set}\n")
                    log_incorrect.write("No close match found in the CSV.\n")
                    log_incorrect.write("-----------------------------------------------------------------------------------------\n\n")
            else:
                # Receiving the prints found in this "else" statement means that the specified path did not lead to the desired CSV.
                # Check if the CSV is indeed found in the path that will be printed. 
                # The written set name in set_prediction.py may be the issue.
                wrongly_predicted += 1
                log_incorrect.write(f"\nProcessing image: {image_path}\n")
                log_incorrect.write(f"Extracted Tuple: {card_tuple}\n")
                log_incorrect.write(f"CSV file {csv_filename} not found.\n")
                log_incorrect.write("-----------------------------------------------------------------------------------------\n\n")
        else:
            # The extracted text from Set was so unclear that no clear enough similarity was found.
            # This might be an issue of the image itself or the Set Bounding Box of said image.
            wrongly_predicted += 1
            log_incorrect.write(f"\nProcessing image: {image_path}\n")
            log_incorrect.write(f"Extracted Tuple: {card_tuple}\n")
            log_incorrect.write("Closest set not determined.\n")
            log_incorrect.write("-----------------------------------------------------------------------------------------\n\n")
    else: 
        # Ending up in this else means that part of the tuple (either Name, Set or ID) was not found.
        # Usually means that a Bounding Box for any of those three was not specified in the JSON file.
        # Check your JSON file, find the specified image and make sure all 3 Bounding Boxes exist and are labeled accordingly.
        incomplete_data_count += 1
        log_incorrect.write(f"\nProcessing image: {image_path}\n")
        log_incorrect.write("Incomplete data extracted, cannot form card tuple.\n")
        log_incorrect.write("-----------------------------------------------------------------------------------------\n\n")

# Write final summary to output files.
log_correct.write("########################################################")
log_correct.write("\nFinal Summary:\n")
log_correct.write(f"Correctly Predicted: {correctly_predicted}\n")

log_incorrect.write("########################################################")
log_incorrect.write("\nFinal Summary:\n")
log_incorrect.write(f"Wrongly Predicted: {wrongly_predicted}\n")
log_incorrect.write(f"Number of 'Incomplete data extracted, cannot form card tuple.': {incomplete_data_count}\n")

# Calculate accuracy
total_predictions = correctly_predicted + wrongly_predicted
accuracy_percentage = (correctly_predicted / total_predictions) * 100 if total_predictions > 0 else 0
#Write final summary to terminal.
print(f"\nFinal Summary:\nCorrectly predicted: {correctly_predicted}\nWrongly Predicted: {wrongly_predicted}\nNumber of 'Incomplete data extracted, cannot form card tuple.': {incomplete_data_count}\n")

print(f"Accuracy: {accuracy_percentage:.2f}%")

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Running time: {elapsed_time:.2f} seconds.")

# Close the log files
log_correct.close()
log_incorrect.close()