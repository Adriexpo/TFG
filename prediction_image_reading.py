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
images_directory = os.path.join(current_directory, 'Testing_Images')
crops_directory = os.path.join(current_directory, 'Testing_Images/Testing_Regions')

# Create the crops directory if it doesn't exist.
os.makedirs(crops_directory, exist_ok=True)

# Counters
processed_images = 0
incomplete_data_count = 0
confident_guesses = 0  # Counter for confident guesses

# Initialize log data
log_entries = []

# Function to find image in subdirectories
def find_image_in_subdirectories(root_directory, filename):
    for dirpath, _, filenames in os.walk(root_directory):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

# Function to save log entries periodically
def save_log_entries(entries, index):
    with open(os.path.join(current_directory, f'output_log_part_{index}.json'), 'w') as f:
        json.dump({"Log Entries": entries, "Final Summary (Manual)": None}, f, indent=4)

# Function to merge all part files into a single JSON file
def merge_json_files(output_directory, part_file_prefix, final_file_name):
    all_entries = []
    
    # List all files that match the part file prefix
    part_files = [f for f in os.listdir(output_directory) if f.startswith(part_file_prefix) and f.endswith('.json')]
    
    # Read and merge each part file
    for part_file in sorted(part_files):  # Sorting to ensure the order of merging
        part_file_path = os.path.join(output_directory, part_file)
        with open(part_file_path, 'r') as f:
            data = json.load(f)
            if "Log Entries" in data:
                all_entries.extend(data["Log Entries"])
    
    # Read the final summary
    final_summary_file_path = os.path.join(output_directory, 'output_log_summary.json')
    with open(final_summary_file_path, 'r') as f:
        final_summary = json.load(f)
    
    # Write the combined data and final summary to a single JSON file
    unified_file_path = os.path.join(output_directory, final_file_name)
    with open(unified_file_path, 'w') as f:
        json.dump({"Log Entries": all_entries, "Final Summary": final_summary["Final Summary (Manual)"]}, f, indent=4)
    
    # Delete the part files and summary file
    for part_file in part_files:
        os.remove(os.path.join(output_directory, part_file))
    
    os.remove(final_summary_file_path)
    
    print(f"Merged JSON file saved as {unified_file_path}")

# Loop through each image in the JSON
chunk_size = 100  # Number of entries per chunk
chunk_index = 1
current_chunk_entries = []

for image_id, image_info in json_data.items():
    image_filename = image_info['filename']
    image_path = find_image_in_subdirectories(images_directory, image_filename)
    
    # Check if the image was found
    if image_path is None:
        log_entry = {
            "Processing Image": image_filename,
            "Extracted Tuple": None,
            "Extracted Name": None,
            "Predicted Name": None,
            "Extracted Set Code": None,
            "Associated Set": None,
            "Similarity Score": None,
            "Similarity above threshold": "No",
            "Condition": "Awaiting condition prediction"
        }
        current_chunk_entries.append(log_entry)
        continue

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        log_entry = {
            "Processing Image": image_filename,
            "Extracted Tuple": None,
            "Extracted Name": None,
            "Predicted Name": None,
            "Extracted Set Code": None,
            "Associated Set": None,
            "Similarity Score": None,
            "Similarity above threshold": "No",
            "Condition": "Awaiting condition prediction"
        }
        current_chunk_entries.append(log_entry)
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
    
    if extracted_name and extracted_set and extracted_id:  # If all necessary data has been correctly extracted
        card_tuple = (extracted_name, extracted_set, extracted_id)
        
        # Use the closest set to determine the CSV file
        closest_set = find_closest_set(extracted_set)

        predicted_name = None  # Initialize predicted_name variable

        if closest_set:  # If a similarity can be determined within the group of valid sets
            csv_filename = f"{closest_set}.csv"
            csv_folder_path = os.path.join(current_directory, 'Card_Set_Info')
            csv_file_path = os.path.join(csv_folder_path, csv_filename)
            
            if os.path.exists(csv_file_path):  # If the CSV file is found
                # Load CSV file
                with open(csv_file_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    csv_data = list(csv_reader)
                
                # Find closest match for Name in the loaded CSV data
                predicted_name, similarity_score = find_closest_name(csv_data, extracted_name)
                
                # Determine a similarity threshold
                similarity_threshold = 0.6

                # Determine if similarity is above threshold
                similarity_above_threshold = "Yes" if similarity_score >= similarity_threshold else "No"
                if similarity_above_threshold == "Yes":
                    confident_guesses += 1  # Increment confident guesses counter
                
                # Create log entry
                log_entry = {
                    "Processing Image": image_filename,
                    "Extracted Tuple": card_tuple,
                    "Extracted Name": extracted_name,
                    "Predicted Name": predicted_name,  # Include the predicted name
                    "Extracted Set Code": extracted_set,
                    "Associated Set": closest_set,
                    "Similarity Score": f"{similarity_score:.2f}",
                    "Similarity above threshold": similarity_above_threshold,
                    "Condition": "Awaiting condition prediction"
                }
            else:
                log_entry = {
                    "Processing Image": image_filename,
                    "Extracted Tuple": card_tuple,
                    "Extracted Name": extracted_name,
                    "Predicted Name": predicted_name,  # Include the predicted name
                    "Extracted Set Code": extracted_set,
                    "Associated Set": None,
                    "Similarity Score": None,
                    "Similarity above threshold": "No",
                    "Condition": "Awaiting condition prediction"
                }
        else:
            log_entry = {
                "Processing Image": image_filename,
                "Extracted Tuple": card_tuple,
                "Extracted Name": extracted_name,
                "Predicted Name": predicted_name,  # Include the predicted name
                "Extracted Set Code": extracted_set,
                "Associated Set": None,
                "Similarity Score": None,
                "Similarity above threshold": "No",
                "Condition": "Awaiting condition prediction"
            }
    else: 
        incomplete_data_count += 1
        log_entry = {
            "Processing Image": image_filename,
            "Extracted Tuple": None,
            "Extracted Name": None,
            "Predicted Name": None,
            "Extracted Set Code": None,
            "Associated Set": None,
            "Similarity Score": None,
            "Similarity above threshold": "No",
            "Condition": "Awaiting condition prediction"
        }

    current_chunk_entries.append(log_entry)
    
    # Periodically save log entries to avoid high memory usage
    if len(current_chunk_entries) >= chunk_size:
        save_log_entries(current_chunk_entries, chunk_index)
        chunk_index += 1
        current_chunk_entries = []

# Save any remaining log entries after processing all images
if current_chunk_entries:
    save_log_entries(current_chunk_entries, chunk_index)

# Write final summary
final_summary = {
    "Number of Incomplete Data": incomplete_data_count,
    "Number of Confident Guesses": confident_guesses,
    "Number of Processed Images": processed_images,
    "Accuracy": f"{(confident_guesses / (processed_images - incomplete_data_count) * 100) if processed_images - incomplete_data_count > 0 else 0:.2f}%"
}

# Save final summary to a separate file
with open(os.path.join(current_directory, 'output_log_summary.json'), 'w') as f:
    json.dump({"Log Entries": [], "Final Summary (Manual)": final_summary}, f, indent=4)

# Merge all part files into a single JSON file and delete the part files
merge_json_files(current_directory, 'output_log_part_', 'output_log_combined.json')

# Calculate and print elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Final Summary (Manual):\nNumber of Processed Images: {processed_images}\n")
print(f"Number of Confident Guesses: {confident_guesses}\n")
print(f"Number of Incomplete Data: {incomplete_data_count}\n")
print(f"Accuracy: {final_summary['Accuracy']}")
print(f"Running time: {elapsed_time:.2f} seconds.")
