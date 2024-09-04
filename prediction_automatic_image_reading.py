import os
import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression
import difflib
import time
import csv
import json

from set_prediction import find_closest_set
from name_prediction import find_closest_name

# Record the start time
start_time = time.time()

# Load the pre-trained EAST text detector model
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

log_entries = []
log_batch_size = 100
batch_number = 1
incomplete_data_count = 0
confident_guesses = 0
processed_images = 0

def sort_bounding_boxes(boxes):
    return sorted(boxes, key=lambda box: (box[1], box[0]))

def group_bounding_boxes(boxes, max_gap=25):
    grouped_boxes = []
    current_group = []

    for i, box in enumerate(boxes):
        if not current_group:
            current_group.append(box)
            continue

        _, prev_startY, _, prev_endY = current_group[-1]
        _, startY, _, endY = box

        if abs(startY - prev_endY) <= max_gap:
            current_group.append(box)
        else:
            grouped_boxes.append(current_group)
            current_group = [box]

    if current_group:
        grouped_boxes.append(current_group)

    return grouped_boxes

def merge_text_from_boxes(image, boxes, rW, rH, boundary=3):
    merged_text = ""

    for box in boxes:
        (startX, startY, endX, endY) = box

        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # Adjust boundaries to ensure they are within image bounds
        startY_bound = max(0, startY - boundary)
        endY_bound = min(image.shape[0] - 1, endY + boundary)
        startX_bound = max(0, startX - boundary)
        endX_bound = min(image.shape[1] - 1, endX + boundary)

        # Extract text region from the original image
        text_region = image[startY_bound:endY_bound, startX_bound:endX_bound]

        if text_region.size > 0:
            text_region = cv2.cvtColor(text_region.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(text_region)
            merged_text += text.strip() + " "
        else:
            print(f"Empty region for box: {box}")

    return merged_text.strip()

def text_detector(image):
    orig = image.copy()
    (H, W) = image.shape[:2]

    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)
    boxes = sort_bounding_boxes(boxes)
    grouped_boxes = group_bounding_boxes(boxes)

    merged_text_box_1 = ""

    for i, group in enumerate(grouped_boxes):
        # Sort bounding boxes within each group by their x-coordinate
        group = sorted(group, key=lambda box: box[0])
        
        merged_text = merge_text_from_boxes(orig, group, rW, rH)

        if i == 0:  # If it's the first group (Box 1)
            merged_text_box_1 = merged_text

        for box in group:
            (startX, startY, endX, endY) = box

            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
            label = f"Box {i+1}"
            cv2.putText(orig, label, (startX, startY - 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(orig, merged_text, (group[-1][0], group[-1][3] + 0), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)

    return orig, merged_text_box_1

def save_log_batch(log_entries, batch_number):
    log_filename = f"output_log_part_{batch_number}.json"
    with open(log_filename, 'w') as log_file:
        json.dump({"Log Entries": log_entries}, log_file, indent=4)

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
        json.dump({"Log Entries": all_entries, "Final Summary": final_summary["Final Summary (Automatic)"]}, f, indent=4)
    
    # Delete the part files and summary file
    for part_file in part_files:
        os.remove(os.path.join(output_directory, part_file))
    
    os.remove(final_summary_file_path)
    
    print(f"Merged JSON file saved as {unified_file_path}")

def process_images_from_folder(folder_path):
    global processed_images, confident_guesses, incomplete_data_count, batch_number  # Declare batch_number as global
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # Create windows with the ability to resize

    #cv2.namedWindow("Orig Image", cv2.WINDOW_NORMAL)  # UNCOMMENT TO SHOW IMAGE WINDOWS
    #cv2.namedWindow("Text Detection", cv2.WINDOW_NORMAL)  # UNCOMMENT TO SHOW IMAGE WINDOWS

    # Set the window size to a sensible size
    #window_width = 650  # UNCOMMENT TO SHOW IMAGE WINDOWS
    #window_height = 850  # UNCOMMENT TO SHOW IMAGE WINDOWS
    #cv2.resizeWindow("Orig Image", window_width, window_height)  # UNCOMMENT TO SHOW IMAGE WINDOWS
    #cv2.resizeWindow("Text Detection", window_width, window_height)  # UNCOMMENT TO SHOW IMAGE WINDOWS

    for root, dirs, files in os.walk(folder_path):
        
        # Skip Testing_Regions folder which does not belong to this model.
        if 'Testing_Regions' in dirs:
            dirs.remove('Testing_Regions')

        image_files = [f for f in files if f.lower().endswith(valid_extensions)]
        
        for image_file in image_files:
            image_path = os.path.join(root, image_file)
            img = cv2.imread(image_path)
            
            if img is not None:
                imageO = cv2.resize(img, (1504, 2016), interpolation=cv2.INTER_AREA)
                orig = cv2.resize(img, (1504, 2016), interpolation=cv2.INTER_AREA)
                textDetected, merged_text_box_1 = text_detector(imageO)

                # Get the folder name (folderSet)
                folderSet = os.path.basename(root)

                # Show the images in the resized windows
                #cv2.imshow("Orig Image", orig)  # UNCOMMENT TO SHOW IMAGE WINDOWS
                #cv2.imshow("Text Detection", textDetected)  # UNCOMMENT TO SHOW IMAGE WINDOWS
                processed_images += 1

                closest_set = find_closest_set(folderSet)

                csv_filename = f"{closest_set}.csv"
                csv_folder_path = os.path.join(current_directory, 'Card_Set_Info')
                csv_file_path = os.path.join(csv_folder_path, csv_filename)

                if os.path.exists(csv_file_path):
                    # Load CSV file
                    with open(csv_file_path, 'r') as csv_file:
                        csv_reader = csv.reader(csv_file)
                        csv_data = list(csv_reader)
                    
                    # Find closest match for Name in the loaded CSV data
                    closest_name, similarity_score = find_closest_name(csv_data, merged_text_box_1)

                    # Determine a similarity threshold.
                    similarity_threshold = 0.6

                    log_entry = {
                        "Processing Image": image_file,
                        "Extracted Tuple": [merged_text_box_1, folderSet, "unknown"],  # You might want to add other elements
                        "Extracted Name": merged_text_box_1,
                        "Predicted Name": closest_name,
                        "Extracted Set Code": folderSet,
                        "Associated Set": closest_set,
                        "Similarity Score": f"{similarity_score:.2f}",
                        "Similarity above threshold": "Yes" if similarity_score >= similarity_threshold else "No",
                        "Condition": "Awaiting condition prediction"
                    }
                    log_entries.append(log_entry)

                    if similarity_score >= similarity_threshold:
                        confident_guesses += 1

                    if len(log_entries) >= log_batch_size:
                        save_log_batch(log_entries, batch_number)
                        log_entries.clear()
                        batch_number += 1
                else:
                    incomplete_data_count += 1

                #time.sleep(2)

                k = cv2.waitKey(30)
                if k == 27:
                    break
            else:
                print(f"Failed to read image: {image_file}")

    if log_entries:
        save_log_batch(log_entries, batch_number)
    
    # Write final summary
    final_summary = {
        "Number of Incomplete Data": incomplete_data_count,
        "Number of Confident Guesses": confident_guesses,
        "Number of Processed Images": processed_images,
        "Accuracy": f"{(confident_guesses / (processed_images - incomplete_data_count) * 100) if processed_images - incomplete_data_count > 0 else 0:.2f}%"
    }

    with open(os.path.join(current_directory, 'output_log_summary.json'), 'w') as f:
        json.dump({"Log Entries": [], "Final Summary (Automatic)": final_summary}, f, indent=4)

    # Merge all JSON log batches into a single final JSON file and delete the part files
    merge_json_files(current_directory, 'output_log_part_', 'output_log_automatic_combined.json')

    #cv2.destroyAllWindows()  # UNCOMMENT TO SHOW IMAGE WINDOWS
    # Print the final summary
    print(f"Final Summary (Automatic):\nNumber of Processed Images: {processed_images}\n")
    print(f"Number of Confident Guesses: {confident_guesses}\n")
    print(f"Number of Incomplete Data: {incomplete_data_count}\n")
    print(f"Accuracy: {final_summary['Accuracy']}")
    # Record the end time and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Running time: {elapsed_time:.2f} seconds.")


# Specify the folder containing images
current_directory = os.path.dirname(os.path.abspath(__file__))
images_directory = os.path.join(current_directory, 'Testing_Images')  # Change this to the folder path where your images are located
process_images_from_folder(images_directory)