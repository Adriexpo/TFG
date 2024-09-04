import os
import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression
import difflib
import time
import csv

from set_prediction import find_closest_set
from name_prediction import find_closest_name

# Record the start time
start_time = time.time()

log_correct = open('correct_automatic.txt', 'w')
log_incorrect = open('wrong_automatic.txt', 'w')

# Load the pre-trained EAST text detector model
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

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

def process_images_from_folder(folder_path):
    processed_images = 0
    over_threshold = 0
    correct_predictions = 0
    incorrect_predictions = 0
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # Create windows with the ability to resize
    cv2.namedWindow("Orig Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Text Detection", cv2.WINDOW_NORMAL)

    # Set the window size to a sensible size
    window_width = 650
    window_height = 850
    cv2.resizeWindow("Orig Image", window_width, window_height)
    cv2.resizeWindow("Text Detection", window_width, window_height)

    for root, dirs, files in os.walk(folder_path):
        
        # Skip Training_Regions folder which does not belong to this model.
        if 'Training_Regions' in dirs:
            dirs.remove('Training_Regions')

        image_files = [f for f in files if f.lower().endswith(valid_extensions)]
        
        for image_file in image_files:
            image_path = os.path.join(root, image_file)
            img = cv2.imread(image_path)
            
            if img is not None:
                imageO = cv2.resize(img, (1504, 2016), interpolation=cv2.INTER_AREA)
                orig = cv2.resize(img, (1504, 2016), interpolation=cv2.INTER_AREA)
                textDetected, merged_text_box_1 = text_detector(imageO)

                # Extract expected name from filename
                expected_name = image_file.split(';')[0]

                # Get the folder name (folderSet)
                folderSet = os.path.basename(root)

                # Show the images in the resized windows
                cv2.imshow("Orig Image", orig)
                cv2.imshow("Text Detection", textDetected)
                processed_images += 1

                closest_set = find_closest_set(folderSet)

                csv_filename = f"{closest_set}.csv"
                csv_folder_path = os.path.join(current_directory, 'Card_Set_Info')
                csv_file_path = os.path.join(csv_folder_path, csv_filename)
                #print(csv_file_path)

                if os.path.exists(csv_file_path):
                    # Load CSV file
                    with open(csv_file_path, 'r') as csv_file:
                        csv_reader = csv.reader(csv_file)
                        csv_data = list(csv_reader)
                    
                    # Find closest match for Name in the loaded CSV data
                    closest_name, similarity_score = find_closest_name(csv_data, merged_text_box_1)
                    #print(closest_name)

                    # Determine a similarity threshold.
                    similarity_threshold = 0.7

                    if closest_name.lower() == expected_name.lower():
                        correct_predictions += 1

                        log_correct.write(f"Processing Image: {image_file}                  Number: {processed_images}\n")
                        log_correct.write(f"Extracted Name: {merged_text_box_1}\n")
                        log_correct.write(f"Expected Name: {expected_name}\n")

                        log_correct.write(f"\nSimilarity Score between Extracted and Expected Name: {similarity_score:.2f}\n")
                        if similarity_score >= similarity_threshold:
                            log_correct.write(f"Similarity is above the threshold of {similarity_threshold}.\n")
                            over_threshold += 1
                        else:
                            log_correct.write(f"Similarity is below the threshold of {similarity_threshold}.\n")
                        

                        log_correct.write(f"\nThe set code for this card is: {folderSet}\n")
                        log_correct.write(f"The set associated with this card's code is: {closest_set}\n")
                        log_correct.write(f"\nFinal prediction was CORRECT for: {closest_name}\n")
                        log_correct.write(f" ------------------------------------------------------------------------------------\n\n")

                    else:
                        incorrect_predictions += 1

                        log_incorrect.write(f"Processing Image: {image_file}                  Number: {processed_images}\n")
                        log_incorrect.write(f"Extracted Name: {merged_text_box_1}\n")
                        log_incorrect.write(f"Expected Name: {expected_name}\n")

                        log_incorrect.write(f"\nSimilarity Score between Extracted and Expected Name: {similarity_score:.2f}\n")
                        if similarity_score >= similarity_threshold:
                            log_incorrect.write(f"Similarity is above the threshold of {similarity_threshold}.\n")
                            over_threshold += 1
                        else:
                            log_incorrect.write(f"Similarity is below the threshold of {similarity_threshold}.\n")

                        log_incorrect.write(f"\nThe set code for this card is: {folderSet}\n")
                        log_incorrect.write(f"The set associated with this card's code is: {closest_set}\n")
                        log_incorrect.write(f"\nFinal prediction was INCORRECT for: {closest_name}, it should have been {expected_name}\n")
                        log_incorrect.write(f" ------------------------------------------------------------------------------------\n\n")
                        
                            
                time.sleep(2)

                k = cv2.waitKey(30)
                if k == 27:
                    break
            else:
                print(f"Failed to read image: {image_file}")

    cv2.destroyAllWindows()
    print(f"Total processed images: {processed_images}")
    print(f"Total good similarity names: {over_threshold}")
    print(f"Total correctly identified names: {correct_predictions}")
    accuracy = correct_predictions / processed_images * 100
    print(f"The total accuracy is: {accuracy:.2f}%")

# Specify the folder containing images
current_directory = os.path.dirname(os.path.abspath(__file__))
images_directory = os.path.join(current_directory, 'Training_Images')  # Change this to the folder path where your images are located
process_images_from_folder(images_directory)

# Record the end time and calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Running time: {elapsed_time:.2f} seconds.")

# Close the log files
log_correct.close()
log_incorrect.close()