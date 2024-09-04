import subprocess
import os
from condition.condition_prediction import predict_image
import json

# Example parameters
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)

# Define the images directory
images_directory = os.path.join(current_directory, 'Testing_Images')
crops_directory = os.path.join(images_directory, 'Testing_Regions')

def prediction_setup(json_file):

    # Define the JSON file path
    json_file_path = os.path.join(current_directory, json_file)

    # Iterate through the Testing_Images directory, excluding the Testing_Regions subdirectory
    for dirpath, dirnames, filenames in os.walk(images_directory):
        # Skip the Testing_Regions directory
        if dirpath == crops_directory:
            continue

        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dirpath, filename)
                prediction = predict_image(img_path, json_file_path)
                print(f"Image {img_path} is predicted as: {prediction}")

def only_condition(output_json_file):

    # Initialize an empty list to hold the image data
    images_data = []

    # Iterate through the Testing_Images directory, excluding the Testing_Regions subdirectory
    for dirpath, dirnames, filenames in os.walk(images_directory):
        # Skip the Testing_Regions directory
        if dirpath == crops_directory:
            continue

        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                
                # Create a dictionary for each image
                image_entry = {
                    "Processing Image": filename,
                    "Condition": "Awaiting condition."
                }

                # Add the image entry to the list
                images_data.append(image_entry)

    # Write the list of image data to the JSON file
    output_json_path = os.path.join(current_directory, output_json_file)
    with open(output_json_path, 'w') as json_file:
        json.dump(images_data, json_file, indent=4)

    print(f"JSON file '{output_json_file}' has been created with predictions.")


if __name__ == "__main__":
    
    print("## Welcome to AI Card Grading Tool ##\n")
    
    while True:
        # Prompt the user with the question
        identification = input("Would you like to identify your card(s)? (Yes/Y or No/N):\n> ").strip().lower()

        # Evaluate the response
        if identification in ["yes", "y"]:
            print("You have chosen to identify your card(s).\n")
            
            while True:
                identification_type = input("Would you like the identification to be manual or automatic? (Manual/M or Auto/A)\n> ").strip().lower()
                
                if identification_type in ["auto", "a"]:
                    print("You have chosen automatic identification.\n")
                    # Execute the automatic identification script
                    subprocess.run(["python3", "prediction_automatic_image_reading.py"])
                    
                    while True:
                        condition = input("Would you like to know the predicted condition of your card(s)? (Yes/Y or No/N)\n> ").strip().lower()
                        
                        if condition in ["yes", "y"]:
                            print("The automatically identified card(s) will now be graded.\n")
                            prediction_setup("output_log_automatic_combined.json")
                            print("\nThe grading process has finished. Please check the corresponding JSON file.\n")
                            break
                        elif condition in ["no", "n"]:
                            print("You have chosen not to grade your cards.\n")
                            break
                        else:
                            print("Invalid response. Please answer with Yes/Y or No/N.\n")
                    break
                elif identification_type in ["manual", "m"]:
                    print("You have chosen manual identification.\n")
                    # Execute the manual identification script
                    subprocess.run(["python3", "prediction_image_reading.py"])

                    while True:
                        condition = input("Would you like to know the predicted condition of your card(s)? (Yes/Y or No/N)\n> ").strip().lower()
                        
                        if condition in ["yes", "y"]:
                            print("The manually identified card(s) will now be graded.\n")
                            prediction_setup("output_log_combined.json")
                            print("\nThe grading process has finished. Please check the corresponding JSON file.\n")
                            break
                        elif condition in ["no", "n"]:
                            print("You have chosen not to grade your cards.\n")
                            break
                        else:
                            print("Invalid response. Please answer with Yes/Y or No/N.\n")
                    break
                else:
                    print("Invalid response. Please answer with Manual/M or Auto/A.\n")
        
        elif identification in ["no", "n"]:
            print("Proceeding directly to card grading. No card identification will be done.\n")
            only_condition("output_condition_only.json")
            print("Now starting grading process.\n")
            prediction_setup("output_condition_only.json")
            print("The grading process has finished. Please check the corresponding JSON file.\n")
            break
        
        else:
            print("Invalid response. Please answer with Yes/Y or No/N.\n")
        break