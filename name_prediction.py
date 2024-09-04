import difflib

# Function to find the closest match for a given name in a CSV file
def find_closest_name(csv_data, extracted_name):
    closest_name = None
    closest_score = -1

    for row in csv_data:
        csv_name = row[0].split('$')[1]  # Assuming the Name is the second element after splitting by '$'
        similarity_score = difflib.SequenceMatcher(None, extracted_name.lower(), csv_name.lower()).ratio()
        
        if similarity_score > closest_score:
            closest_score = similarity_score
            closest_name = csv_name

    return closest_name, closest_score