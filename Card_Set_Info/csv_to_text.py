import csv
import os

# Define the folder containing the CSV files and the folder to save the output files
input_folder = '.'
output_folder = 'Card_Set_Info_READABLE'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to format a row into the desired output format
def format_row(row):
    return f"['{row[0]}', '{row[1]}', '{row[2]}', '{row[3]}', '{row[4]}', '{row[5]}', '{row[6]}', '{row[7]}', '{row[8]}', '{row[9]}', '{row[10]}', '{row[11]}', '{row[12]}', '{row[13]}', '{row[14]}', '{row[15]}', '{row[16]}', '{row[17]}', '{row[18]}', '{row[19]}', '{row[20]}']\n"

# Function to read a CSV file and write the output in the specified format
def create_readable_sets(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='$')
        headers = next(reader)  # Read the header row

        with open(output_file, 'w', encoding='utf-8') as txtfile:
            # Write the header row
            txtfile.write(format_row(headers))

            # Write each row of data
            for row in reader:
                txtfile.write(format_row(row))

# Process each CSV file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename.replace('.csv', '.txt'))
        create_readable_sets(input_file_path, output_file_path)
