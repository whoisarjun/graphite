import os

def count_pngs_in_cleaned_data():
    root_folder = './CLEANED_DATA'
    png_count = 0

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for file in filenames:
            if file.lower().endswith('.png'):
                png_count += 1

    print(f"Total .png files in '{root_folder}': {png_count}")
    return png_count

# Call the function
count_pngs_in_cleaned_data()