# Using image_processor.py's process() function to create a 'cleaned' version of the CROHME data
import image_processor as ip
from tqdm import tqdm

import os

def _process(source_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)

    for filename in tqdm(os.listdir(source_folder)):
        if filename.endswith(".png"):
            src_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(dest_folder, filename)

            ip.process(src_path, dest_path)

folder_conversions = [
    ('./DATA/IMG/test/CROHME2019_test', './CLEANED_DATA/test/2019_test'),
    ('./DATA/IMG/test/CROHME2023_test', './CLEANED_DATA/test/2023_test'),
    ('./DATA/IMG/train/CROHME2019', './CLEANED_DATA/train/2019_train'),
    ('./DATA/IMG/train/CROHME2013_train', './CLEANED_DATA/train/2023_train'),
    ('./DATA/IMG/val/CROHME2016_test', './CLEANED_DATA/val/2016_val'),
    ('./DATA/IMG/val/CROHME2023_val', './CLEANED_DATA/val/2023_val')
]

for folders in folder_conversions:
    src, dest = folders
    print(f'{src.split("/")[-1]} -> {dest.split("/")[-1]}')
    _process(
        source_folder=src,
        dest_folder=dest
    )
