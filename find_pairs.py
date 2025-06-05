import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

def _extract_latex(inkml_path):
    tree = ET.parse(inkml_path)
    root = tree.getroot()

    # Namespace may be present, handle if needed
    namespace = {'ns': 'http://www.w3.org/2003/InkML'}

    # Look for annotation with type="truth"
    # This works whether or not namespaces are used
    for annotation in root.findall('.//{http://www.w3.org/2003/InkML}annotation'):
        if annotation.attrib.get('type') == 'truth':
            return annotation.text.strip()

    # fallback: try without namespace if above fails
    for annotation in root.findall('.//annotation'):
        if annotation.attrib.get('type') == 'truth':
            return annotation.text.strip()

    return None  # if not found

def find_pairs(image_folder: str, inkml_folder: str):
    image_files = [f for f in os.listdir(image_folder) if '.png' in f]
    pairs = []
    for img in tqdm(image_files):
        inkml_file = os.path.join(inkml_folder, f'{img[:-4]}.inkml')
        try:
            pairs.append({
                'img': os.path.join(image_folder, img),
                'latex': _extract_latex(inkml_file)
            })
        except FileNotFoundError:
            pass
    return pairs

folder_pairs = [
    ('./CLEANED_DATA/test/2019_test', './DATA/INKML/test/CROHME2019_test'),
    ('./CLEANED_DATA/test/2023_test', './DATA/INKML/test/CROHME2023_test'),
    ('./CLEANED_DATA/train/2019_train', './DATA/INKML/train/CROHME2019'),
    ('./CLEANED_DATA/train/2023_train', './DATA/INKML/train/CROHME2023_train'),
    ('./CLEANED_DATA/val/2016_val', './DATA/INKML/val/CROHME2016_test'),
    ('./CLEANED_DATA/val/2023_val', './DATA/INKML/val/CROHME2023_val')
]

img_latex_pairs = []

for image, latex in folder_pairs:
    img_latex_pairs += find_pairs(image, latex)

with open('pairs.json', 'w') as f:
    json.dump({
        'pairs': img_latex_pairs
    }, f, indent=4)
