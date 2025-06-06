from PIL import Image, ImageOps
import image_processor as ip

source = './10TH_BATCH_SYMBOLS/'
destination = './CLEANED_DATA/symbols8/'

import os
from tqdm import tqdm

folders = sorted(os.listdir(source))
total = len(folders)
for i, symbol_folder in enumerate(folders, start=1):
    symbol_path = os.path.join(source, symbol_folder)
    if not os.path.isdir(symbol_path):
        continue

    print(f"Processing folder ({i}/{total}): {symbol_folder}")
    dest_folder_path = os.path.join(destination, symbol_folder)
    os.makedirs(dest_folder_path, exist_ok=True)

    for filename in tqdm(os.listdir(symbol_path), desc=f"{symbol_folder}", unit="img"):
        ext = filename.lower().split('.')[-1]
        if ext in ['jpg', 'jpeg', 'png']:
            src_path = os.path.join(symbol_path, filename)
            dst_filename = filename.rsplit('.', 1)[0] + '.png'
            dst_path = os.path.join(dest_folder_path, dst_filename)

            img = Image.open(src_path).convert('L')
            # If input is jpg/jpeg, convert to png by saving with .png extension (done by save)
            img = img.resize((512, 512), Image.LANCZOS)
            img = img.point(lambda x: 255 if x > 128 else 0, '1')
            img.save(dst_path)
