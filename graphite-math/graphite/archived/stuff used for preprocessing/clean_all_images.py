# Using image_processor.py's process() function to create a 'cleaned' version of the CROHME data
import image_processor as ip
from tqdm import tqdm

import os
from multiprocessing import Pool, cpu_count

def _process_file(args):
    src_path, dest_path = args
    try:
        ip.process(src_path, dest_path)
        return src_path, True, None
    except Exception as e:
        return src_path, False, str(e)

def _process(source_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    files = [f for f in os.listdir(source_folder) if f.endswith('.png')]

    args = []
    for filename in files:
        src_path = os.path.join(source_folder, filename)
        dest_path = os.path.join(dest_folder, filename)
        args.append((src_path, dest_path))

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(_process_file, args), total=len(args)))

    for src_path, success, error in results:
        if success:
            print(f"✔️ Processed {os.path.basename(src_path)}")
        else:
            print(f"❌ Failed {os.path.basename(src_path)}: {error}")

# Then keep your folder_conversions and loop as is

folder_conversions = [
    ('./2ND_BATCH_CLEANED/CROHME_test_2011', './CLEANED_DATA/2011_test'),
    ('./2ND_BATCH_CLEANED/CROHME_training_2011', './CLEANED_DATA/2011_train'),
    ('./2ND_BATCH_CLEANED/MatricesTest2014', './CLEANED_DATA/2014_test'),
    ('./2ND_BATCH_CLEANED/MatricesTrain2014', './CLEANED_DATA/2014_train'),
    ('./2ND_BATCH_CLEANED/testData_2012', './CLEANED_DATA/2012_test'),
    ('./2ND_BATCH_CLEANED/TestINKML_2013', './CLEANED_DATA/2013_test'),
    ('./2ND_BATCH_CLEANED/trainData_2012_part1', './CLEANED_DATA/2012_train_1'),
    ('./2ND_BATCH_CLEANED/trainData_2012_part2', './CLEANED_DATA/2012_train_2'),
    ('./2ND_BATCH_CLEANED/TrainINKML_2013', './CLEANED_DATA/2013_train'),
]

for folders in folder_conversions:
    src, dest = folders
    print(f'{src.split("/")[-1]} -> {dest.split("/")[-1]}')
    _process(
        source_folder=src,
        dest_folder=dest
    )
