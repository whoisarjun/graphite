from concurrent.futures import ProcessPoolExecutor
import os
import shutil

def _split_imgs(input_dir, output_dirs):
    for key in output_dirs:
        os.makedirs(output_dirs[key], exist_ok=True)

    for file in os.listdir(input_dir):
        if file.lower().endswith('.png'):
            src_path = os.path.join(input_dir, file)
            first_char = file[0]
            if first_char in output_dirs:
                dst_path = os.path.join(output_dirs[first_char], file)
                shutil.copy(src_path, dst_path)

letters = 'AbCdefGHijklMNopqRSTvwXyz'
for letter in letters:
    l, u = letter.lower(), letter.upper()
    _split_imgs(f'./3RD_BATCH_SYMBOLS/{letter}', {
        f'{l}': f'./3RD_BATCH_SYMBOLS/{l}_lower/',
        f'{u}': f'./3RD_BATCH_SYMBOLS/{l}_upper/'
    })