import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import os
import random
from PIL import Image, ImageFilter
from tqdm import tqdm
import cv2
from multiprocessing import Pool, cpu_count

NUM_WORKERS = 6
ROOT_DIR = './CLEANED_DATA'
VARIANTS_COUNT = 4
IMAGE_SIZE = (512, 512)

def pixelate(image, factor):
    """Pixelate an image by resizing down then up."""
    new_size = (max(1, int(image.width / factor)), max(1, int(image.height / factor)))
    small = image.resize(new_size, Image.NEAREST)
    return small.resize(image.size, Image.NEAREST)

def elastic_distort(image, alpha=36, sigma=6):
    """Apply mild elastic distortion to a PIL grayscale image."""
    image_np = np.array(image).astype(np.float32)

    random_state = np.random.RandomState(None)
    shape = image_np.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distorted = map_coordinates(image_np, indices, order=1, mode='reflect').reshape(shape)
    return Image.fromarray(np.clip(distorted, 0, 255).astype(np.uint8))

# --- Fisheye distortion ---
def fisheye_distort(image, strength=0.00001):
    """Apply a mild fisheye effect to a grayscale PIL image."""
    img_np = np.array(image)
    h, w = img_np.shape
    y, x = np.indices((h, w))
    x_c = w / 2
    y_c = h / 2
    x = x - x_c
    y = y - y_c
    r = np.sqrt(x**2 + y**2)
    r = r / np.max(r)
    factor = 1 + strength * (r**2)
    map_x = (x * factor + x_c).astype(np.float32)
    map_y = (y * factor + y_c).astype(np.float32)
    coords = [map_y.flatten(), map_x.flatten()]
    distorted = map_coordinates(img_np.astype(np.float32), coords, order=1, mode='reflect')
    distorted = distorted.reshape((h, w))
    return Image.fromarray(np.clip(distorted, 0, 255).astype(np.uint8))

def apply_blur(image):
    radius = random.uniform(0.5, 1.5)
    return image.filter(ImageFilter.GaussianBlur(radius))

def add_noise(image):
    noise_level = random.randint(5, 15)
    img_np = np.array(image).astype(np.int16)
    noise = np.random.randint(-noise_level, noise_level + 1, img_np.shape, dtype=np.int16)
    noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def perspective_warp(image):
    img_np = np.array(image)
    h, w = img_np.shape
    strength = random.uniform(0.005, 0.015)
    shift = int(min(h, w) * strength)

    src = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    dst = np.float32([
        [random.randint(-shift, shift), random.randint(-shift, shift)],
        [w - 1 + random.randint(-shift, shift), random.randint(-shift, shift)],
        [random.randint(-shift, shift), h - 1 + random.randint(-shift, shift)],
        [w - 1 + random.randint(-shift, shift), h - 1 + random.randint(-shift, shift)],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_np, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(warped.astype(np.uint8))

def apply_occlusion(image):
    img_np = np.array(image)
    h, w = img_np.shape
    occ_w = random.randint(int(w * 0.05), int(w * 0.1))
    occ_h = random.randint(int(h * 0.05), int(h * 0.1))
    top_left_x = random.randint(0, w - occ_w)
    top_left_y = random.randint(0, h - occ_h)
    img_np[top_left_y:top_left_y+occ_h, top_left_x:top_left_x+occ_w] = 255
    return Image.fromarray(img_np)

def process_image(img_path, rotation_deg, apply_full_aug=True):
    """Open, apply augmentations, rotate, convert to B&W, resize and return processed image.
    If apply_full_aug is False, skip elastic_distort, fisheye_distort, apply_blur, add_noise, perspective_warp.
    """
    img = Image.open(img_path).convert('L')  # convert to grayscale first

    if apply_full_aug:
        # Apply moderate elastic distortion (simulate slight handwriting warps)
        img = elastic_distort(img, alpha=50, sigma=5)

        # Apply mild fisheye distortion
        img = fisheye_distort(img, strength=0.005)

        # Apply new effects
        img = apply_blur(img)
        img = add_noise(img)
        img = perspective_warp(img)

    # Apply subtle rotation (OCR-friendly)
    img = img.rotate(rotation_deg * 0.5, resample=Image.BILINEAR, expand=False, fillcolor=255)

    # Convert to pure B&W (1-bit)
    img = img.point(lambda x: 0 if x < 128 else 255, '1')

    # Resize to standard size
    img = img.resize(IMAGE_SIZE, Image.NEAREST)

    return img

def process_single_image(args):
    img_path, out_path, rotation_deg, apply_full_aug = args
    img_processed = process_image(img_path, rotation_deg, apply_full_aug=apply_full_aug)
    img_processed.save(out_path)

def is_complete_folder(folder_path):
    """Returns True if folder contains images directly (no subfolders)."""
    for entry in os.scandir(folder_path):
        if entry.is_file() and entry.name.lower().endswith('.png'):
            return True
    return False

def get_all_images_in_folder(folder_path):
    """Return list of image file paths directly inside folder."""
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.png')]

def get_all_images_in_symbols_folder(folder_path):
    """Return list of (inner_subfolder, image_path) tuples for symbols folder structure."""
    images = []
    for inner_subfolder in os.listdir(folder_path):
        inner_path = os.path.join(folder_path, inner_subfolder)
        if os.path.isdir(inner_path):
            for img_file in os.listdir(inner_path):
                if img_file.lower().endswith('.png'):
                    images.append( (inner_subfolder, os.path.join(inner_path, img_file)) )
    return images

def make_variant_folder(base_folder, variant_index):
    """Return variant folder path, creating if needed."""
    variant_folder = f"{base_folder}_var{variant_index}"
    if not os.path.exists(variant_folder):
        os.makedirs(variant_folder)
    return variant_folder

def make_inner_folder(parent_folder, inner_folder):
    """Create inner folder path under parent."""
    path = os.path.join(parent_folder, inner_folder)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def generate_variants():
    print(f"Scanning root folder: {ROOT_DIR}")
    for root_entry in os.listdir(ROOT_DIR):
        root_path = os.path.join(ROOT_DIR, root_entry)
        if not os.path.isdir(root_path):
            continue

        print(f"\nProcessing root folder: {root_entry}")

        for var_idx in range(1, VARIANTS_COUNT + 1):
            variant_folder = make_variant_folder(root_path, var_idx)
            rotation_deg = random.uniform(-20, 20)  # subtle rotation for OCR
            print(f" Creating variant: {variant_folder} (rot: {rotation_deg:.2f}°)")

            apply_full_aug = 'symbols' in root_entry.lower()

            if is_complete_folder(root_path):
                images = get_all_images_in_folder(root_path)
                args_list = []
                for img_path in images:
                    img_name = os.path.basename(img_path)
                    out_path = os.path.join(variant_folder, img_name)
                    args_list.append((img_path, out_path, rotation_deg, apply_full_aug))

                with Pool(NUM_WORKERS) as pool:
                    list(tqdm(pool.imap_unordered(process_single_image, args_list), total=len(args_list), desc=f"  Processing images var{var_idx}", ncols=80))
            else:
                inner_subfolders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
                all_images = []
                for inner_subfolder in inner_subfolders:
                    orig_inner_path = os.path.join(root_path, inner_subfolder)
                    variant_inner_path = os.path.join(variant_folder, inner_subfolder)
                    if not os.path.exists(variant_inner_path):
                        os.makedirs(variant_inner_path)
                    images = [f for f in os.listdir(orig_inner_path) if f.lower().endswith('.png')]
                    for img_name in images:
                        img_path = os.path.join(orig_inner_path, img_name)
                        out_path = os.path.join(variant_inner_path, img_name)
                        all_images.append((img_path, out_path))

                args_list = [(img_path, out_path, rotation_deg, apply_full_aug) for img_path, out_path in all_images]
                with Pool(NUM_WORKERS) as pool:
                    list(tqdm(pool.imap_unordered(process_single_image, args_list), total=len(args_list), desc=f"  Processing symbols var{var_idx}", ncols=80))

    print("\nAll variants generated! have fun lol")

if __name__ == '__main__':
    generate_variants()