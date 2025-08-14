import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import label

def _highlight(img: Image, threshold: int):
    new = img.copy()
    pixels, pixels_new = img.load(), new.load()
    w, h = img.size

    for x in range(1, w - 1):
        for y in range(1, h - 1):
            neighbors = [pixels[x, y], pixels[x - 1, y], pixels[x + 1, y], pixels[x, y - 1], pixels[x, y + 1]]
            blacks = 0
            for n in neighbors:
                if n == 0:
                    blacks += 1
            if blacks >= threshold:
                pixels_new[x, y] = 0
            else:
                pixels_new[x, y] = 255

    return new

def _crop(image: Image, noise_thresh=20):
    bw = np.array(image)
    labeled_array, num_features = label(bw == 0)
    if num_features == 0:
        return image

    boxes = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == i)
        size = coords.shape[0]
        if size >= noise_thresh:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            boxes.append((x0, y0, x1, y1))

    if not boxes:
        return image

    # Combine all boxes into one bounding box
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)

    return image.crop((min_x, min_y, max_x, max_y))

def _silence(image: Image, noise_thresh=15):
    # Convert to NumPy array
    bw = np.array(image)

    # Label connected black regions
    labeled_array, num_features = label(bw == 0)

    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_array == i)
        if coords.shape[0] < noise_thresh:
            for y, x in coords:
                bw[y, x] = 255  # set to white

    return Image.fromarray(bw)

def _resize(img: Image, size=(512, 512)):
    img.thumbnail(size, Image.Resampling.LANCZOS)

    delta_w = size[0] - img.width
    delta_h = size[1] - img.height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))

    padded_img = ImageOps.expand(img, padding, fill=255)
    return padded_img

def process(fn: str, output_fn='test.png'):

    # 1. Make everything black and white:
    img = Image.open(fn)
    img = img.convert('1')
    bw = _highlight(_highlight(_highlight(img, 2), 2), 3)

    # 2. Crop the images to the main math thing and remove noise
    cropped = _silence(_highlight(_crop(bw), 2))

    # 3. Resize and save file
    _resize(cropped).save(output_fn)



