import torch
import json
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

class MathDataset(Dataset):
    def __init__(self, json_path, tokenizer, transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            self.data = [item for item in json_data['pairs'] if isinstance(item.get('latex'), str)]

        self.tokenizer = tokenizer
        self.transform = transform if transform else T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def get_type(self, idx):
        return self.data[idx].get("type", "unknown")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['img']
        latex = item['latex']

        image = Image.open(img_path).convert('L')  # grayscale

        # Convert grayscale to RGB by repeating channels
        image = image.convert('RGB')

        image = self.transform(image)

        token_ids = self.tokenizer.encode(latex)

        return image, torch.tensor(token_ids, dtype=torch.long)
