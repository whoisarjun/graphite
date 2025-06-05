import torch
import json
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

class MathDataset(Dataset):
    def __init__(self, json_path, tokenizer, transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            self.data = json_data['pairs']

        self.tokenizer = tokenizer
        self.transform = transform if transform else T.Compose([
            T.ToTensor(),
            T.Resize((256, 256)),
            T.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['img']
        latex = item['latex']

        image = Image.open(img_path).convert('L')
        image = self.transform(image)

        token_ids = self.tokenizer.encode(latex)

        return image, torch.tensor(token_ids, dtype=torch.long)
