import torch
import json
import torchvision.transforms as T
from torch.utils.data import Dataset
# Removed tree_utils import
from PIL import Image

class MathDataset(Dataset):
    def __init__(self, json_path, tokenizer, transform=None, parse_func=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            self.data = [
                item for item in json_data['pairs']
                if isinstance(item.get('latex'), str) and item.get('img') is not None and item.get('latex').strip() != ''
            ]

        self.tokenizer = tokenizer
        self.transform = transform if transform else T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.parse_func = parse_func

    def get_type(self, idx):
        return self.data[idx].get("type", "unknown")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image path from data and open image
        img_path = self.data[idx]['img'] if 'img' in self.data[idx] else None
        if img_path is None:
            raise ValueError(f"No image path found for index {idx}")

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        type_str = self.get_type(idx)
        type_id = 0 if type_str == 'symbol' else 1

        latex_str = self.data[idx]['latex']
        if self.parse_func:
            parsed = self.parse_func(latex_str)
        else:
            parsed = latex_str
        tokens_list = self.tokenizer.encode(parsed)
        token_ids = [self.tokenizer.token_to_id.get(tok, self.tokenizer.token_to_id['<UNK>']) for tok in tokens_list]
        tokens = torch.tensor(token_ids, dtype=torch.long)

        return image, tokens, type_id, latex_str