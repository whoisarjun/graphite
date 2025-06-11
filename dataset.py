import torch
import json
import torchvision.transforms as T
from torch.utils.data import Dataset
from tree_utils import latex_to_sympy_tree, serialize_tree, flatten_tree_serialized
from PIL import Image

class MathDataset(Dataset):
    def __init__(self, json_path, tokenizer, transform=None):
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

        if type_id == 1:  # complete mode
            latex_str = self.data[idx]['latex']
            tree = latex_to_sympy_tree(latex_str)
            if tree is None:
                tree_seq = ['<UNK>']
            else:
                serialized = serialize_tree(tree)
                tree_seq = flatten_tree_serialized(serialized)
            token_ids = [self.tokenizer.token_to_id.get(tok, self.tokenizer.token_to_id['<UNK>']) for tok in tree_seq]
            tokens = torch.tensor(token_ids, dtype=torch.long)
        else:
            latex_str = self.data[idx]['latex']
            tokens_list = self.tokenizer.encode(latex_str)
            token_ids = [self.tokenizer.token_to_id.get(tok, self.tokenizer.token_to_id['<UNK>']) for tok in tokens_list]
            tokens = torch.tensor(token_ids, dtype=torch.long)

        return image, tokens, type_id, latex_str