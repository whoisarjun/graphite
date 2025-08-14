import warnings

warnings.filterwarnings(
    "ignore",
    message=r"This DataLoader will create \d+ worker processes in total.*",
    category=UserWarning,
    module="torch.utils.data.dataloader"
)

import torch
import os
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from .create_model import TransformerDecoder, LatexTokenizer  # adjust imports if needed

import re

import torch
import requests

MODEL_PATH = os.path.join(os.path.dirname(__file__), "graphite.pt")
MODEL_URL = "https://huggingface.co/whoisarjun/graphite/resolve/main/graphite.pt"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def clean_latex(raw_latex: str) -> str:
    replacements = {
        'LEFTBRACE': '{',
        'RIGHTBRACE': '}',
        'LEFTBRACKET': '[',
        'RIGHTBRACKET': ']',
        'LEFTPAREN': '(',
        'RIGHTPAREN': ')',
        'SPACE': ' ',
    }
    for token, symbol in replacements.items():
        raw_latex = raw_latex.replace(token, symbol)

    # Remove spaces BEFORE opening delimiters like { [ (
    raw_latex = re.sub(r'\s+([{\[\(])', r'\1', raw_latex)

    # Remove spaces AFTER closing delimiters like } ] )
    raw_latex = re.sub(r'([}\]\)])\s+', r'\1', raw_latex)

    # Remove spaces immediately INSIDE braces/brackets/parens
    raw_latex = re.sub(r'{\s+', '{', raw_latex)
    raw_latex = re.sub(r'\s+}', '}', raw_latex)
    raw_latex = re.sub(r'\[\s+', '[', raw_latex)
    raw_latex = re.sub(r'\s+\]', ']', raw_latex)
    raw_latex = re.sub(r'\(\s+', '(', raw_latex)
    raw_latex = re.sub(r'\s+\)', ')', raw_latex)

    # Remove empty braces/brackets/parens {}, [], ()
    raw_latex = re.sub(r'\{\s*\}', '', raw_latex)
    raw_latex = re.sub(r'\[\s*\]', '', raw_latex)
    raw_latex = re.sub(r'\(\s*\)', '', raw_latex)

    # Remove spaces around common LaTeX operators and symbols
    # e.g. around ^, _, +, -, =, *, / to avoid extra spaces
    raw_latex = re.sub(r'\s*([\^_+\-=*/])\s*', r'\1', raw_latex)

    # Collapse all multiple spaces to one space & strip edges
    raw_latex = re.sub(r'\s+', ' ', raw_latex).strip()

    return raw_latex

class Graphite:
    def __init__(self, model_path=MODEL_PATH, device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        checkpoint = torch.load(model_path, map_location=self.device)
        tokenizer_vocab = checkpoint['tokenizer']

        self.tokenizer = LatexTokenizer()
        self.tokenizer.token_to_id = tokenizer_vocab
        self.tokenizer.id_to_token = {v: k for k, v in tokenizer_vocab.items()}

        vocab_size = len(tokenizer_vocab)

        from .create_model import EncoderViT
        self.encoder = EncoderViT(output_dim=256).to(self.device)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            encoder_dim=256,
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            dropout=0.1,
            max_len=256
        ).to(self.device)

        # Load encoder state dict with strict=False to allow missing keys
        missing_keys, unexpected_keys = self.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        if 'type_embed.weight' in missing_keys:
            # Manually initialize type_embed.weight if missing
            nn.init.normal_(self.encoder.type_embed.weight, mean=0.0, std=0.02)

        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.encoder.eval()
        self.decoder.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def predict(self, image_path, max_len=256):
        # Load + preprocess image
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)  # (1,1,224,224)

        with torch.no_grad():
            encoder_out = self.encoder(img_tensor)  # (1, seq_len, 256)

            # Start decoding with <SOS> token
            sos_id = self.tokenizer.token_to_id['<SOS>']
            eos_id = self.tokenizer.token_to_id['<EOS>']
            decoded_tokens = [sos_id]

            for _ in range(max_len):
                tgt_seq = torch.tensor(decoded_tokens, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, seq_len)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq.size(1)).to(self.device)

                preds = self.decoder(encoder_out, tgt_seq, tgt_mask=tgt_mask)  # (1, seq_len, vocab_size)
                next_token_logits = preds[0, -1, :]
                next_token = next_token_logits.argmax().item()

                decoded_tokens.append(next_token)

                if next_token == eos_id:
                    break

            latex_str = self.tokenizer.decode(decoded_tokens[1:])
            latex_str = clean_latex(latex_str)
            return latex_str