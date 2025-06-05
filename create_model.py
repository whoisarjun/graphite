import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch import nn
from dataset import MathDataset
from latex_tokenizer import LatexTokenizer
from sympy.parsing.latex import parse_latex
from sympy.core.sympify import SympifyError
from tqdm import tqdm
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import Union
import timm

BATCH_SIZE = 128
CORES = 16
JSON_PATH = 'pairs.json'

def collate_fn(batch):
    imgs, token_lists = zip(*batch)

    imgs = torch.stack(imgs, dim=0)  # stack images (all same size)

    lengths = [len(t) for t in token_lists]
    max_length = max(lengths)

    padded_tokens_ = torch.zeros(len(token_lists), max_length, dtype=torch.long)
    for j, tokens in enumerate(token_lists):
        end = lengths[j]
        padded_tokens_[j, :end] = tokens if isinstance(tokens, torch.Tensor) else torch.tensor(tokens, dtype=torch.long)

    return imgs, padded_tokens_, lengths


class EncoderCNN(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        self.backbone.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(1280, output_dim)

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.pool(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        x = self.fc(x)
        return x  # (batch_size, seq_len, output_dim)

class EncoderViT(nn.Module):
    def __init__(self, output_dim=256, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, img_size=224, num_classes=0, global_pool='')
        self.patch_embed = self.vit.patch_embed  # Patch embedding module
        self.pos_embed = self.vit.pos_embed      # Positional embeddings
        self.cls_token = self.vit.cls_token      # CLS token
        self.blocks = self.vit.blocks            # Transformer encoder blocks
        self.norm = self.vit.norm                # Layer norm
        self.output_dim = output_dim
        self.proj = nn.Linear(self.vit.embed_dim, output_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + num_patches, embed_dim)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.proj(x)  # (B, seq_len, output_dim)
        return x  # (batch_size, seq_len, output_dim)

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # encoder features to attention space
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # decoder hidden to attention space
        self.full_att = nn.Linear(attention_dim, 1)               # combined to scalar score
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)             # [B, N, attention_dim]
        att2 = self.decoder_att(decoder_hidden)          # [B, attention_dim]
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # [B, N]
        alpha = self.softmax(att)                         # attention weights
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # weighted sum context vector
        return context, alpha

class TAMERDecoder(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=256, dropout=0.5, max_len=256):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.attention_dim = attention_dim

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)

        embeddings = self.embedding(captions)
        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (caption_lengths - 1).tolist()
        max_len = max(decode_lengths)

        predictions = torch.zeros(batch_size, max_len, self.vocab_size).to(encoder_out.device)

        for t in range(max_len):
            batch_size_t = sum([l > t for l in decode_lengths])
            context, _ = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            context = gate * context

            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], context], dim=1)
            h, c = self.decode_step(lstm_input, (h[:batch_size_t], c[:batch_size_t]))

            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds

        return predictions

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim=256, embed_dim=256, num_heads=8, num_layers=6, dropout=0.1, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward=2048, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_outputs, tgt_seq, tgt_mask=None):
        seq_len = tgt_seq.size(1)
        tgt_emb = self.embed(tgt_seq) + self.pos_embed[:, :seq_len, :]
        tgt_emb = self.dropout(tgt_emb)
        output = self.transformer_decoder(tgt_emb, encoder_outputs, tgt_mask=tgt_mask)
        logits = self.fc(output)
        return logits

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
latex_list = [pair['latex'] for pair in raw_data['pairs']]

tokenizer = LatexTokenizer()
tokenizer.build_vocab(latex_list)

# Dataset + DataLoader
dataset = MathDataset(JSON_PATH, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=CORES)

# Build encoder and decoder
encoder = EncoderCNN(output_dim=256)
decoder = TransformerDecoder(
    vocab_size=len(tokenizer.token_to_id),
    encoder_dim=256,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    dropout=0.1,
    max_len=256
)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
encoder = encoder.to(device)
decoder = decoder.to(device)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_prob = nn.functional.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

criterion = LabelSmoothingLoss(smoothing=0.1)

optimizer = optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()),
    lr=1e-4
)
# NOTE: CrossEntropyLoss may not capture semantic errors (e.g., missing terms or wrong structure)
# Consider evaluating with sequence-level metrics like BLEU or edit distance in addition to CrossEntropy

def decode_predictions(preds, tokenizer):
    pred_tokens = preds.argmax(dim=-1)
    decoded = [tokenizer.decode(seq.tolist()) for seq in pred_tokens]
    # TIP: You can add a sanity check here to compare decoded output with valid LaTeX via sympy
    return decoded

def validate_math_expression(latex_str):
    try:
        expr = parse_latex(latex_str)
        return True, str(expr)
    except (SympifyError, Exception):
        return False, None

# Consider using math expression validation during training as an auxiliary loss or evaluation metric

def create():
    print('Getting started...')

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, prefetch_factor=4, num_workers=CORES, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, prefetch_factor=4, num_workers=CORES, pin_memory=False)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs = 50
    patience = 7

    for epoch in range(n_epochs):
        print(f'Running epoch {epoch+1}/{n_epochs}')
        encoder.train()
        decoder.train()
        running_loss = 0.0
        for images, padded_tokens, lengths in tqdm(train_loader):
            images = images.to(device, non_blocking=True)
            padded_tokens = padded_tokens.to(device, non_blocking=True)
            lengths = torch.tensor(lengths).to(device, non_blocking=True)

            optimizer.zero_grad()

            # Encoder forward
            encoder_out = encoder(images)
            # encoder_out is already (batch_size, seq_len, encoder_dim)

            # Create tgt_mask for Transformer decoder
            seq_len = padded_tokens.size(1) - 1
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

            predictions = decoder(encoder_out, padded_tokens[:, :-1], tgt_mask=tgt_mask)

            # Decoder loss
            targets = padded_tokens[:, 1:].reshape(-1)
            preds = predictions.reshape(-1, predictions.size(-1))
            decoder_loss = criterion(preds, targets)

            decoder_loss.backward()
            optimizer.step()

            running_loss += decoder_loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}')

        # Validation
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        predictions_accum = []
        with torch.no_grad():
            for images, padded_tokens, lengths in val_loader:
                images = images.to(device)
                padded_tokens = padded_tokens.to(device)
                lengths = torch.tensor(lengths).to(device)

                encoder_out = encoder(images)

                seq_len = padded_tokens.size(1) - 1
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

                predictions = decoder(encoder_out, padded_tokens[:, :-1], tgt_mask=tgt_mask)

                predictions_accum.append(predictions.cpu())

                targets = padded_tokens[:, 1:].reshape(-1)
                preds = predictions.reshape(-1, predictions.size(-1))

                loss = criterion(preds, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}/{n_epochs}, Validation Loss: {avg_val_loss:.4f}')

        # --- Structure-aware semantic validation logging ---
        # Only on first epoch or every epoch, as desired
        # Pad predictions so all have the same sequence length
        from torch.nn.functional import pad
        max_seq_len = max(p.shape[1] for p in predictions_accum)
        for i in range(len(predictions_accum)):
            seq_len = predictions_accum[i].shape[1]
            if seq_len < max_seq_len:
                pad_len = max_seq_len - seq_len
                predictions_accum[i] = pad(predictions_accum[i], (0, 0, 0, pad_len))
        all_preds = torch.cat(predictions_accum, dim=0)
        decoded_preds = decode_predictions(all_preds, tokenizer)
        print('Structure-aware validation (first 5 predictions):')
        for pred in decoded_preds[:5]:  # just first few examples
            valid, parsed = validate_math_expression(pred)
            print(f'Raw: {pred} | Valid: {valid} | Parsed: {parsed}')

        # Optional: Add sequence-level evaluation metric here (e.g., BLEU or edit distance) for future improvement
        # e.g., compute BLEU between decoded_preds and ground truth sequences

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tokenizer': tokenizer.token_to_id,
            }, 'graphite.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

if __name__ == '__main__':
    create()
