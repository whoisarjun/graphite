import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Sampler
from torch import nn
from .dataset import MathDataset
from .latex_tokenizer import LatexTokenizer
from sympy.parsing.latex import parse_latex
from sympy.core.sympify import SympifyError
from tqdm import tqdm
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import timm
from .latex_list import lst
import os

BATCH_SIZE = 16
CORES = 16
TOTAL_EPOCHS = 25
import os

JSON_PATH = os.path.join(os.path.dirname(__file__), 'empty.json')
PT_SAVE = 'graphite_test_val.pt'

def collate_fn(batch):
    # Unpack batch: (img, tokens, type_id)
    imgs, token_lists, type_ids = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # stack images (all same size)
    lengths = [len(t) for t in token_lists]
    max_length = max(lengths)
    padded_tokens_ = torch.zeros(len(token_lists), max_length, dtype=torch.long)
    for j, tokens in enumerate(token_lists):
        end = lengths[j]
        padded_tokens_[j, :end] = tokens if isinstance(tokens, torch.Tensor) else torch.tensor(tokens, dtype=torch.long)
    type_ids_tensor = torch.tensor(type_ids, dtype=torch.long)
    return imgs, padded_tokens_, lengths, type_ids_tensor

class EncoderViT(nn.Module):
    def __init__(self, output_dim=256, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, img_size=224, num_classes=0)
        self.patch_embed = self.vit.patch_embed  # Patch embedding module
        self.pos_embed = self.vit.pos_embed      # Positional embeddings
        self.cls_token = self.vit.cls_token      # CLS token
        self.blocks = self.vit.blocks            # Transformer encoder blocks
        self.norm = self.vit.norm                # Layer norm
        self.output_dim = output_dim
        self.proj = nn.Linear(self.vit.embed_dim, output_dim)
        # Add type embedding for 'symbol' and 'complete'
        self.type_embed = nn.Embedding(2, self.vit.embed_dim)
        self.resize_pos_embed(new_img_size=224, patch_size=16)

    def resize_pos_embed(self, new_img_size=224, patch_size=16):
        num_patches = (new_img_size // patch_size) ** 2
        cls_token = self.cls_token
        pos_embed = self.pos_embed[:, 1:, :]  # remove CLS token
        B, N, D = pos_embed.shape

        old_size = int(N ** 0.5)
        new_size = int(num_patches ** 0.5)

        pos_embed = pos_embed.reshape(1, old_size, old_size, D).permute(0, 3, 1, 2)
        pos_embed = torch.nn.functional.interpolate(pos_embed, size=(new_size, new_size), mode='bilinear')
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_size * new_size, D)

        self.pos_embed = nn.Parameter(torch.cat([cls_token, pos_embed], dim=1))

    # Accept type_ids tensor, add corresponding embedding to patch embeddings before positional embedding
    def forward(self, x, type_ids=None):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        if type_ids is not None:
            # Add type embedding for each sample in the batch (0: symbol, 1: complete)
            type_embedding = self.type_embed(type_ids).unsqueeze(1)  # (B, 1, embed_dim)
            x = x + type_embedding  # broadcast add
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

tokenizer = LatexTokenizer()
tokenizer.build_vocab(lst)

class MathDatasetWithType(MathDataset):
    """
    Extension of MathDataset to return (img, tokens, type_id) where type_id is 0 for 'symbol', 1 for 'complete'.
    """
    def __getitem__(self, idx):
        img, tokens = super().__getitem__(idx)
        type_str = self.get_type(idx)
        type_id = 0 if type_str == 'symbol' else 1
        return img, tokens, type_id

# Dataset + DataLoader
dataset = MathDatasetWithType(JSON_PATH, tokenizer)

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        # Group indices by 'type'
        self.symbol_indices = []
        self.complete_indices = []
        for idx in range(len(dataset)):
            item_type = dataset.get_type(idx)  # Assuming MathDataset has get_type method returning 'symbol' or 'complete'
            if item_type == 'symbol':
                self.symbol_indices.append(idx)
            elif item_type == 'complete':
                self.complete_indices.append(idx)
            else:
                # If other types exist, ignore or handle accordingly
                pass
        self.symbol_pos = 0
        self.complete_pos = 0
        self.num_batches = len(dataset) // batch_size

        # Shuffle indices initially
        self._shuffle_indices()

    def _shuffle_indices(self):
        import random
        random.shuffle(self.symbol_indices)
        random.shuffle(self.complete_indices)
        self.symbol_pos = 0
        self.complete_pos = 0

    def __iter__(self):
        self._shuffle_indices()
        batches = []
        half_batch = self.batch_size // 2
        for _ in range(self.num_batches):
            batch = []
            # Sample half batch from symbol_indices
            for _ in range(half_batch):
                if self.symbol_pos >= len(self.symbol_indices):
                    self.symbol_pos = 0
                    import random
                    random.shuffle(self.symbol_indices)
                batch.append(self.symbol_indices[self.symbol_pos])
                self.symbol_pos += 1
            # Sample half batch from complete_indices
            for _ in range(half_batch):
                if self.complete_pos >= len(self.complete_indices):
                    self.complete_pos = 0
                    import random
                    random.shuffle(self.complete_indices)
                batch.append(self.complete_indices[self.complete_pos])
                self.complete_pos += 1
            batches.append(batch)
        # If batch_size is odd, fill last slot from either group
        if self.batch_size % 2 != 0:
            # Add one more sample from symbol_indices if available else complete_indices
            if self.symbol_pos < len(self.symbol_indices):
                batches[-1].append(self.symbol_indices[self.symbol_pos])
                self.symbol_pos += 1
            elif self.complete_pos < len(self.complete_indices):
                batches[-1].append(self.complete_indices[self.complete_pos])
                self.complete_pos += 1
        for batch in batches:
            yield batch

    def __len__(self):
        return self.num_batches

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=CORES)

# Build encoder and decoder
encoder = EncoderViT(output_dim=256)
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
        return True, expr
    except (SympifyError, Exception):
        return False, None

# Compare two LaTeX expressions using SymPy's expression trees for structural equivalence
def expressions_match(latex_pred, latex_gt):
    try:
        expr_pred = parse_latex(latex_pred)
        expr_gt = parse_latex(latex_gt)
        return expr_pred.equals(expr_gt)
    except Exception:
        return False

# Consider using math expression validation during training as an auxiliary loss or evaluation metric

class SubsetWithGetType(torch.utils.data.Dataset):
    def __init__(self, original_dataset, indices):
        self.dataset = original_dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    def get_type(self, idx):
        return self.dataset.get_type(self.indices[idx])


def create():
    print('Getting started...')

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create BalancedBatchSampler for train_dataset
    # Need to map indices in train_dataset back to original dataset indices
    train_indices = train_dataset.indices if hasattr(train_dataset, 'indices') else list(range(train_size))
    train_subset = SubsetWithGetType(dataset, train_indices)

    train_sampler = BalancedBatchSampler(train_subset, batch_size=BATCH_SIZE)

    train_loader = DataLoader(train_subset, batch_sampler=train_sampler, collate_fn=collate_fn,
                              num_workers=CORES, pin_memory=True, persistent_workers=True, prefetch_factor=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                            num_workers=CORES, pin_memory=True, persistent_workers=True, prefetch_factor=8)

    # Enable cuDNN autotuner and AMP
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs = TOTAL_EPOCHS
    patience = 7

    for epoch in range(n_epochs):
        print(f'Running epoch {epoch+1}/{n_epochs}')
        encoder.train()
        decoder.train()
        running_loss = 0.0
        # Training loop: extract type_ids from batch and pass to encoder
        for images, padded_tokens, lengths, type_ids in tqdm(train_loader):
            images = images.to(device, non_blocking=True)
            padded_tokens = padded_tokens.to(device, non_blocking=True)
            lengths = torch.tensor(lengths).to(device, non_blocking=True)
            type_ids = type_ids.to(device, non_blocking=True)

            optimizer.zero_grad()

            # AMP autocast for forward/backward
            with torch.cuda.amp.autocast():
                # Pass type_ids to encoder for type embedding
                encoder_out = encoder(images, type_ids=type_ids)
                seq_len = padded_tokens.size(1) - 1
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                predictions = decoder(encoder_out, padded_tokens[:, :-1], tgt_mask=tgt_mask)
                targets = padded_tokens[:, 1:].reshape(-1)
                preds = predictions.reshape(-1, predictions.size(-1))
                decoder_loss = criterion(preds, targets)

            scaler.scale(decoder_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += decoder_loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}')

        # Validation
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        predictions_accum = []
        with torch.no_grad():
            for images, padded_tokens, lengths, type_ids in val_loader:
                images = images.to(device)
                padded_tokens = padded_tokens.to(device)
                lengths = torch.tensor(lengths).to(device)
                type_ids = type_ids.to(device)

                # Pass type_ids to encoder for type embedding
                encoder_out = encoder(images, type_ids=type_ids)

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
            print(f'Raw: {pred} | Valid: {valid} | Parsed Tree: {parsed}')

        # Structure-aware accuracy using SymPy parsing and comparison
        structure_match_count = 0
        total = len(decoded_preds)
        for pred_str, gt_tokens in zip(decoded_preds, padded_tokens):
            gt_str = tokenizer.decode(gt_tokens.tolist())
            if expressions_match(pred_str, gt_str):
                structure_match_count += 1
        structure_accuracy = structure_match_count / total
        print(f'Structure-aware Accuracy: {structure_accuracy:.4f}')

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
            }, PT_SAVE)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break

if __name__ == '__main__':
    create()