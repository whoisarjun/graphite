import json
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPModel
from torch import nn
from dataset import MathDataset
from latex_tokenizer import LatexTokenizer
from tree_utils import to_tree, to_latex, clean_latex

BATCH_SIZE = 4
CORES = 1
TOTAL_EPOCHS = 30
PT_SAVE = 'graphite.pt'

TRAIN_JSON = './datasets/train.json'
TEST_JSON = './datasets/test.json'
VAL_JSON = './datasets/val.json'


def collate_fn(batch):
    # Unpack batch: (img, tokens, type_id)
    imgs, token_lists, type_ids, latex_strs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)  # stack images (all same size)
    lengths = [len(t) for t in token_lists]
    max_length = max(lengths)
    padded_tokens_ = torch.zeros(len(token_lists), max_length, dtype=torch.long)
    for j, tokens in enumerate(token_lists):
        end = lengths[j]
        padded_tokens_[j, :end] = tokens if isinstance(tokens, torch.Tensor) else torch.tensor(tokens, dtype=torch.long)
    type_ids_tensor = torch.tensor(type_ids, dtype=torch.long)
    return imgs, padded_tokens_, lengths, type_ids_tensor, latex_strs


class EncoderViT(nn.Module):
    def __init__(self, output_dim=256, model_name='openai/clip-vit-large-patch14'):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)

        for param in self.model.vision_model.parameters():
            param.requires_grad = False

        self.output_dim = output_dim
        self.proj = nn.Linear(self.model.vision_model.config.hidden_size, output_dim)

        self.type_embed = nn.Embedding(2, self.model.vision_model.config.hidden_size)

    def forward(self, images, type_ids=None):
        if type_ids is not None:
            type_embedding = self.type_embed(type_ids).unsqueeze(1)
        else:
            type_embedding = 0

        vision_outputs = self.model.vision_model(pixel_values=images, output_hidden_states=True)
        last_hidden = vision_outputs.hidden_states[-1]

        if isinstance(type_embedding, torch.Tensor):
            last_hidden = last_hidden + type_embedding

        x = self.proj(last_hidden)
        return x


class TamerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, dropout=0.1, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, encoder_out, tgt_seq, tgt_mask=None):
        B, T = tgt_seq.shape
        x = self.embed(tgt_seq) + self.pos_embed[:, :T, :]
        x = self.dropout(x)
        x, _ = self.gru(x)
        logits = self.fc(x)
        return logits


with open(TRAIN_JSON, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

with open(TEST_JSON, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

with open(VAL_JSON, 'r', encoding='utf-8') as f:
    val_data = json.load(f)

train_latex = [pair['latex'] for pair in train_data['pairs'] if pair.get('latex') is not None]
val_latex = [pair['latex'] for pair in val_data['pairs'] if pair.get('latex') is not None]
test_latex = [pair['latex'] for pair in test_data['pairs'] if pair.get('latex') is not None]

tokenizer = LatexTokenizer()


def safe_to_tree(latex, show_success=False):
    try:
        result = to_tree(latex)
        if show_success:
            print(f"[SUCCESS] Successfully parsed LaTeX: {latex[:50]}...")
        return result
    except Exception as e:
        print(f"[FAILED] Failed to parse LaTeX: {latex[:50]}... | Error: {str(e)[:100]}")
        return None


def safe_to_latex(sequence):
    try:
        return to_latex(sequence)
    except Exception as e:
        print(f"[WARNING] Failed to convert sequence to LaTeX, using fallback")
        return ""


def safe_clean_latex(latex):
    try:
        return clean_latex(latex)
    except Exception as e:
        print(f"[WARNING] Failed to clean LaTeX, using original: {latex[:50]}...")
        return latex


# Filter out None values (failed parses)
train_sequences = [seq for seq in [safe_to_tree(latex) for latex in train_latex] if seq is not None]
val_sequences = [seq for seq in [safe_to_tree(latex) for latex in val_latex] if seq is not None]
test_sequences = [seq for seq in [safe_to_tree(latex) for latex in test_latex] if seq is not None]

print(
    f"Successfully parsed: Train={len(train_sequences)}/{len(train_latex)}, Val={len(val_sequences)}/{len(val_latex)}, Test={len(test_sequences)}/{len(test_latex)}")

tokenizer.build_vocab(train_sequences + val_sequences + test_sequences)

# Create datasets with safe parsing
train_dataset = MathDataset(TRAIN_JSON, tokenizer, parse_func=lambda x: safe_to_tree(x))
val_dataset = MathDataset(VAL_JSON, tokenizer, parse_func=lambda x: safe_to_tree(x))
test_dataset = MathDataset(TEST_JSON, tokenizer, parse_func=lambda x: safe_to_tree(x))

# Build encoder and decoder
encoder = EncoderViT(output_dim=256)
decoder = TamerDecoder(
    vocab_size=len(tokenizer.token_to_id),
    embed_dim=256,
    dropout=0.1,
    max_len=256
)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
encoder = encoder.to(device)
decoder = decoder.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

optimizer = optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()),
    lr=1e-6
)


def decode_predictions(preds, tokenizer):
    pred_tokens = preds.argmax(dim=-1)
    decoded = []
    for seq in pred_tokens:
        try:
            latex_str = safe_to_latex(tokenizer.decode(seq.tolist()))
            cleaned = safe_clean_latex(latex_str)
            decoded.append(cleaned)
        except Exception as e:
            print(f"[WARNING] Failed to decode prediction, using empty string")
            decoded.append("")
    return decoded


def create():
    print('Getting started...')

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
        num_workers=CORES, pin_memory=True, persistent_workers=True, prefetch_factor=8
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
        num_workers=CORES, pin_memory=True, persistent_workers=True, prefetch_factor=8
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
        num_workers=CORES, pin_memory=True, persistent_workers=True, prefetch_factor=8
    )

    # Enable cuDNN autotuner and AMP
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs = TOTAL_EPOCHS
    patience = 7

    for epoch in range(n_epochs):
        print(f'Running epoch {epoch + 1}/{n_epochs}')
        encoder.train()
        decoder.train()
        running_loss = 0.0

        # Training loop: extract type_ids from batch and pass to encoder
        for images, padded_tokens, lengths, type_ids, latex_strs in tqdm(train_loader):
            # Skip batch if any sample failed to parse
            if any(length == 0 for length in lengths):
                print("[WARNING] Skipping batch with failed parsing")
                continue

            for i, latex in enumerate(latex_strs):
                print(f"Sample {i} LaTeX: {safe_clean_latex(latex)} | Token length: {lengths[i]}")

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
                if seq_len <= 0:
                    print("[WARNING] Zero or negative sequence length encountered. Applying zero-length padding fix.")
                    padded_tokens = torch.cat(
                        [padded_tokens,
                         torch.full((padded_tokens.size(0), 1), tokenizer.pad_token_id, device=padded_tokens.device)],
                        dim=1
                    )
                    seq_len = padded_tokens.size(1) - 1
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                predictions = decoder(encoder_out, padded_tokens[:, :-1], tgt_mask=tgt_mask)

                # --- SANITY CHECKS ---
                # Check for NaN or Inf in predictions
                if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                    print("[WARNING] NaN or Inf detected in predictions.")
                    print(f"predictions shape: {predictions.shape}")
                    print(f"images shape: {images.shape}")
                    print(f"padded_tokens shape: {padded_tokens.shape}")
                    print(f"seq_len: {seq_len}")
                    raise RuntimeError(f"Sanity check failed: NaN or Inf in predictions (shape: {predictions.shape})")
                # Check for NaN or Inf in targets
                targets = padded_tokens[:, 1:].reshape(-1)
                if torch.isnan(targets.float()).any() or torch.isinf(targets.float()).any():
                    print("[WARNING] NaN or Inf detected in targets.")
                    print(f"targets shape: {targets.shape}")
                    raise RuntimeError(f"Sanity check failed: NaN or Inf in targets (shape: {targets.shape})")
                # Print shapes
                print(
                    f"[Sanity] images shape: {images.shape}, padded_tokens shape: {padded_tokens.shape}, predictions shape: {predictions.shape}, targets shape: {targets.shape}")
                # Check seq_len
                if seq_len <= 0:
                    print("[WARNING] seq_len is not greater than zero after fix.")
                    raise RuntimeError(f"Sanity check failed: seq_len <= 0 (seq_len: {seq_len})")
                print(f"[Sanity] seq_len: {seq_len}")
                # Check padded_tokens slicing shapes
                if padded_tokens[:, :-1].shape[1] != seq_len or padded_tokens[:, 1:].shape[1] != seq_len:
                    print("[WARNING] padded_tokens slicing shapes are invalid.")
                    print(f"padded_tokens[:, :-1].shape: {padded_tokens[:, :-1].shape}")
                    print(f"padded_tokens[:, 1:].shape: {padded_tokens[:, 1:].shape}")
                    raise RuntimeError(
                        f"Sanity check failed: Invalid shapes for padded_tokens slices (got {padded_tokens[:, :-1].shape} and {padded_tokens[:, 1:].shape}, expected seq_len={seq_len})")

                # Additional check: print unique target tokens before loss calculation
                unique_targets = torch.unique(targets)
                print(f"[Sanity] Unique target tokens before loss: {unique_targets.tolist()}")
                # Additional check: check for extreme values in predictions
                if (predictions > 1e4).any() or (predictions < -1e4).any():
                    print("[ERROR] Extreme value detected in predictions (>1e4 or <-1e4).")
                    print(f"Max prediction: {predictions.max().item()}, Min prediction: {predictions.min().item()}")
                    raise RuntimeError("Sanity check failed: Extreme value in predictions (>1e4 or <-1e4).")

                preds = predictions.reshape(-1, predictions.size(-1))
                decoder_loss = criterion(preds, targets)

            # NaN loss check
            if torch.isnan(decoder_loss):
                print("[WARNING] NaN in loss. Exiting with error.")
                raise RuntimeError("NaN in loss.")

            scaler.scale(decoder_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += decoder_loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}')

        # Validation
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        predictions_accum = []
        with torch.no_grad():
            for images, padded_tokens, lengths, type_ids, latex_strs in tqdm(val_loader):
                # Skip batch if any sample failed to parse
                if any(length == 0 for length in lengths):
                    print("[WARNING] Skipping validation batch with failed parsing")
                    continue

                for i, latex in enumerate(latex_strs):
                    print(f"[VAL] Sample {i} LaTeX: {safe_clean_latex(latex)} | Token length: {lengths[i]}")
                images = images.to(device)
                padded_tokens = padded_tokens.to(device)
                lengths = torch.tensor(lengths).to(device)
                type_ids = type_ids.to(device)

                # Pass type_ids to encoder for type embedding
                encoder_out = encoder(images, type_ids=type_ids)

                seq_len = padded_tokens.size(1) - 1
                if seq_len <= 0:
                    print(
                        "[WARNING] Zero or negative sequence length encountered in validation. Applying zero-length padding fix.")
                    padded_tokens = torch.cat(
                        [padded_tokens,
                         torch.full((padded_tokens.size(0), 1), tokenizer.pad_token_id, device=padded_tokens.device)],
                        dim=1
                    )
                    seq_len = padded_tokens.size(1) - 1
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                predictions = decoder(encoder_out, padded_tokens[:, :-1], tgt_mask=tgt_mask)

                # --- SANITY CHECKS (Validation) ---
                if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                    print("[WARNING] NaN or Inf detected in predictions (validation).")
                    print(f"predictions shape: {predictions.shape}")
                    print(f"images shape: {images.shape}")
                    print(f"padded_tokens shape: {padded_tokens.shape}")
                    print(f"seq_len: {seq_len}")
                    raise RuntimeError(
                        f"Sanity check failed: NaN or Inf in predictions (validation, shape: {predictions.shape})")
                targets = padded_tokens[:, 1:].reshape(-1)
                if torch.isnan(targets.float()).any() or torch.isinf(targets.float()).any():
                    print("[WARNING] NaN or Inf detected in targets (validation).")
                    print(f"targets shape: {targets.shape}")
                    raise RuntimeError(
                        f"Sanity check failed: NaN or Inf in targets (validation, shape: {targets.shape})")
                print(
                    f"[Sanity][Val] images shape: {images.shape}, padded_tokens shape: {padded_tokens.shape}, predictions shape: {predictions.shape}, targets shape: {targets.shape}")
                if seq_len <= 0:
                    print("[WARNING] seq_len is not greater than zero after fix (validation).")
                    raise RuntimeError(f"Sanity check failed: seq_len <= 0 (validation, seq_len: {seq_len})")
                print(f"[Sanity][Val] seq_len: {seq_len}")
                if padded_tokens[:, :-1].shape[1] != seq_len or padded_tokens[:, 1:].shape[1] != seq_len:
                    print("[WARNING] padded_tokens slicing shapes are invalid (validation).")
                    print(f"padded_tokens[:, :-1].shape: {padded_tokens[:, :-1].shape}")
                    print(f"padded_tokens[:, 1:].shape: {padded_tokens[:, 1:].shape}")
                    raise RuntimeError(
                        f"Sanity check failed: Invalid shapes for padded_tokens slices (validation, got {padded_tokens[:, :-1].shape} and {padded_tokens[:, 1:].shape}, expected seq_len={seq_len})")

                # Additional check: print unique target tokens before loss calculation
                unique_targets = torch.unique(targets)
                print(f"[Sanity][Val] Unique target tokens before loss: {unique_targets.tolist()}")
                # Additional check: check for extreme values in predictions
                if (predictions > 1e4).any() or (predictions < -1e4).any():
                    print("[ERROR][Val] Extreme value detected in predictions (>1e4 or <-1e4).")
                    print(f"Max prediction: {predictions.max().item()}, Min prediction: {predictions.min().item()}")
                    raise RuntimeError(
                        "Sanity check failed: Extreme value in predictions (>1e4 or <-1e4) in validation.")

                predictions_accum.append(predictions.cpu())

                preds = predictions.reshape(-1, predictions.size(-1))

                loss = criterion(preds, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}/{n_epochs}, Validation Loss: {avg_val_loss:.4f}')

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
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

    # After early stopping or last epoch
    print("Training done! Running test set evaluation...")
    test(encoder, decoder, test_loader)


# Add test() function below create()
def test(encoder, decoder, dataloader):
    encoder.eval()
    decoder.eval()
    predictions_accum = []
    with torch.no_grad():
        for images, padded_tokens, lengths, type_ids, latex_strs in dataloader:
            # Skip batch if any sample failed to parse
            if any(length == 0 for length in lengths):
                print("[WARNING] Skipping test batch with failed parsing")
                continue

            for i, latex in enumerate(latex_strs):
                print(f"[Test] Sample {i} LaTeX: {safe_clean_latex(latex)} | Token length: {lengths[i]}")
            images = images.to(device)
            padded_tokens = padded_tokens.to(device)
            lengths = torch.tensor(lengths).to(device)
            type_ids = type_ids.to(device)

            encoder_out = encoder(images, type_ids=type_ids)

            seq_len = padded_tokens.size(1) - 1
            if seq_len <= 0:
                print(
                    "[WARNING] Zero or negative sequence length encountered in test. Applying zero-length padding fix.")
                padded_tokens = torch.cat(
                    [padded_tokens,
                     torch.full((padded_tokens.size(0), 1), tokenizer.pad_token_id, device=padded_tokens.device)],
                    dim=1
                )
                seq_len = padded_tokens.size(1) - 1
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

            predictions = decoder(encoder_out, padded_tokens[:, :-1], tgt_mask=tgt_mask)

            # --- SANITY CHECKS (Test) ---
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                print("[WARNING] NaN or Inf detected in predictions (test).")
                print(f"predictions shape: {predictions.shape}")
                print(f"images shape: {images.shape}")
                print(f"padded_tokens shape: {padded_tokens.shape}")
                print(f"seq_len: {seq_len}")
                raise RuntimeError(f"Sanity check failed: NaN or Inf in predictions (test, shape: {predictions.shape})")
            targets = padded_tokens[:, 1:].reshape(-1)
            if torch.isnan(targets.float()).any() or torch.isinf(targets.float()).any():
                print("[WARNING] NaN or Inf detected in targets (test).")
                print(f"targets shape: {targets.shape}")
                raise RuntimeError(f"Sanity check failed: NaN or Inf in targets (test, shape: {targets.shape})")
            print(
                f"[Sanity][Test] images shape: {images.shape}, padded_tokens shape: {padded_tokens.shape}, predictions shape: {predictions.shape}, targets shape: {targets.shape}")
            if seq_len <= 0:
                print("[WARNING] seq_len is not greater than zero after fix (test).")
                raise RuntimeError(f"Sanity check failed: seq_len <= 0 (test, seq_len: {seq_len})")
            print(f"[Sanity][Test] seq_len: {seq_len}")
            if padded_tokens[:, :-1].shape[1] != seq_len or padded_tokens[:, 1:].shape[1] != seq_len:
                print("[WARNING] padded_tokens slicing shapes are invalid (test).")
                print(f"padded_tokens[:, :-1].shape: {padded_tokens[:, :-1].shape}")
                print(f"padded_tokens[:, 1:].shape: {padded_tokens[:, 1:].shape}")
                raise RuntimeError(
                    f"Sanity check failed: Invalid shapes for padded_tokens slices (test, got {padded_tokens[:, :-1].shape} and {padded_tokens[:, 1:].shape}, expected seq_len={seq_len})")

            # Additional check: print unique target tokens before further processing
            unique_targets = torch.unique(targets)
            print(f"[Sanity][Test] Unique target tokens: {unique_targets.tolist()}")
            # Additional check: check for extreme values in predictions
            if (predictions > 1e4).any() or (predictions < -1e4).any():
                print("[ERROR][Test] Extreme value detected in predictions (>1e4 or <-1e4).")
                print(f"Max prediction: {predictions.max().item()}, Min prediction: {predictions.min().item()}")
                raise RuntimeError("Sanity check failed: Extreme value in predictions (>1e4 or <-1e4) in test.")

            predictions_accum.append(predictions.cpu())

    all_preds = torch.cat(predictions_accum, dim=0)
    decoded_preds = decode_predictions(all_preds, tokenizer)


if __name__ == '__main__':
    create()