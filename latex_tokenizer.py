import re
import torch

def _tokenize(latex):
    latex = latex.strip('$')
    pattern = r'(\\[a-zA-Z]+|\\.|[0-9]+|[{}_^()\[\]=+\-*/]|[a-zA-Z]|[^ \t\n])'
    return re.findall(pattern, latex)

def collate_fn(batch):
    images, token_lists = zip(*batch)  # unzip batch

    images = torch.stack(images, dim=0)  # stack images (all same size)

    lengths = [len(t) for t in token_lists]
    max_len = max(lengths)

    padded_tokens = torch.zeros(len(token_lists), max_len, dtype=torch.long)
    for i, tokens in enumerate(token_lists):
        end = lengths[i]
        padded_tokens[i, :end] = torch.tensor(tokens, dtype=torch.long)

    return images, padded_tokens, lengths

class LatexTokenizer:
    def __init__(self):
        self.special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.token_to_id = {}
        self.id_to_token = {}

    def build_vocab(self, latex_list):
        # Note: You may consider preprocessing tokens to unify forms (e.g., normalizing curly brackets) or handling token merges here if needed
        token_set = set()
        for latex in latex_list:
            tokens = _tokenize(latex)
            token_set.update(tokens)

            # Normalize tokens to handle common LaTeX variations
            normalized_tokens = []
            for token in tokens:
                if token == '{':
                    normalized_tokens.append('LEFTBRACE')
                elif token == '}':
                    normalized_tokens.append('RIGHTBRACE')
                else:
                    normalized_tokens.append(token)
            token_set.update(normalized_tokens)

        full_vocab = self.special_tokens + sorted(token_set)
        self.token_to_id = {token: idx for idx, token in enumerate(full_vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, latex_str):
        tokens = _tokenize(latex_str)
        token_ids = [self.token_to_id.get('<SOS>')]
        for tok in tokens:
            if tok == '{':
                tok = 'LEFTBRACE'
            elif tok == '}':
                tok = 'RIGHTBRACE'
            token_ids.append(self.token_to_id.get(tok, self.token_to_id['<UNK>']))
        token_ids.append(self.token_to_id.get('<EOS>'))
        return token_ids

    def decode(self, token_ids):
        tokens = [self.id_to_token.get(tid, '<UNK>') for tid in token_ids]
        tokens = [t for t in tokens if t not in ['<SOS>', '<EOS>', '<PAD>']]
        return ' '.join(tokens)

    def tokenize_raw(self, latex_str):
        tokens = _tokenize(latex_str)
        normalized = []
        for token in tokens:
            if token == '{':
                normalized.append('LEFTBRACE')
            elif token == '}':
                normalized.append('RIGHTBRACE')
            else:
                normalized.append(token)
        return normalized

    def tokens_to_ids(self, tokens):
        ids = [self.token_to_id.get('<SOS>')]
        ids += [self.token_to_id.get(tok, self.token_to_id['<UNK>']) for tok in tokens]
        ids.append(self.token_to_id.get('<EOS>'))
        return ids

    def ids_to_tokens(self, ids):
        return [self.id_to_token.get(i, '<UNK>') for i in ids if self.id_to_token.get(i) not in self.special_tokens]
