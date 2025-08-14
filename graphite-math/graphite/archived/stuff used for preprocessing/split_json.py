import json
import random
import os

# File paths
input_path = 'pairs.json'
train_path = 'train.json'
val_path = 'val.json'
test_path = 'test.json'

# Load all data
with open(input_path, 'r') as f:
    data = json.load(f)
    all_pairs = data['pairs']

# Shuffle the data
random.shuffle(all_pairs)

# Calculate split sizes
total = len(all_pairs)
train_end = int(0.7 * total)
val_end = train_end + int(0.15 * total)

train_pairs = all_pairs[:train_end]
val_pairs = all_pairs[train_end:val_end]
test_pairs = all_pairs[val_end:]

# Save splits
def save_json(path, pairs):
    with open(path, 'w') as f:
        json.dump({'pairs': pairs}, f, indent=2)

save_json(train_path, train_pairs)
save_json(val_path, val_pairs)
save_json(test_path, test_pairs)

print(f'Split complete: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test')