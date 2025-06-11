import json
import re

# Load your JSON data
with open('./datasets/pairs.json', 'r') as f:
    data = json.load(f)

unique_tokens = set()

# Regex to catch LaTeX commands (like \frac, \sum), numbers, letters, operators, etc.
token_pattern = re.compile(
    r'(\\[a-zA-Z]+)|'      # LaTeX commands
    r'([0-9]+(?:\.[0-9]+)?)|'  # Numbers
    r'([a-zA-Z])|'         # Letters
    r'([\+\-\*/\^\=\(\)\{\}\[\],])'  # Operators
)

for pair in data['pairs']:
    latex = pair.get('latex', '')
    if not isinstance(latex, str):
        continue
    latex = latex.replace('$', '')
    matches = token_pattern.findall(latex)
    for match in matches:
        token = next(t for t in match if t)
        # Normalize numbers to <NUM>
        if re.match(r'^[0-9]+(?:\.[0-9]+)?$', token):
            token = '<NUM>'
        unique_tokens.add(token)

with open('latex_tags.txt', 'w') as f:
    f.write('\n'.join(sorted(unique_tokens)))