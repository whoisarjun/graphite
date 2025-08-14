import re
from typing import List, Tuple, Optional


def tokenize_latex(latex_str: str) -> List[str]:
    """Tokenize LaTeX string into meaningful tokens."""
    # Remove outer delimiters if present
    latex_str = latex_str.strip()
    if latex_str.startswith('$$') and latex_str.endswith('$$'):
        latex_str = latex_str[2:-2]
    elif latex_str.startswith('$') and latex_str.endswith('$'):
        latex_str = latex_str[1:-1]
    elif latex_str.startswith('\\[') and latex_str.endswith('\\]'):
        latex_str = latex_str[2:-2]

    # Tokenization pattern - order matters!
    pattern = r'''
        \\frac(?=\s|\{)|      # fraction command (lookahead for space or brace)
        \\times|\\cdot|       # multiplication symbols
        \\[a-zA-Z]+|          # other commands like \sin, \log, etc
        \^|\\_|               # exponent or subscript
        \{|\}|                # braces
        \(|\)|                # parentheses
        \[|\]|                # brackets
        [0-9]+\.?[0-9]*|      # numbers (including decimals)
        [a-zA-Z]+|            # variables (can be multi-letter)
        [+\-*/=<>]|           # operators
        \s+|                  # whitespace
        .                     # any other character
    '''

    tokens = re.findall(pattern, latex_str, re.VERBOSE)
    # Filter out whitespace tokens
    return [token for token in tokens if not token.isspace()]


def safe_parse_expression(tokens: List[str]) -> List[str]:
    """Parse tokens into flattened prefix notation with maximum fault tolerance."""

    def peek(pos: int) -> Optional[str]:
        return tokens[pos] if pos < len(tokens) else None

    def safe_parse_primary(pos: int) -> Tuple[List[str], int]:
        """Parse primary expressions - never fails, always returns something."""
        if pos >= len(tokens):
            return [], pos

        token = tokens[pos]

        if token == '\\frac':
            # Try to parse fraction, but don't fail if malformed
            pos += 1

            # Try to get numerator
            if peek(pos) == '{':
                try:
                    num_tokens, pos = safe_parse_braced_group(pos)
                except:
                    num_tokens = []
            else:
                # No brace? Just take next token as numerator
                if pos < len(tokens):
                    num_tokens = [tokens[pos]]
                    pos += 1
                else:
                    num_tokens = []

            # Try to get denominator
            if peek(pos) == '{':
                try:
                    den_tokens, pos = safe_parse_braced_group(pos)
                except:
                    den_tokens = []
            else:
                # No brace? Just take next token as denominator
                if pos < len(tokens):
                    den_tokens = [tokens[pos]]
                    pos += 1
                else:
                    den_tokens = []

            return ['FRAC'] + num_tokens + den_tokens, pos

        elif token == '{':
            # Try to parse braced group
            try:
                return safe_parse_braced_group(pos)
            except:
                # If braced group fails, just return the brace as a token
                return [token], pos + 1

        elif token == '(':
            # Try to parse parenthesized expression
            pos += 1
            try:
                expr_tokens, pos = safe_parse_additive(pos)
                if peek(pos) == ')':
                    pos += 1  # consume closing paren
                return expr_tokens, pos
            except:
                # If parsing fails, just return what we have
                return [], pos

        else:
            # Any other token - just return it
            return [token], pos + 1

    def safe_parse_braced_group(pos: int) -> Tuple[List[str], int]:
        """Parse expression inside braces - tries hard but doesn't fail."""
        if pos >= len(tokens) or tokens[pos] != '{':
            return [], pos

        pos += 1  # skip opening brace

        # Collect tokens until closing brace or end
        group_tokens = []
        brace_count = 1
        start_pos = pos

        while pos < len(tokens) and brace_count > 0:
            token = tokens[pos]
            if token == '{':
                brace_count += 1
            elif token == '}':
                brace_count -= 1

            if brace_count > 0:  # Don't include the final closing brace
                group_tokens.append(token)
            pos += 1

        # If we collected tokens, try to parse them
        if group_tokens:
            try:
                parsed_tokens = safe_parse_expression(group_tokens)
                return parsed_tokens, pos
            except:
                # If parsing fails, return raw tokens
                return group_tokens, pos

        return [], pos

    def safe_parse_power(pos: int) -> Tuple[List[str], int]:
        """Parse powers and subscripts - handles malformed input gracefully."""
        base_tokens, pos = safe_parse_primary(pos)

        while pos < len(tokens) and tokens[pos] in ['^', '_']:
            op_token = tokens[pos]
            pos += 1

            # Try to get the exponent/subscript
            if peek(pos) == '{':
                try:
                    exp_tokens, pos = safe_parse_braced_group(pos)
                except:
                    exp_tokens = []
            elif pos < len(tokens):
                exp_tokens, pos = safe_parse_primary(pos)
            else:
                exp_tokens = []

            op = 'POW' if op_token == '^' else 'SUBSCRIPT'
            base_tokens = [op] + base_tokens + exp_tokens

        return base_tokens, pos

    def safe_parse_multiplicative(pos: int) -> Tuple[List[str], int]:
        """Parse multiplication - very permissive."""
        left_tokens, pos = safe_parse_power(pos)

        while pos < len(tokens):
            token = peek(pos)

            if not token:
                break

            # Explicit multiplication
            if token in ['\\times', '\\cdot', '*']:
                pos += 1
                right_tokens, pos = safe_parse_power(pos)
                left_tokens = ['MUL'] + left_tokens + right_tokens

            # Implicit multiplication: very permissive check
            elif (token not in ['+', '-', ')', '}', '^', '_', '/', '=', '<', '>'] and
                  token not in ['\\times', '\\cdot', '*']):

                # Try to parse as multiplication
                try:
                    right_tokens, new_pos = safe_parse_power(pos)
                    if right_tokens:  # Only if we got something
                        left_tokens = ['MUL'] + left_tokens + right_tokens
                        pos = new_pos
                    else:
                        break
                except:
                    break
            else:
                break

        return left_tokens, pos

    def safe_parse_additive(pos: int) -> Tuple[List[str], int]:
        """Parse addition and subtraction."""
        left_tokens, pos = safe_parse_multiplicative(pos)

        while pos < len(tokens) and tokens[pos] in ['+', '-']:
            op_token = tokens[pos]
            pos += 1
            right_tokens, pos = safe_parse_multiplicative(pos)

            op = 'ADD' if op_token == '+' else 'MINUS'
            left_tokens = [op] + left_tokens + right_tokens

        return left_tokens, pos

    # Start parsing - always succeeds
    if not tokens:
        return []

    try:
        result_tokens, final_pos = safe_parse_additive(0)

        # If we didn't consume all tokens, append the rest as raw tokens
        if final_pos < len(tokens):
            remaining = tokens[final_pos:]
            result_tokens.extend(remaining)

        return result_tokens
    except:
        # Ultimate fallback - return all tokens as-is
        return tokens


def to_tree(latex_str: str) -> List[str]:
    """Convert LaTeX string to flattened prefix notation - never fails."""
    tokens = tokenize_latex(latex_str)
    if not tokens:
        return []

    # Try structured parsing first
    try:
        result = safe_parse_expression(tokens)
        if result:
            return result
    except:
        pass

    # Fallback: return raw tokens
    return tokens


def prefix_to_latex(tokens: List[str]) -> str:
    """Convert prefix notation back to LaTeX - handles malformed input."""

    def safe_parse_tokens(pos: int) -> Tuple[str, int]:
        """Parse tokens - never crashes."""
        if pos >= len(tokens):
            return "", pos

        token = tokens[pos]

        try:
            if token == 'FRAC':
                pos += 1
                num_latex, pos = safe_parse_tokens(pos)
                den_latex, pos = safe_parse_tokens(pos)
                return f"\\frac{{{num_latex}}}{{{den_latex}}}", pos

            elif token == 'POW':
                pos += 1
                base_latex, pos = safe_parse_tokens(pos)
                exp_latex, pos = safe_parse_tokens(pos)

                if ' ' in exp_latex or any(op in exp_latex for op in ['+', '-', '*', '/']):
                    return f"{base_latex}^{{{exp_latex}}}", pos
                else:
                    return f"{base_latex}^{exp_latex}", pos

            elif token == 'SUBSCRIPT':
                pos += 1
                base_latex, pos = safe_parse_tokens(pos)
                sub_latex, pos = safe_parse_tokens(pos)

                if ' ' in sub_latex or any(op in sub_latex for op in ['+', '-', '*', '/']):
                    return f"{base_latex}_{{{sub_latex}}}", pos
                else:
                    return f"{base_latex}_{sub_latex}", pos

            elif token == 'ADD':
                pos += 1
                left_latex, pos = safe_parse_tokens(pos)
                right_latex, pos = safe_parse_tokens(pos)
                return f"{left_latex} + {right_latex}", pos

            elif token == 'MINUS':
                pos += 1
                left_latex, pos = safe_parse_tokens(pos)
                right_latex, pos = safe_parse_tokens(pos)
                return f"{left_latex} - {right_latex}", pos

            elif token == 'MUL':
                pos += 1
                left_latex, pos = safe_parse_tokens(pos)
                right_latex, pos = safe_parse_tokens(pos)
                return f"{left_latex} {right_latex}", pos

            else:
                # Terminal token or unknown - just return it
                return token, pos + 1

        except:
            # If anything fails, just return the token as-is
            return token, pos + 1

    if not tokens:
        return ""

    try:
        result, _ = safe_parse_tokens(0)
        return result
    except:
        # Ultimate fallback - join all tokens
        return " ".join(tokens)


def to_latex(serialized_tokens: List[str]) -> str:
    """Convert flattened prefix tokens back to LaTeX string - never fails."""
    return prefix_to_latex(serialized_tokens)

def clean_latex(latex: str) -> str:
    return latex

# Test the functions
if __name__ == "__main__":
    # Test with well-formed input
    test_latex = r"\frac{2}{\frac{3 m - 2 n \times 9^n - 9^m}{2 n - 1}}"
    print("=== Well-formed input ===")
    print(f"Input: {test_latex}")
    prefix_tokens = to_tree(test_latex)
    print(f"Prefix: {prefix_tokens}")
    reconstructed = to_latex(prefix_tokens)
    print(f"Output: {reconstructed}")
    print()

    # Test with malformed/incomplete input
    malformed_tests = [
        ")",  # just a closing paren
        "\\frac",  # incomplete fraction
        "\\frac{a}",  # missing denominator
        "x^",  # incomplete power
        "{a + b",  # unmatched brace
        "x + + y",  # double operator
        "",  # empty string
        "x y z ) (",  # mixed valid/invalid
    ]

    print("=== Malformed input tests ===")
    for test in malformed_tests:
        print(f"Input: '{test}'")
        prefix_tokens = to_tree(test)
        print(f"Prefix: {prefix_tokens}")
        reconstructed = to_latex(prefix_tokens)
        print(f"Output: '{reconstructed}'")
        print()