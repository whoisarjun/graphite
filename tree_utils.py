import re


def to_tree(latex_str: str) -> list:
    token_specification = [
        ('FRAC', r'\\frac'),  # \frac
        ('SIN', r'\\sin'),  # \sin
        ('COS', r'\\cos'),  # \cos
        ('TAN', r'\\tan'),  # \tan
        ('LOG', r'\\log'),  # \log
        ('LN', r'\\ln'),  # \ln
        ('EXP', r'\\exp'),  # \exp
        ('SQRT', r'\\sqrt'),  # \sqrt
        ('VEC', r'\\vec'),  # \vec
        ('SUM', r'\\sum'),  # \sum
        ('PROD', r'\\prod'),  # \prod
        ('INT', r'\\int'),  # \int
        ('BEGIN_BMATRIX', r'\\begin\{bmatrix\}'),  # \begin{bmatrix}
        ('END_BMATRIX', r'\\end\{bmatrix\}'),  # \end{bmatrix}
        ('MATRIX_SEP', r'\\\\'),  # \\ (row separator)
        ('AMPERSAND', r'&'),  # & (column separator)
        # GREEK: All standard LaTeX Greek letters (lowercase and uppercase)
        ('GREEK',
         r'\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|varepsilon|vartheta|varpi|varrho|varsigma|varphi|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega)\b'),
        ('CARET', r'\^'),  # ^
        ('UNDERSCORE', r'_'),  # _
        ('LBRACE', r'\{'),  # {
        ('RBRACE', r'\}'),  # }
        ('LPAREN', r'\('),  # (
        ('RPAREN', r'\)'),  # )
        ('ADD', r'\+'),  # +
        ('SUB', r'-'),  # -
        ('MUL', r'\*'),  # *
        ('DIV', r'/'),  # /
        ('NUMBER', r'\d+(\.\d*)?'),  # Integer or decimal number
        ('VAR', r'[a-zA-Z=]'),  # Variables (single letter)
        ('WS', r'\s+'),  # Whitespace
        ('OTHER', r'.'),  # Any other character
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)

    def tokenize(code):
        for mo in re.finditer(tok_regex, code):
            kind = mo.lastgroup
            value = mo.group()
            if kind == 'WS':
                continue
            elif kind == 'OTHER':
                continue
            else:
                yield (kind, value)

    tokens = list(tokenize(latex_str))
    tokens = [tok for tok in tokens if tok[0] not in ('LPAREN', 'RPAREN')]
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else (None, None)

    def advance():
        nonlocal pos
        pos += 1

    def expect(kind):
        tk = peek()
        if tk[0] == kind:
            advance()
            return tk[1]
        raise ValueError(f'Expected {kind}, got {tk}')

    def parse_expr():
        tk, val = peek()
        if tk is None:
            return 'EMPTY'
        node = parse_term()
        if node == 'EMPTY':
            return 'EMPTY'
        while True:
            tk, val = peek()
            if tk == 'ADD':
                advance()
                node2 = parse_term()
                node = ['ADD', node, node2]
            elif tk == 'SUB':
                advance()
                node2 = parse_term()
                node = ['SUB', node, node2]
            else:
                break
        return node

    def parse_term():
        node = parse_factor()
        while True:
            tk, val = peek()
            if tk == 'MUL':
                advance()
                right = parse_factor()
                node = ['MUL', node, right]
            elif tk == 'DIV':
                advance()
                right = parse_factor()
                node = ['DIV', node, right]
            elif tk in ('NUMBER', 'VAR', 'GREEK', 'FRAC', 'SIN', 'COS', 'TAN', 'LOG', 'LN', 'EXP', 'SQRT', 'VEC', 'SUM',
                        'PROD', 'INT', 'BEGIN_BMATRIX'):
                right = parse_factor()
                node = ['IMPMUL', node, right]
            else:
                break
        return node

    def parse_factor():
        tk, val = peek()
        if tk == 'FRAC':
            advance()
            expect('LBRACE')
            num = parse_expr()
            expect('RBRACE')
            expect('LBRACE')
            denom = parse_expr()
            expect('RBRACE')
            return ['FRAC', num, denom]
        elif tk in ('SIN', 'COS', 'TAN', 'LOG', 'LN', 'EXP', 'SQRT', 'VEC'):
            return parse_func()
        elif tk == 'BEGIN_BMATRIX':
            return parse_matrix()
        elif tk == 'INT':
            return parse_integral()
        elif tk == 'SUM':
            return parse_sum()
        elif tk == 'PROD':
            return parse_product()
        else:
            return parse_base()

    def parse_func():
        tk, val = peek()
        func_name = tk
        advance()
        tk2, val2 = peek()
        if tk2 == 'LBRACE':
            advance()
            arg = parse_expr()
            expect('RBRACE')
        else:
            arg = parse_expr()
        return [func_name.upper(), arg]

    def parse_matrix():
        expect('BEGIN_BMATRIX')
        rows = []
        current_row = []

        while True:
            tk, val = peek()
            if tk == 'END_BMATRIX' or tk is None:
                # Add the last row if it has elements
                if current_row:
                    rows.append(current_row)
                break
            elif tk == 'MATRIX_SEP':  # \\ - row separator
                advance()
                if current_row:  # Only add row if it has elements
                    rows.append(current_row)
                    current_row = []
            elif tk == 'AMPERSAND':  # & - column separator
                advance()
                # Continue to next element in the same row
            else:
                # Parse an expression for this matrix element
                element = parse_expr()
                if element != 'EMPTY':
                    current_row.append(element)

        expect('END_BMATRIX')
        return ['MATRIX', rows]

    def parse_integral():
        advance()
        lower_expr = None
        upper_expr = None

        tk, val = peek()
        if tk == 'UNDERSCORE':
            advance()
            tk2, val2 = peek()
            if tk2 == 'LBRACE':
                advance()
                lower_expr = parse_expr()
                expect('RBRACE')
            else:
                lower_expr = parse_atom()
            tk, val = peek()
        if tk == 'CARET':
            advance()
            tk2, val2 = peek()
            if tk2 == 'LBRACE':
                advance()
                upper_expr = parse_expr()
                expect('RBRACE')
            else:
                upper_expr = parse_atom()

        integrand = parse_expr()
        if integrand == 'EMPTY':
            integrand = 'EMPTY'

        tk, val = peek()
        if tk == 'VAR' and val == 'd':
            advance()
            var = parse_atom()
            integrand = ['MUL', integrand, ['DIFFERENTIAL', var]]

        return ['INTEGRAL', lower_expr, upper_expr, integrand]

    def parse_sum():
        advance()
        lower_expr = None
        upper_expr = None

        tk, val = peek()
        if tk == 'UNDERSCORE':
            advance()
            tk2, val2 = peek()
            if tk2 == 'LBRACE':
                advance()
                lower_expr = parse_expr()
                expect('RBRACE')
            else:
                lower_expr = parse_atom()
            tk, val = peek()
        if tk == 'CARET':
            advance()
            tk2, val2 = peek()
            if tk2 == 'LBRACE':
                advance()
                upper_expr = parse_expr()
                expect('RBRACE')
            else:
                upper_expr = parse_atom()

        sum_body = parse_expr()
        if sum_body == 'EMPTY':
            sum_body = 'EMPTY'

        return ['SUM', lower_expr, upper_expr, sum_body]

    def parse_product():
        advance()
        lower_expr = None
        upper_expr = None

        tk, val = peek()
        if tk == 'UNDERSCORE':
            advance()
            tk2, val2 = peek()
            if tk2 == 'LBRACE':
                advance()
                lower_expr = parse_expr()
                expect('RBRACE')
            else:
                lower_expr = parse_atom()
            tk, val = peek()
        if tk == 'CARET':
            advance()
            tk2, val2 = peek()
            if tk2 == 'LBRACE':
                advance()
                upper_expr = parse_expr()
                expect('RBRACE')
            else:
                upper_expr = parse_atom()

        prod_body = parse_expr()
        if prod_body == 'EMPTY':
            prod_body = 'EMPTY'

        return ['PRODUCT', lower_expr, upper_expr, prod_body]

    def parse_base():
        node = parse_atom()
        tk, val = peek()
        if tk == 'CARET':
            advance()
            tk2, val2 = peek()
            if tk2 == 'LBRACE':
                advance()
                exp = parse_expr()
                expect('RBRACE')
            else:
                exp = parse_atom()
            node = ['POW', node, exp]
        return node

    def parse_atom():
        tk, val = peek()
        if tk == 'NUMBER':
            advance()
            return val
        elif tk == 'VAR':
            advance()
            return val
        elif tk == 'GREEK':
            advance()
            return val[1:]  # Remove the backslash
        elif tk == 'LBRACE':
            advance()
            node = parse_expr()
            expect('RBRACE')
            return node
        elif tk is None:
            return 'EMPTY'
        else:
            return 'EMPTY'

    tree = parse_expr()

    # Convert tree to serialized format but preserve structure
    def serialize_tree(node):
        if isinstance(node, list):
            if node[0] == 'MATRIX':
                # Special handling for matrices
                result = ['MATRIX']
                for row in node[1]:
                    result.append('ROW_START')
                    for element in row:
                        result.extend(serialize_tree(element))
                    result.append('ROW_END')
                return result
            else:
                result = [node[0]]  # operation name
                for child in node[1:]:
                    result.extend(serialize_tree(child))
                return result
        else:
            return [str(node)]

    return serialize_tree(tree)


def to_latex(serialized_tokens: list) -> str:
    def parse_serialized(tokens, pos=0):
        if pos >= len(tokens):
            return None, pos

        token = tokens[pos]
        pos += 1

        # Operations that take 2 arguments
        if token in ['ADD', 'SUB', 'MUL', 'DIV', 'IMPMUL', 'FRAC', 'POW']:
            left, pos = parse_serialized(tokens, pos)
            right, pos = parse_serialized(tokens, pos)
            return (token, left, right), pos

        # Functions that take 1 argument
        elif token in ['SIN', 'COS', 'TAN', 'LOG', 'LN', 'EXP', 'SQRT', 'VEC']:
            arg, pos = parse_serialized(tokens, pos)
            return (token, arg), pos

        # Operations with 3 arguments (lower, upper, body)
        elif token in ['SUM', 'PRODUCT', 'INTEGRAL']:
            lower, pos = parse_serialized(tokens, pos)
            upper, pos = parse_serialized(tokens, pos)
            body, pos = parse_serialized(tokens, pos)
            return (token, lower, upper, body), pos

        # Matrix parsing
        elif token == 'MATRIX':
            rows = []
            while pos < len(tokens) and tokens[pos] == 'ROW_START':
                pos += 1  # skip ROW_START
                row = []
                while pos < len(tokens) and tokens[pos] != 'ROW_END':
                    element, pos = parse_serialized(tokens, pos)
                    row.append(element)
                if pos < len(tokens) and tokens[pos] == 'ROW_END':
                    pos += 1  # skip ROW_END
                rows.append(row)
            return ('MATRIX', rows), pos

        # Differential
        elif token == 'DIFFERENTIAL':
            var, pos = parse_serialized(tokens, pos)
            return (token, var), pos

        # Skip row markers when encountered individually
        elif token in ['ROW_START', 'ROW_END']:
            return parse_serialized(tokens, pos)

        # Leaf nodes
        else:
            return token, pos

    def to_latex_str(node):
        if node is None or node == 'EMPTY':
            return ''

        GREEK_MAP = {
            'alpha': r'\alpha', 'beta': r'\beta', 'gamma': r'\gamma', 'delta': r'\delta',
            'epsilon': r'\epsilon', 'zeta': r'\zeta', 'eta': r'\eta', 'theta': r'\theta',
            'iota': r'\iota', 'kappa': r'\kappa', 'lambda': r'\lambda', 'mu': r'\mu',
            'nu': r'\nu', 'xi': r'\xi', 'omicron': r'\omicron', 'pi': r'\pi',
            'rho': r'\rho', 'sigma': r'\sigma', 'tau': r'\tau', 'upsilon': r'\upsilon',
            'phi': r'\phi', 'chi': r'\chi', 'psi': r'\psi', 'omega': r'\omega',
            'varepsilon': r'\varepsilon', 'vartheta': r'\vartheta', 'varpi': r'\varpi',
            'varrho': r'\varrho', 'varsigma': r'\varsigma', 'varphi': r'\varphi',
            'Gamma': r'\Gamma', 'Delta': r'\Delta', 'Theta': r'\Theta', 'Lambda': r'\Lambda',
            'Xi': r'\Xi', 'Pi': r'\Pi', 'Sigma': r'\Sigma', 'Upsilon': r'\Upsilon',
            'Phi': r'\Phi', 'Psi': r'\Psi', 'Omega': r'\Omega',
        }

        if isinstance(node, tuple):
            if node[0] == 'ADD':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} + {right_str}"

            elif node[0] == 'SUB':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} - {right_str}"

            elif node[0] == 'MUL':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} \\cdot {right_str}"

            elif node[0] == 'DIV':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} / {right_str}"

            elif node[0] == 'IMPMUL':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} {right_str}"

            elif node[0] == 'FRAC':
                num_str = to_latex_str(node[1])
                denom_str = to_latex_str(node[2])
                return f"\\frac{{{num_str}}}{{{denom_str}}}"

            elif node[0] == 'POW':
                base_str = to_latex_str(node[1])
                exp_str = to_latex_str(node[2])
                return f"{base_str}^{{{exp_str}}}"

            elif node[0] in ['SIN', 'COS', 'TAN', 'LOG', 'LN', 'EXP', 'SQRT', 'VEC']:
                func = '\\' + node[0].lower()
                arg_str = to_latex_str(node[1])
                return f"{func}{{{arg_str}}}"

            elif node[0] == 'SUM':
                lower_str = to_latex_str(node[1]) if node[1] else ''
                upper_str = to_latex_str(node[2]) if node[2] else ''
                body_str = to_latex_str(node[3])
                subscript = f"_{{{lower_str}}}" if lower_str else ""
                superscript = f"^{{{upper_str}}}" if upper_str else ""
                return f"\\sum{subscript}{superscript} {body_str}"

            elif node[0] == 'MATRIX':
                rows_str = []
                for row in node[1]:
                    row_elements = []
                    for element in row:
                        row_elements.append(to_latex_str(element))
                    rows_str.append(' & '.join(row_elements))
                matrix_content = ' \\\\ '.join(rows_str)
                return f"\\begin{{bmatrix}} {matrix_content} \\end{{bmatrix}}"

            elif node[0] == 'DIFFERENTIAL':
                var_str = to_latex_str(node[1])
                return f"d{var_str}"

        elif isinstance(node, str):
            if node in GREEK_MAP:
                return GREEK_MAP[node]
            return node

        return str(node)

    tree, _ = parse_serialized(serialized_tokens, 0)
    return to_latex_str(tree)


# Test the matrix functionality
test_cases = [
    r'\sum_{i=5\\i=6}^7'
]

for lll in test_cases:
    try:
        tree = to_tree(lll)
        back = to_latex(tree)
        print(f'Original: {lll}')
        print(f'Serialized: {tree}')
        print(f'Reconstructed: {back}')
        print('---')
    except Exception as e:
        print(f'Error with: {lll}')
        print(f'Error: {e}')
        print('---')

# Things to fix:
# add support for: \cdots, \ldots, \dots, \cdot, \div, \times, \pm, \parallel, \prime, \geq, \leq, \gt, \lt, \neq, \forall, \exists, \in, \mid, \lim, \limits, \to, \rightarrow, \infty, \left, \right, \Big, \Bigg, \lbrack, \rbrack, [, ]
# Allow for single symbols, i.e. replace missing data with EMPTYs
# Allow \\ for sums