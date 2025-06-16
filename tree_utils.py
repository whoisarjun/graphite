import re

def to_tree(latex_str: str) -> list:
    token_specification = [
        ('COLON', r':'),  # :
        ('COMMA', r','),  # ,
        ('FRAC', r'\\frac'),  # \frac
        ('SIN', r'\\sin'),  # \sin
        ('COS', r'\\cos'),  # \cos
        ('TAN', r'\\tan'),  # \tan
        ('LOG', r'\\log'),  # \log
        ('LN', r'\\ln'),  # \ln
        ('EXP', r'\\exp'),  # \exp
        ('SQRT', r'\\sqrt'),  # \sqrt
        ('VEC', r'\\vec'),  # \vec
        ('TEXT_CMD', r'\\(mbox|text|mathrm|mathit|mathbf|mathsf|mathtt|mathcal|mathfrak)\b'),  # Text commands
        ('SUM', r'\\sum'),  # \sum
        ('PROD', r'\\prod'),  # \prod
        ('INT', r'\\int'),  # \int
        ('LIM', r'\\lim'),  # \lim
        ('LIMITS', r'\\limits'),  # \limits
        ('BEGIN_BMATRIX', r'\\begin\{bmatrix\}'),  # \begin{bmatrix}
        ('END_BMATRIX', r'\\end\{bmatrix\}'),  # \end{bmatrix}
        ('MATRIX_SEP', r'\\\\'),  # \\ (row separator)
        ('AMPERSAND', r'&'),  # & (column separator)
        ('CDOTS', r'\\cdots'),  # \cdots
        ('LDOTS', r'\\ldots'),  # \ldots
        ('DOTS', r'\\dots'),  # \dots
        ('CDOT', r'\\cdot'),  # \cdot
        ('DIV_SYM', r'\\div'),  # \div
        ('TIMES', r'\\times'),  # \times
        ('PM', r'\\pm'),  # \pm
        ('PARALLEL', r'\\parallel'),  # \parallel
        ('PRIME', r'\\prime'),  # \prime
        ('GEQ', r'\\geq'),  # \geq
        ('LEQ', r'\\leq'),  # \leq
        ('GT', r'\\gt'),  # \gt
        ('LT', r'\\lt'),  # \lt
        ('NEQ', r'\\neq'),  # \neq
        ('FORALL', r'\\forall'),  # \forall
        ('EXISTS', r'\\exists'),  # \exists
        ('MID', r'\\mid'),  # \mid
        ('TO', r'\\to'),  # \to
        ('RIGHTARROW', r'\\rightarrow'),  # \rightarrow
        ('INFTY', r'\\infty'),  # \infty
        ('IN', r'\\in'),  # \in
        ('LEFT', r'\\left'),  # \left
        ('RIGHT', r'\\right'),  # \right
        ('BIG', r'\\Big'),  # \Big
        ('BIGG', r'\\Bigg'),  # \Bigg
        ('LBRACK_CMD', r'\\lbrack'),  # \lbrack
        ('RBRACK_CMD', r'\\rbrack'),  # \rbrack
        # GREEK: All standard LaTeX Greek letters (lowercase and uppercase)
        ('GREEK',
         r'\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|varepsilon|vartheta|varpi|varrho|varsigma|varphi|Gamma|Delta|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega)\b'),
        ('CARET', r'\^'),  # ^
        ('UNDERSCORE', r'_'),  # _
        ('LBRACE', r'\{'),  # {
        ('RBRACE', r'\}'),  # }
        ('LPAREN', r'\('),  # (
        ('RPAREN', r'\)'),  # )
        ('LBRACK', r'\['),  # [
        ('RBRACK', r'\]'),  # ]
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
                if node2 == 'EMPTY':
                    node2 = 'EMPTY'
                node = ['ADD', node, node2]
            elif tk == 'SUB':
                advance()
                node2 = parse_term()
                if node2 == 'EMPTY':
                    node2 = 'EMPTY'
                node = ['SUB', node, node2]
            elif tk in ('GEQ', 'LEQ', 'GT', 'LT', 'NEQ', 'IN', 'TO', 'RIGHTARROW', 'PARALLEL'):
                op = tk
                advance()
                node2 = parse_term()
                if node2 == 'EMPTY':
                    node2 = 'EMPTY'
                node = [op, node, node2]
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
                if right == 'EMPTY':
                    right = 'EMPTY'
                node = ['MUL', node, right]
            elif tk == 'DIV':
                advance()
                right = parse_factor()
                if right == 'EMPTY':
                    right = 'EMPTY'
                node = ['DIV', node, right]
            elif tk in ('CDOT', 'TIMES', 'DIV_SYM'):
                op = tk
                advance()
                right = parse_factor()
                if right == 'EMPTY':
                    right = 'EMPTY'
                node = [op, node, right]
            elif tk == 'PM':
                advance()
                right = parse_factor()
                if right == 'EMPTY':
                    right = 'EMPTY'
                node = ['PM', node, right]
            elif tk in ('NUMBER', 'VAR', 'GREEK', 'FRAC', 'SIN', 'COS', 'TAN', 'LOG', 'LN', 'EXP', 'SQRT', 'VEC', 'SUM',
                        'PROD', 'INT', 'LIM', 'BEGIN_BMATRIX', 'LEFT', 'FORALL', 'EXISTS', 'INFTY', 'CDOTS', 'LDOTS',
                        'DOTS', 'PRIME', 'COLON', 'LPAREN', 'RPAREN', 'LBRACK', 'RBRACK', 'COMMA',
                        'MATRIX_SEP'):  # ADDED comma and matrix_sep
                right = parse_factor()
                if right == 'EMPTY':
                    right = 'EMPTY'
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
            if num == 'EMPTY':
                num = 'EMPTY'
            expect('RBRACE')
            expect('LBRACE')
            denom = parse_expr()
            if denom == 'EMPTY':
                denom = 'EMPTY'
            expect('RBRACE')
            return ['FRAC', num, denom]
        elif tk in ('SIN', 'COS', 'TAN', 'LOG', 'LN', 'EXP', 'SQRT', 'VEC'):
            return parse_func()
        elif tk == 'TEXT_CMD':
            return parse_text_cmd()
        elif tk == 'LIM':
            return parse_limit()
        elif tk == 'BEGIN_BMATRIX':
            return parse_matrix()
        elif tk == 'INT':
            return parse_integral()
        elif tk == 'SUM':
            return parse_sum()
        elif tk == 'PROD':
            return parse_product()
        elif tk == 'LEFT':
            return parse_delimited()
        elif tk in ('FORALL', 'EXISTS', 'INFTY', 'CDOTS', 'LDOTS', 'DOTS', 'PRIME'):
            symbol = tk
            advance()
            return [symbol]
        # CHANGED: Handle unary minus with better precedence
        elif tk == 'SUB':
            advance()
            operand = parse_base()  # Use parse_base instead of parse_factor for better precedence
            return ['MINUS', operand]
        # ADDED: Handle unary plus
        elif tk == 'ADD':
            advance()
            operand = parse_base()  # Use parse_base instead of parse_factor for better precedence
            return ['PLUS', operand]
        else:
            return parse_base()

    def parse_func():
        tk, val = peek()
        func_name = tk
        advance()

        # Special handling for \frac which takes two arguments
        if func_name == 'FRAC':
            # Parse numerator
            tk2, val2 = peek()
            if tk2 == 'LBRACE':
                advance()
                brace_count = 1
                arg_tokens = []
                while brace_count > 0:
                    tk, val = peek()
                    if tk is None:
                        raise ValueError("Unclosed brace in function argument")
                    if tk == 'LBRACE':
                        brace_count += 1
                    elif tk == 'RBRACE':
                        brace_count -= 1
                    if brace_count > 0:
                        arg_tokens.append((tk, val))
                    advance()

                if not arg_tokens:
                    arg1 = 'EMPTY'
                else:
                    old_tokens = tokens
                    old_pos = pos
                    globals()['tokens'] = arg_tokens
                    globals()['pos'] = 0
                    try:
                        arg1 = parse_expr()
                    finally:
                        globals()['tokens'] = old_tokens
                        globals()['pos'] = old_pos
            else:
                arg1 = parse_factor()

            # Parse denominator
            tk3, val3 = peek()
            if tk3 == 'LBRACE':
                advance()
                brace_count = 1
                arg_tokens = []
                while brace_count > 0:
                    tk, val = peek()
                    if tk is None:
                        raise ValueError("Unclosed brace in function argument")
                    if tk == 'LBRACE':
                        brace_count += 1
                    elif tk == 'RBRACE':
                        brace_count -= 1
                    if brace_count > 0:
                        arg_tokens.append((tk, val))
                    advance()

                if not arg_tokens:
                    arg2 = 'EMPTY'
                else:
                    old_tokens = tokens
                    old_pos = pos
                    globals()['tokens'] = arg_tokens
                    globals()['pos'] = 0
                    try:
                        arg2 = parse_expr()
                    finally:
                        globals()['tokens'] = old_tokens
                        globals()['pos'] = old_pos
            else:
                arg2 = parse_factor()

            return ['FRAC', arg1, arg2]

        # Handle other functions (single argument)
        else:
            tk2, val2 = peek()
            if tk2 == 'LBRACE':
                advance()
                brace_count = 1
                arg_tokens = []
                while brace_count > 0:
                    tk, val = peek()
                    if tk is None:
                        raise ValueError("Unclosed brace in function argument")
                    if tk == 'LBRACE':
                        brace_count += 1
                    elif tk == 'RBRACE':
                        brace_count -= 1
                    if brace_count > 0:
                        arg_tokens.append((tk, val))
                    advance()

                if not arg_tokens:
                    arg = 'EMPTY'
                else:
                    old_tokens = tokens
                    old_pos = pos
                    globals()['tokens'] = arg_tokens
                    globals()['pos'] = 0
                    try:
                        arg = parse_expr()
                    finally:
                        globals()['tokens'] = old_tokens
                        globals()['pos'] = old_pos
            else:
                arg = parse_factor()

            return [func_name, arg]

    def parse_text_cmd():
        tk, val = peek()
        cmd_name = val  # This will be something like '\mbox'
        advance()
        expect('LBRACE')

        # Parse content inside braces more carefully
        # We need to parse until we find the matching closing brace
        brace_count = 1
        content_tokens = []

        while brace_count > 0:
            tk, val = peek()
            if tk is None:
                raise ValueError("Unclosed brace in text command")
            if tk == 'LBRACE':
                brace_count += 1
            elif tk == 'RBRACE':
                brace_count -= 1

            if brace_count > 0:  # Don't include the final closing brace
                content_tokens.append((tk, val))
            advance()

        # Now parse the collected tokens as content
        if not content_tokens:
            content = 'EMPTY'
        else:
            # Create a mini-parser for the content tokens
            old_tokens = tokens
            old_pos = pos
            globals()['tokens'] = content_tokens
            globals()['pos'] = 0

            try:
                content = parse_expr()
            finally:
                globals()['tokens'] = old_tokens
                globals()['pos'] = old_pos

        return ['TEXT_CMD', cmd_name, content]

    def parse_limit():
        advance()  # consume \lim
        lower_expr = None

        # Check for \limits modifier
        tk, val = peek()
        if tk == 'LIMITS':
            advance()

        # Check for subscript (limit variable and value)
        tk, val = peek()
        if tk == 'UNDERSCORE':
            advance()
            tk2, val2 = peek()
            if tk2 == 'LBRACE':
                advance()
                lower_expr = parse_expr()
                if lower_expr == 'EMPTY':
                    lower_expr = 'EMPTY'
                expect('RBRACE')
            else:
                lower_expr = parse_atom()
                if lower_expr == 'EMPTY':
                    lower_expr = 'EMPTY'

        # Parse the expression being limited
        body = parse_expr()
        if body == 'EMPTY':
            body = 'EMPTY'

        return ['LIMIT', lower_expr, body]

    def parse_delimited():
        advance()  # consume \left
        tk, val = peek()
        if tk == 'LPAREN':
            left_delim = '('
            advance()
        else:
            left_delim = parse_atom()

        content = parse_expr()
        if content == 'EMPTY':
            content = 'EMPTY'
        expect('RIGHT')

        tk, val = peek()
        if tk == 'RPAREN':
            right_delim = ')'
            advance()
        else:
            right_delim = parse_atom()

        return ['DELIMITED', left_delim, content, right_delim]

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
            elif tk == 'MATRIX_SEP':  # \\ - row separator (ONLY treated specially in matrix context)
                advance()
                if current_row:  # Only add row if it has elements
                    rows.append(current_row)
                    current_row = []
            elif tk == 'AMPERSAND':  # & - column separator (ONLY treated specially in matrix context)
                advance()
                # Continue to next element in the same row
            else:
                # Parse an expression for this matrix element
                element = parse_expr()
                if element == 'EMPTY':
                    element = 'EMPTY'
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
                if lower_expr == 'EMPTY':
                    lower_expr = 'EMPTY'
                expect('RBRACE')
            else:
                lower_expr = parse_atom()
                if lower_expr == 'EMPTY':
                    lower_expr = 'EMPTY'
            tk, val = peek()
        if tk == 'CARET':
            advance()
            tk2, val2 = peek()
            if tk2 == 'LBRACE':
                advance()
                upper_expr = parse_expr()
                if upper_expr == 'EMPTY':
                    upper_expr = 'EMPTY'
                expect('RBRACE')
            else:
                upper_expr = parse_atom()
                if upper_expr == 'EMPTY':
                    upper_expr = 'EMPTY'

        integrand = parse_expr()
        if integrand == 'EMPTY':
            integrand = 'EMPTY'

        tk, val = peek()
        if tk == 'VAR' and val == 'd':
            advance()
            var = parse_atom()
            if var == 'EMPTY':
                var = 'EMPTY'
            integrand = ['MUL', integrand, ['DIFFERENTIAL', var]]

        return ['INTEGRAL', lower_expr, upper_expr, integrand]

    def parse_sum():
        advance()  # consume \sum
        lower_expr = None
        upper_expr = None

        # Parse subscript
        tk, val = peek()
        if tk == 'UNDERSCORE':
            advance()
            if peek()[0] == 'LBRACE':
                advance()
                lower_expr = parse_expr()
                expect('RBRACE')
            else:
                lower_expr = parse_atom()

        # Parse superscript
        tk, val = peek()
        if tk == 'CARET':
            advance()
            if peek()[0] == 'LBRACE':
                advance()
                upper_expr = parse_expr()
                expect('RBRACE')
            else:
                upper_expr = parse_atom()

        # Parse the summand (what comes after)
        sum_body = parse_factor()  # NOT parse_expr()!

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
                if lower_expr == 'EMPTY':
                    lower_expr = 'EMPTY'
                expect('RBRACE')
            else:
                lower_expr = parse_atom()
                if lower_expr == 'EMPTY':
                    lower_expr = 'EMPTY'
            tk, val = peek()
        if tk == 'CARET':
            advance()
            tk2, val2 = peek()
            if tk2 == 'LBRACE':
                advance()
                upper_expr = parse_expr()
                if upper_expr == 'EMPTY':
                    upper_expr = 'EMPTY'
                expect('RBRACE')
            else:
                upper_expr = parse_atom()
                if upper_expr == 'EMPTY':
                    upper_expr = 'EMPTY'

        prod_body = parse_expr()
        if prod_body == 'EMPTY':
            prod_body = 'EMPTY'

        return ['PRODUCT', lower_expr, upper_expr, prod_body]

    def parse_base():
        node = parse_atom()

        # Handle both subscripts and superscripts
        while True:
            tk, val = peek()
            if tk == 'UNDERSCORE':
                advance()
                tk2, val2 = peek()
                if tk2 == 'LBRACE':
                    advance()
                    sub = parse_expr()
                    expect('RBRACE')
                else:
                    sub = parse_atom()
                    if sub == 'EMPTY':
                        sub = 'EMPTY'
                node = ['SUBSCRIPT', node, sub]
            elif tk == 'CARET':
                advance()
                tk2, val2 = peek()
                if tk2 == 'LBRACE':
                    advance()
                    exp = parse_expr()
                    expect('RBRACE')
                else:
                    exp = parse_atom()
                    if exp == 'EMPTY':
                        exp = 'EMPTY'
                node = ['POW', node, exp]
            else:
                break

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

            # Similar fix for grouped expressions
            brace_count = 1
            group_tokens = []

            while brace_count > 0:
                tk, val = peek()
                if tk is None:
                    raise ValueError("Unclosed brace in grouped expression")
                if tk == 'LBRACE':
                    brace_count += 1
                elif tk == 'RBRACE':
                    brace_count -= 1

                if brace_count > 0:  # Don't include the final closing brace
                    group_tokens.append((tk, val))
                advance()

            # Parse the collected tokens
            if not group_tokens:
                node = 'EMPTY'
            else:
                old_tokens = tokens
                old_pos = pos
                globals()['tokens'] = group_tokens
                globals()['pos'] = 0

                try:
                    node = parse_expr()
                finally:
                    globals()['tokens'] = old_tokens
                    globals()['pos'] = old_pos

            return node
        # CHANGED: Treat brackets, parentheses, colon, comma, and matrix separator as regular variables
        elif tk in ('LBRACK', 'LBRACK_CMD', 'RBRACK', 'RBRACK_CMD', 'LPAREN', 'RPAREN', 'COLON', 'COMMA', 'MATRIX_SEP'):
            advance()
            return val
        elif tk == 'MID':
            advance()
            return ['MID']
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

    return ['SOS'] + serialize_tree(tree) + ['EOS']

def to_latex(serialized_tokens: list) -> str:
    if serialized_tokens[0] == 'SOS':
        serialized_tokens = serialized_tokens[1:]
    if serialized_tokens[-1] == 'EOS':
        serialized_tokens = serialized_tokens[:-1]

    def parse_serialized(tokens, pos=0):
        if pos >= len(tokens):
            return None, pos

        token = tokens[pos]
        pos += 1

        # Operations that take 2 arguments
        if token in ['ADD', 'SUB', 'MUL', 'DIV', 'IMPMUL', 'FRAC', 'POW', 'SUBSCRIPT', 'CDOT', 'TIMES', 'DIV_SYM',
                     'PM',
                     'GEQ', 'LEQ', 'GT', 'LT', 'NEQ', 'IN', 'TO', 'RIGHTARROW', 'PARALLEL']:
            left, pos = parse_serialized(tokens, pos)
            right, pos = parse_serialized(tokens, pos)
            return (token, left, right), pos

        # Functions that take 1 argument (INCLUDING MINUS and PLUS)
        elif token in ['SIN', 'COS', 'TAN', 'LOG', 'LN', 'EXP', 'SQRT', 'VEC', 'BRACKETS', 'MINUS', 'PLUS']:
            arg, pos = parse_serialized(tokens, pos)
            return (token, arg), pos

        # Limit with 2 arguments (lower bound, body)
        elif token == 'LIMIT':
            lower, pos = parse_serialized(tokens, pos)
            body, pos = parse_serialized(tokens, pos)
            return (token, lower, body), pos

        # Operations with 3 arguments (lower, upper, body)
        elif token in ['SUM', 'PRODUCT', 'INTEGRAL']:
            lower, pos = parse_serialized(tokens, pos)
            upper, pos = parse_serialized(tokens, pos)
            body, pos = parse_serialized(tokens, pos)
            return (token, lower, upper, body), pos

        # Delimited expressions with 3 arguments (left_delim, content, right_delim)
        elif token == 'DELIMITED':
            left_delim, pos = parse_serialized(tokens, pos)
            content, pos = parse_serialized(tokens, pos)
            right_delim, pos = parse_serialized(tokens, pos)
            return (token, left_delim, content, right_delim), pos

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

        # Single symbols
        elif token in ['FORALL', 'EXISTS', 'INFTY', 'CDOTS', 'LDOTS', 'DOTS', 'PRIME', 'MID']:
            return (token,), pos

        # Skip row markers when encountered individually
        elif token in ['ROW_START', 'ROW_END']:
            return parse_serialized(tokens, pos)

        # Leaf nodes
        else:
            return token, pos

    def parse_serialized(tokens, pos=0):
        if pos >= len(tokens):
            return None, pos

        token = tokens[pos]
        pos += 1

        # Operations that take 2 arguments
        if token in ['ADD', 'SUB', 'MUL', 'DIV', 'IMPMUL', 'FRAC', 'POW', 'SUBSCRIPT', 'CDOT', 'TIMES', 'DIV_SYM',
                     'PM',
                     'GEQ', 'LEQ', 'GT', 'LT', 'NEQ', 'IN', 'TO', 'RIGHTARROW', 'PARALLEL']:
            left, pos = parse_serialized(tokens, pos)
            right, pos = parse_serialized(tokens, pos)
            return (token, left, right), pos

        # Functions that take 1 argument (INCLUDING MINUS and PLUS)
        elif token in ['SIN', 'COS', 'TAN', 'LOG', 'LN', 'EXP', 'SQRT', 'VEC', 'BRACKETS', 'MINUS', 'PLUS']:
            arg, pos = parse_serialized(tokens, pos)
            return (token, arg), pos

        # Limit with 2 arguments (lower bound, body)
        elif token == 'LIMIT':
            lower, pos = parse_serialized(tokens, pos)
            body, pos = parse_serialized(tokens, pos)
            return (token, lower, body), pos

        # Operations with 3 arguments (lower, upper, body)
        elif token in ['SUM', 'PRODUCT', 'INTEGRAL']:
            lower, pos = parse_serialized(tokens, pos)
            upper, pos = parse_serialized(tokens, pos)
            body, pos = parse_serialized(tokens, pos)
            return (token, lower, upper, body), pos

        # Delimited expressions with 3 arguments (left_delim, content, right_delim)
        elif token == 'DELIMITED':
            left_delim, pos = parse_serialized(tokens, pos)
            content, pos = parse_serialized(tokens, pos)
            right_delim, pos = parse_serialized(tokens, pos)
            return (token, left_delim, content, right_delim), pos

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

        # Single symbols
        elif token in ['FORALL', 'EXISTS', 'INFTY', 'CDOTS', 'LDOTS', 'DOTS', 'PRIME', 'MID']:
            return (token,), pos

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

        SYMBOL_MAP = {
            'CDOTS': r'\cdots',
            'LDOTS': r'\ldots',
            'DOTS': r'\dots',
            'PRIME': r'\prime',
            'INFTY': r'\infty',
            'FORALL': r'\forall',
            'EXISTS': r'\exists',
            'MID': r'\mid'
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

            elif node[0] == 'CDOT':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} \\cdot {right_str}"

            elif node[0] == 'TIMES':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} \\times {right_str}"

            elif node[0] == 'DIV':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} / {right_str}"

            elif node[0] == 'DIV_SYM':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} \\div {right_str}"

            elif node[0] == 'PM':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} \\pm {right_str}"

            # CHANGED: Remove space for implicit multiplication
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

            # ADDED: Handle subscripts
            elif node[0] == 'SUBSCRIPT':
                base_str = to_latex_str(node[1])
                sub_str = to_latex_str(node[2])
                return f"{base_str}_{{{sub_str}}}"

            elif node[0] == 'TEXT_CMD':
                cmd_str = node[1]  # e.g., '\mbox'
                content_str = to_latex_str(node[2])
                return f"{cmd_str}{{{content_str}}}"

            # ADDED: Handle unary minus and plus
            elif node[0] == 'MINUS':
                operand_str = to_latex_str(node[1])
                return f"-{operand_str}"

            elif node[0] == 'PLUS':
                operand_str = to_latex_str(node[1])
                return f"+{operand_str}"

            elif node[0] in ['SIN', 'COS', 'TAN', 'LOG', 'LN', 'EXP', 'SQRT', 'VEC']:
                func = '\\' + node[0].lower()
                arg_str = to_latex_str(node[1])
                return f"{func}{{{arg_str}}}"

            elif node[0] == 'LIMIT':
                lower_str = to_latex_str(node[1]) if node[1] else ''
                body_str = to_latex_str(node[2])
                subscript = f"_{{{lower_str}}}" if lower_str else ""
                return f"\\lim{subscript} {body_str}"

            elif node[0] == 'SUM':
                lower_str = to_latex_str(node[1]) if node[1] else ''
                upper_str = to_latex_str(node[2]) if node[2] else ''
                body_str = to_latex_str(node[3])
                subscript = f"_{{{lower_str}}}" if lower_str else ""
                superscript = f"^{{{upper_str}}}" if upper_str else ""
                return f"\\sum{subscript}{superscript} {body_str}"

            elif node[0] == 'PRODUCT':
                lower_str = to_latex_str(node[1]) if node[1] else ''
                upper_str = to_latex_str(node[2]) if node[2] else ''
                body_str = to_latex_str(node[3])
                subscript = f"_{{{lower_str}}}" if lower_str else ""
                superscript = f"^{{{upper_str}}}" if upper_str else ""
                return f"\\prod{subscript}{superscript} {body_str}"

            elif node[0] == 'INTEGRAL':
                lower_str = to_latex_str(node[1]) if node[1] else ''
                upper_str = to_latex_str(node[2]) if node[2] else ''
                body_str = to_latex_str(node[3])
                subscript = f"_{{{lower_str}}}" if lower_str else ""
                superscript = f"^{{{upper_str}}}" if upper_str else ""
                return f"\\int{subscript}{superscript} {body_str}"

            elif node[0] == 'MATRIX':
                rows_str = []
                for row in node[1]:
                    row_elements = []
                    for element in row:
                        row_elements.append(to_latex_str(element))
                    rows_str.append(' & '.join(row_elements))
                matrix_content = ' \\\\ '.join(rows_str)
                return f"\\begin{{bmatrix}} {matrix_content} \\end{{bmatrix}}"

            elif node[0] == 'BRACKETS':
                if isinstance(node[1], list):
                    elements_str = ', '.join(to_latex_str(element) for element in node[1])
                else:
                    elements_str = to_latex_str(node[1])
                return f"[{elements_str}]"

            elif node[0] == 'DELIMITED':
                left_delim_str = to_latex_str(node[1])
                content_str = to_latex_str(node[2])
                right_delim_str = to_latex_str(node[3])
                return f"\\left{left_delim_str} {content_str} \\right{right_delim_str}"

            elif node[0] == 'DIFFERENTIAL':
                var_str = to_latex_str(node[1])
                return f"d{var_str}"

            elif node[0] == 'GEQ':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} \\geq {right_str}"

            elif node[0] == 'LEQ':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} \\leq {right_str}"

            elif node[0] == 'GT':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} \\gt {right_str}"

            elif node[0] == 'LT':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} \\lt {right_str}"

            elif node[0] == 'NEQ':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} \\neq {right_str}"

            elif node[0] == 'IN':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} \\in {right_str}"

            elif node[0] == 'TO':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} \\to {right_str}"

            elif node[0] == 'RIGHTARROW':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} \\rightarrow {right_str}"

            elif node[0] == 'PARALLEL':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} \\parallel {right_str}"

            elif node[0] == 'FUNCTION_CALL':
                func_str = to_latex_str(node[1])
                arg_str = to_latex_str(node[2])
                return f"{func_str}({arg_str})"

            elif node[0] == 'COLON':
                left_str = to_latex_str(node[1])
                right_str = to_latex_str(node[2])
                return f"{left_str} : {right_str}"

            # Single symbol nodes
            elif len(node) == 1 and node[0] in SYMBOL_MAP:
                return SYMBOL_MAP[node[0]]

        elif isinstance(node, str):
            if node in GREEK_MAP:
                return GREEK_MAP[node]
            return node

        return str(node)

    tree, _ = parse_serialized(serialized_tokens, 0)
    return to_latex_str(tree)

def clean_latex(latex: str) -> str:
    """
    Clean LaTeX string by normalizing spacing according to LaTeX conventions.
    """
    # First fix malformed quantifiers
    latex = re.sub(r'\\forall([a-zA-Z])', r'\\forall \1', latex)
    latex = re.sub(r'\\exists([a-zA-Z])', r'\\exists \1', latex)
    latex = re.sub(r'\\nexists([a-zA-Z])', r'\\nexists \1', latex)

    # Remove unnecessary braces around single characters or simple numbers
    latex = re.sub(r'\{([a-zA-Z0-9])\}', r'\1', latex)

    # Normalize spaces around binary operators - remove existing spaces first, then add correct spacing
    binary_ops = [
        r'\+', r'=', r'-',
        r'\\cdot', r'\\times', r'\\div', r'\\pm', r'\\mp',
        r'\\geq', r'\\leq', r'\\gt', r'\\lt', r'\\neq', r'\\approx',
        r'\\equiv', r'\\sim', r'\\simeq', r'\\cong',
        r'\\in', r'\\notin', r'\\subset', r'\\supset', r'\\subseteq', r'\\supseteq',
        r'\\to', r'\\rightarrow', r'\\leftarrow', r'\\leftrightarrow',
        r'\\Rightarrow', r'\\Leftarrow', r'\\Leftrightarrow',
        r'\\parallel', r'\\perp'
    ]

    for op in binary_ops:
        # Remove any existing spaces around the operator, then add proper spacing
        latex = re.sub(f'\\s*({op})\\s*', r' \1 ', latex)

    # Add space after functions before their arguments
    functions = [
        r'\\sin', r'\\cos', r'\\tan', r'\\sec', r'\\csc', r'\\cot',
        r'\\arcsin', r'\\arccos', r'\\arctan', r'\\sinh', r'\\cosh', r'\\tanh',
        r'\\log', r'\\ln', r'\\lg', r'\\exp',
        r'\\lim', r'\\sup', r'\\inf', r'\\max', r'\\min',
        r'\\det', r'\\dim', r'\\ker', r'\\deg', r'\\gcd'
    ]

    for func in functions:
        latex = re.sub(f'({func})(?![_^{{])', r'\1 ', latex)

    # Add space before and after text operators
    text_ops = [r'\\text\{[^}]+\}', r'\\mathrm\{[^}]+\}', r'\\mathit\{[^}]+\}']
    for op in text_ops:
        latex = re.sub(f'({op})', r' \1 ', latex)

    # Add space after large operators when not followed by limits
    large_ops = [r'\\sum', r'\\prod', r'\\int', r'\\oint', r'\\iint', r'\\iiint']
    for op in large_ops:
        latex = re.sub(f'({op})(?![_^])', r'\1 ', latex)

    # Add space after quantifiers
    latex = re.sub(r'(\\forall)(?!\w)', r'\1 ', latex)
    latex = re.sub(r'(\\exists)(?!\w)', r'\1 ', latex)
    latex = re.sub(r'(\\nexists)(?!\w)', r'\1 ', latex)

    # Add space around colons
    latex = re.sub(r'\s*:\s*', r' : ', latex)

    # Add space after commas
    latex = re.sub(r',(?![^{]*}[^{]*\\end)', r', ', latex)

    # Remove spaces around matrix separators
    latex = re.sub(r'\s*&\s*', '&', latex)
    latex = re.sub(r'\s*\\\\\s*', r'\\\\', latex)

    # Remove spaces inside braces for single characters/simple expressions
    latex = re.sub(r'\{\s*([^{}]*?)\s*\}', r'{\1}', latex)

    # Remove spaces around ^ and _ for superscripts/subscripts
    latex = re.sub(r'\s*\^\s*', '^', latex)
    latex = re.sub(r'\s*_\s*', '_', latex)

    # Remove spaces inside \left and \right delimiters
    latex = re.sub(r'\\left\s*([(\[|])', r'\\left\1', latex)
    latex = re.sub(r'\\right\s*([)\]|])', r'\\right\1', latex)

    # Remove spaces around parentheses, brackets, and braces in general
    latex = re.sub(r'\s*([()[\]{}])\s*', r'\1', latex)

    # Clean up multiple spaces
    latex = re.sub(r'\s{2,}', ' ', latex)

    # Remove leading/trailing spaces
    latex = latex.strip()

    # Fix unary minus (remove space after minus at start or after operators)
    latex = re.sub(r'(^|[=+\-*/(<>]\s*)-\s+', r'\1-', latex)

    # Remove space before punctuation
    latex = re.sub(r'\s+([,.;!?])', r'\1', latex)

    return latex

if __name__ == '__main__':
    # Test the enhanced functionality
    test_cases = [
        r'x_i'
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

    print(clean_latex(r'x+y     =x^{x^2}'))
