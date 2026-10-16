def evaluate(expr: str) -> int:
    # Tokenizer
    tokens = []
    i = 0
    n = len(expr)
    while i < n:
        c = expr[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit():
            j = i
            while j < n and expr[j].isdigit():
                j += 1
            tokens.append(('num', int(expr[i:j])))
            i = j
            continue
        if c == '*' and i + 1 < n and expr[i + 1] == '*':
            tokens.append(('op', '**'))
            i += 2
            continue
        if c in '+-*/%':
            tokens.append(('op', c))
            i += 1
            continue
        if c == '(':
            tokens.append(('lparen', '('))
            i += 1
            continue
        if c == ')':
            tokens.append(('rparen', ')'))
            i += 1
            continue
        raise ValueError(f"Unknown character: {c}")
    
    if not tokens:
        raise ValueError("Empty input")
    
    # Parser using recursive descent
    # Precedence (low to high):
    # 1. + - (binary, left-assoc)
    # 2. * / % (left-assoc)
    # 3. unary + - (prefix)
    # 4. ** (right-assoc)
    # 5. atoms (numbers, parenthesized expressions)
    
    pos = 0
    
    def peek():
        if pos < len(tokens):
            return tokens[pos]
        return None
    
    def consume():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end of input")
        tok = tokens[pos]
        pos += 1
        return tok
    
    def parse_expression():
        # Handles + and - (binary, left-associative)
        left = parse_term()
        while True:
            tok = peek()
            if tok is None:
                break
            if tok[0] == 'op' and tok[1] in ('+', '-'):
                consume()
                right = parse_term()
                if tok[1] == '+':
                    left = left + right
                else:
                    left = left - right
            else:
                break
        return left
    
    def parse_term():
        # Handles * / % (left-associative)
        left = parse_unary()
        while True:
            tok = peek()
            if tok is None:
                break
            if tok[0] == 'op' and tok[1] in ('*', '/', '%'):
                consume()
                right = parse_unary()
                if tok[1] == '*':
                    left = left * right
                elif tok[1] == '/':
                    if right == 0:
                        raise ZeroDivisionError("division by zero")
                    # Truncate toward zero
                    q = abs(left) // abs(right)
                    if (left < 0) != (right < 0):
                        q = -q
                    left = q
                else:  # %
                    if right == 0:
                        raise ZeroDivisionError("modulo by zero")
                    # Sign follows dividend
                    r = abs(left) % abs(right)
                    if left < 0:
                        r = -r
                    left = r
            else:
                break
        return left
    
    def parse_unary():
        # Handles unary + and - (prefix, may repeat)
        tok = peek()
        if tok is not None and tok[0] == 'op' and tok[1] in ('+', '-'):
            consume()
            operand = parse_unary()
            if tok[1] == '-':
                return -operand
            else:
                return operand
        return parse_power()
    
    def parse_power():
        # Handles ** (right-associative)
        # In Python, -2**2 is -(2**2) = -4
        # So unary minus binds more loosely than **
        # This means: parse the base (which is an atom or parenthesized expr),
        # then if ** follows, parse the exponent (which can include unary operators)
        base = parse_atom()
        tok = peek()
        if tok is not None and tok[0] == 'op' and tok[1] == '**':
            consume()
            # Exponent can have unary operators, so parse_unary for the exponent
            exponent = parse_unary()
            if exponent < 0:
                raise ValueError("Negative exponent")
            if base == 0 and exponent == 0:
                return 1
            return base ** exponent
        return base
    
    def parse_atom():
        tok = peek()
        if tok is None:
            raise ValueError("Missing operand")
        if tok[0] == 'num':
            consume()
            return tok[1]
        if tok[0] == 'lparen':
            consume()
            val = parse_expression()
            tok2 = peek()
            if tok2 is None or tok2[0] != 'rparen':
                raise ValueError("Unbalanced parentheses")
            consume()
            return val
        raise ValueError(f"Unexpected token: {tok}")
    
    result = parse_expression()
    
    # Check that we consumed all tokens
    if pos != len(tokens):
        raise ValueError("Unexpected token after expression")
    
    return result
