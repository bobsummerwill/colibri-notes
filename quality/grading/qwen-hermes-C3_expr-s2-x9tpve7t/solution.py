def evaluate(expr: str) -> int:
    """
    Evaluate an integer arithmetic expression without using eval, exec, compile, or ast.
    
    Tokens: integer literals, + - * / % ** and parentheses.
    Precedence (high to low): ** (right-assoc), unary +/-, * / % (left-assoc), + - (left-assoc)
    / is truncating division toward zero, % follows dividend sign.
    0**0 = 1. Negative exponent raises ValueError. Division/remainder by zero raises ZeroDivisionError.
    Malformed input raises ValueError.
    """
    
    # Tokenizer
    tokens = []
    i = 0
    n = len(expr)
    
    while i < n:
        ch = expr[i]
        
        # Skip whitespace
        if ch in ' \t\n\r':
            i += 1
            continue
        
        # Number
        if ch.isdigit():
            j = i
            while j < n and expr[j].isdigit():
                j += 1
            num_str = expr[i:j]
            tokens.append(('NUM', int(num_str)))
            i = j
            continue
        
        # Operators
        if ch == '+':
            tokens.append(('OP', '+'))
            i += 1
            continue
        if ch == '-':
            tokens.append(('OP', '-'))
            i += 1
            continue
        if ch == '*':
            if i + 1 < n and expr[i + 1] == '*':
                tokens.append(('OP', '**'))
                i += 2
            else:
                tokens.append(('OP', '*'))
                i += 1
            continue
        if ch == '/':
            tokens.append(('OP', '/'))
            i += 1
            continue
        if ch == '%':
            tokens.append(('OP', '%'))
            i += 1
            continue
        if ch == '(':
            tokens.append(('LPAREN', '('))
            i += 1
            continue
        if ch == ')':
            tokens.append(('RPAREN', ')'))
            i += 1
            continue
        
        # Unknown character
        raise ValueError(f"Unknown character: {ch}")
    
    if not tokens:
        raise ValueError("Empty input")
    
    # Parser
    pos = 0
    
    def peek():
        nonlocal pos
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
        """Parse + and - (left-associative)"""
        left = parse_term()
        while True:
            tok = peek()
            if tok is None:
                break
            if tok[0] == 'OP' and tok[1] in ('+', '-'):
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
        """Parse * / % (left-associative)"""
        left = parse_factor()
        while True:
            tok = peek()
            if tok is None:
                break
            if tok[0] == 'OP' and tok[1] in ('*', '/', '%'):
                consume()
                right = parse_factor()
                if tok[1] == '*':
                    left = left * right
                elif tok[1] == '/':
                    if right == 0:
                        raise ZeroDivisionError("division by zero")
                    # Truncating division toward zero
                    q = abs(left) // abs(right)
                    if (left < 0) != (right < 0):
                        q = -q
                    left = q
                else:  # '%'
                    if right == 0:
                        raise ZeroDivisionError("remainder by zero")
                    # Python's % already gives sign of dividend, which is what we want
                    left = left % right
            else:
                break
        return left
    
    def parse_factor():
        """Parse unary + and - (prefix, may repeat)"""
        tok = peek()
        if tok is None:
            raise ValueError("Missing operand")
        if tok[0] == 'OP' and tok[1] in ('+', '-'):
            consume()
            operand = parse_factor()
            if tok[1] == '+':
                return operand
            else:
                return -operand
        return parse_power()
    
    def parse_power():
        """Parse ** (right-associative)"""
        base = parse_atom()
        tok = peek()
        if tok is not None and tok[0] == 'OP' and tok[1] == '**':
            consume()
            # Right-associative: the exponent is parsed as a factor (which handles unary)
            # But we need the exponent to be a "factor" level so that -2**2 works correctly.
            # Actually, per the spec: "a unary minus on the left of ** binds more loosely than **,"
            # so -2**2 = -(2**2) = -4. This is handled by parse_factor calling parse_power for the base.
            # For the right side of **, we should parse a factor (to allow unary on exponent like 2**+3).
            # But to maintain right-associativity, we parse the exponent as a factor that can itself contain **.
            # Wait: 2**3**2 should be 2**(3**2) = 512. So the exponent should be parsed by parse_power.
            # But the spec says "the exponent may itself start with a unary operator, so 2**+3 is 8."
            # So we need to parse unary operators on the exponent. Let's create a helper.
            exponent = parse_unary_for_exponent()
            if exponent < 0:
                raise ValueError("Negative exponent")
            left = base
            right = exponent
            return left ** right
        return base
    
    def parse_unary_for_exponent():
        """Parse unary +/- then power (right-associative) for the exponent of **"""
        tok = peek()
        if tok is not None and tok[0] == 'OP' and tok[1] in ('+', '-'):
            consume()
            val = parse_unary_for_exponent()
            if tok[1] == '+':
                return val
            else:
                return -val
        return parse_power()
    
    def parse_atom():
        """Parse numbers and parenthesized expressions"""
        tok = peek()
        if tok is None:
            raise ValueError("Missing operand")
        
        if tok[0] == 'NUM':
            consume()
            return tok[1]
        
        if tok[0] == 'LPAREN':
            consume()
            val = parse_expression()
            # Expect closing paren
            tok2 = peek()
            if tok2 is None or tok2[0] != 'RPAREN':
                raise ValueError("Unbalanced parentheses")
            consume()
            return val
        
        raise ValueError(f"Unexpected token: {tok}")
    
    result = parse_expression()
    
    # Check that we consumed all tokens
    if pos != len(tokens):
        raise ValueError("Unexpected token after end of expression")
    
    return result
