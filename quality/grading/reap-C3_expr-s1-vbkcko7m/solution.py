import re

def evaluate(expr: str) -> int:
    # Tokenize
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
        if c == '*' and i+1 < n and expr[i+1] == '*':
            tokens.append(('**', '**'))
            i += 2
            continue
        if c in '+-*/%()':
            tokens.append((c, c))
            i += 1
            continue
        raise ValueError("Unknown character")
    
    if not tokens:
        raise ValueError("Empty input")
    
    # Parser with precedence climbing
    pos = 0
    
    def peek():
        if pos < len(tokens):
            return tokens[pos][0]
        return None
    
    def consume(expected=None):
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end")
        tok = tokens[pos]
        if expected is not None and tok[0] != expected:
            raise ValueError(f"Expected {expected}, got {tok[0]}")
        pos += 1
        return tok[1]
    
    def parse_expression():
        return parse_binary(1)
    
    def parse_binary(min_prec):
        # parse unary first
        left = parse_unary()
        while True:
            op = peek()
            prec = 0
            if op in ('+', '-'):
                prec = 1
            elif op in ('*', '/', '%'):
                prec = 2
            elif op == '**':
                prec = 3
            else:
                break
            if prec < min_prec:
                break
            # For **, right-associative: parse with min_prec = prec (not prec+1)
            if op == '**':
                consume(op)
                right = parse_binary(prec)  # right assoc
                left = apply_op(left, op, right)
            else:
                consume(op)
                right = parse_binary(prec + 1)
                left = apply_op(left, op, right)
        return left
    
    def parse_unary():
        # Handle unary + and - (prefix)
        op = peek()
        if op in ('+', '-'):
            consume(op)
            val = parse_unary()
            if op == '-':
                return -val
            return val
        return parse_primary()
    
    def parse_primary():
        tok = peek()
        if tok == 'num':
            return consume('num')
        if tok == '(':
            consume('(')
            val = parse_expression()
            if peek() != ')':
                raise ValueError("Missing closing parenthesis")
            consume(')')
            return val
        if tok == '**':
            # Unary operator before **? Actually ** is binary, but we might have something like -2**2
            # parse_unary handles leading unary, so here we shouldn't see ** as primary
            raise ValueError("Unexpected **")
        raise ValueError("Missing operand")
    
    def apply_op(a, op, b):
        if op == '+':
            return a + b
        if op == '-':
            return a - b
        if op == '*':
            return a * b
        if op == '/':
            if b == 0:
                raise ZeroDivisionError("division by zero")
            # Integer division truncating toward zero
            return int(a / b)  # Python's / gives float, but for large ints may lose precision
            # Better: use // with adjustment for negative
            # Actually Python's // floors, so we need trunc toward zero:
            # q = abs(a) // abs(b)
            # if (a < 0) != (b < 0): q = -q
            # return q
        if op == '%':
            if b == 0:
                raise ZeroDivisionError("modulo by zero")
            # Python's % gives remainder with sign of divisor, we need sign of dividend
            # r = a - (a//b)*b  but // floors, so adjust:
            # q = trunc_div(a,b)
            # r = a - q*b
            # return r
        if op == '**':
            if b < 0:
                raise ValueError("Negative exponent")
            return a ** b
        raise ValueError("Unknown operator")
    
    # We need to fix apply_op for / and % to handle trunc toward zero correctly
    # Let's redefine apply_op properly:
    def apply_op(a, op, b):
        if op == '+':
            return a + b
        if op == '-':
            return a - b
        if op == '*':
            return a * b
        if op == '/':
            if b == 0:
                raise ZeroDivisionError("division by zero")
            # trunc toward zero
            q = abs(a) // abs(b)
            if (a < 0) != (b < 0):
                q = -q
            return q
        if op == '%':
            if b == 0:
                raise ZeroDivisionError("modulo by zero")
            q = abs(a) // abs(b)
            if (a < 0) != (b < 0):
                q = -q
            return a - q * b
        if op == '**':
            if b < 0:
                raise ValueError("Negative exponent")
            return a ** b
        raise ValueError("Unknown operator")
    
    # But apply_op is defined after parse functions, so we need to move it before or use closure
    # Let's restructure: define apply_op before parse functions.
    
    # Actually, let's rewrite the whole thing cleanly.
    
    # I'll rewrite from scratch to avoid confusion.
    
    # The above code has issues with function ordering. Let's do a clean implementation.
    
    # Clean implementation:
    
    def evaluate(expr: str) -> int:
        # Tokenize
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
            if c == '*' and i+1 < n and expr[i+1] == '*':
                tokens.append(('**', '**'))
                i += 2
                continue
            if c in '+-*/%()':
                tokens.append((c, c))
                i += 1
                continue
            raise ValueError("Unknown character")
        
        if not tokens:
            raise ValueError("Empty input")
        
        pos = 0
        
        def peek():
            if pos < len(tokens):
                return tokens[pos][0]
            return None
        
        def consume(expected=None):
            nonlocal pos
            if pos >= len(tokens):
                raise ValueError("Unexpected end")
            tok = tokens[pos]
            if expected is not None and tok[0] != expected:
                raise ValueError(f"Expected {expected}, got {tok[0]}")
            pos += 1
            return tok[1]
        
        def apply_op(a, op, b):
            if op == '+':
                return a + b
            if op == '-':
                return a - b
            if op == '*':
                return a * b
            if op == '/':
                if b == 0:
                    raise ZeroDivisionError("division by zero")
                q = abs(a) // abs(b)
                if (a < 0) != (b < 0):
                    q = -q
                return q
            if op == '%':
                if b == 0:
                    raise ZeroDivisionError("modulo by zero")
                q = abs(a) // abs(b)
                if (a < 0) != (b < 0):
                    q = -q
                return a - q * b
            if op == '**':
                if b < 0:
                    raise ValueError("Negative exponent")
                return a ** b
            raise ValueError("Unknown operator")
        
        def parse_expression():
            return parse_binary(1)
        
        def parse_binary(min_prec):
            left = parse_unary()
            while True:
                op = peek()
                prec = 0
                if op in ('+', '-'):
                    prec = 1
                elif op in ('*', '/', '%'):
                    prec = 2
                elif op == '**':
                    prec = 3
                else:
                    break
                if prec < min_prec:
                    break
                if op == '**':
                    consume(op)
                    right = parse_binary(prec)  # right assoc
                    left = apply_op(left, op, right)
                else:
                    consume(op)
                    right = parse_binary(prec + 1)
                    left = apply_op(left, op, right)
            return left
        
        def parse_unary():
            op = peek()
            if op in ('+', '-'):
                consume(op)
                val = parse_unary()
                if op == '-':
                    return -val
                return val
            return parse_primary()
        
        def parse_primary():
            tok = peek()
            if tok == 'num':
                return consume('num')
            if tok == '(':
                consume('(')
                val = parse_expression()
                if peek() != ')':
                    raise ValueError("Missing closing parenthesis")
                consume(')')
                return val
            raise ValueError("Missing operand")
        
        result = parse_expression()
        if pos != len(tokens):
            raise ValueError("Unexpected token")
        return result
    
    return evaluate(expr)
