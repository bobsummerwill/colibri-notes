import re


def evaluate(expr: str) -> int:
    """Evaluate an integer arithmetic expression without eval/exec/compile/ast."""
    
    # Tokenizer
    tokens = []
    i = 0
    n = len(expr)
    
    while i < n:
        c = expr[i]
        
        # Skip whitespace
        if c.isspace():
            i += 1
            continue
        
        # Integer literal
        if c.isdigit():
            j = i
            while j < n and expr[j].isdigit():
                j += 1
            tokens.append(('NUM', int(expr[i:j])))
            i = j
            continue
        
        # Operators
        if c == '+':
            tokens.append(('ADD', '+'))
            i += 1
            continue
        if c == '-':
            tokens.append(('SUB', '-'))
            i += 1
            continue
        if c == '*':
            if i + 1 < n and expr[i + 1] == '*':
                tokens.append(('POW', '**'))
                i += 2
            else:
                tokens.append(('MUL', '*'))
                i += 1
            continue
        if c == '/':
            tokens.append(('DIV', '/'))
            i += 1
            continue
        if c == '%':
            tokens.append(('MOD', '%'))
            i += 1
            continue
        if c == '(':
            tokens.append(('LPAREN', '('))
            i += 1
            continue
        if c == ')':
            tokens.append(('RPAREN', ')'))
            i += 1
            continue
        
        # Unknown character
        raise ValueError(f"Unknown character: {c!r}")
    
    if not tokens:
        raise ValueError("Empty input")
    
    # Parser using recursive descent
    # Grammar:
    # expr    := term (('+' | '-') term)*
    # term    := unary (('*' | '/' | '%') unary)*
    # unary   := ('+' | '-')* power  (unary + and - prefix, may repeat)
    #           But note: -2**2 is -(2**2), so unary minus binds looser than **
    #           Actually, in Python: -2**2 = -(2**2) = -4
    #           So the grammar should be:
    #           unary := ('+' | '-')* power
    #           But wait, that would make --3 work, and -2**2 = -((2)**2)? 
    #           Let me think again.
    #           
    #           In Python, the precedence is:
    #           ** is highest, then unary +-, then *, /, %, then +, -
    #           So -2**2 parses as -(2**2) = -4
    #           And 2**-3 would be 2**(-3) which raises ValueError for negative exponent
    #           
    #           So the grammar:
    #           power   := unary_or_power ('**' power)?
    #           Actually, ** is right-associative.
    #           
    #           Let me restructure:
    #           expr    := term (('+' | '-') term)*
    #           term    := factor (('*' | '/' | '%') factor)*
    #           factor  := ('+' | '-')* power
    #           power   := atom ('**' factor)?   # right-associative, and exponent can have unary ops
    #           atom    := NUM | '(' expr ')'
    #           
    #           Wait, but -2**2 should be -(2**2). With factor := ('+'|'-')* power, and power := atom ('**' factor)?
    #           For -2**2: factor sees '-', then calls power. power sees atom=2, then '**', then factor.
    #           factor for the exponent: no unary, power: atom=2. So power = 2**2 = 4.
    #           Then factor = -4. That gives -4. Correct!
    #           
    #           For 2**-3: power sees atom=2, then '**', then factor. factor sees '-', then power: atom=3.
    #           So factor = -3. power = 2**(-3) -> ValueError. Correct!
    #           
    #           For 2**3**2: power sees atom=2, then '**', then factor. factor: no unary, power: atom=3, then '**', factor: atom=2.
    #           So inner power = 3**2 = 9. Outer power = 2**9 = 512. Correct!
    
    pos = 0
    
    def parse_expr():
        nonlocal pos
        val = parse_term()
        while pos < len(tokens) and tokens[pos][0] in ('ADD', 'SUB'):
            op = tokens[pos][0]
            pos += 1
            right = parse_term()
            if op == 'ADD':
                val += right
            else:
                val -= right
        return val
    
    def parse_term():
        nonlocal pos
        val = parse_factor()
        while pos < len(tokens) and tokens[pos][0] in ('MUL', 'DIV', 'MOD'):
            op = tokens[pos][0]
            pos += 1
            right = parse_factor()
            if op == 'MUL':
                val *= right
            elif op == 'DIV':
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                # Truncate toward zero
                result = abs(val) // abs(right)
                if (val < 0) != (right < 0):
                    result = -result
                val = result
            else:  # MOD
                if right == 0:
                    raise ZeroDivisionError("Modulo by zero")
                # Sign follows dividend
                result = abs(val) % abs(right)
                if val < 0:
                    result = -result
                val = result
        return val
    
    def parse_factor():
        nonlocal pos
        # Unary + and -
        sign = 1
        while pos < len(tokens) and tokens[pos][0] in ('ADD', 'SUB'):
            if tokens[pos][0] == 'SUB':
                sign *= -1
            pos += 1
        val = parse_power()
        return sign * val
    
    def parse_power():
        nonlocal pos
        base = parse_atom()
        if pos < len(tokens) and tokens[pos][0] == 'POW':
            pos += 1
            # Exponent is a factor (can have unary ops)
            exponent = parse_factor()
            if exponent < 0:
                raise ValueError("Negative exponent not allowed for integer arithmetic")
            result = base ** exponent
            # Note: 0**0 = 1 in Python, which is what we want
            return result
        return base
    
    def parse_atom():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Unexpected end of expression")
        
        token_type, token_val = tokens[pos]
        
        if token_type == 'NUM':
            pos += 1
            return token_val
        
        if token_type == 'LPAREN':
            pos += 1
            val = parse_expr()
            if pos >= len(tokens) or tokens[pos][0] != 'RPAREN':
                raise ValueError("Missing closing parenthesis")
            pos += 1
            return val
        
        # Unexpected token
        raise ValueError(f"Unexpected token: {token_val!r}")
    
    result = parse_expr()
    
    if pos != len(tokens):
        raise ValueError(f"Unexpected token after expression: {tokens[pos][1]!r}")
    
    return result
