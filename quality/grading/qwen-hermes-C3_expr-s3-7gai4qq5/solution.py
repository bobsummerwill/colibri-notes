def evaluate(expr: str) -> int:
    """Evaluate an integer arithmetic expression without eval/exec/compile/ast."""
    
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
            tokens.append(('NUMBER', int(expr[i:j])))
            i = j
            continue
        if c == '*':
            if i + 1 < n and expr[i + 1] == '*':
                tokens.append(('OP', '**'))
                i += 2
            else:
                tokens.append(('OP', '*'))
                i += 1
            continue
        if c in '+-/%()':
            tokens.append(('OP', c))
            i += 1
            continue
        raise ValueError(f"Unknown character: {c!r}")
    
    if not tokens:
        raise ValueError("Empty input")
    
    # Parser using recursive descent
    # Grammar:
    #   expression  := term (('+' | '-') term)*
    #   term        := unary (('*' | '/' | '%') unary)*
    #   unary       := ('+' | '-')* power
    #   power       := atom ('**' unary)?    # right-associative, exponent is unary
    #   atom        := NUMBER | '(' expression ')'
    
    class Parser:
        def __init__(self, tokens):
            self.tokens = tokens
            self.pos = 0
        
        def peek(self):
            if self.pos < len(self.tokens):
                return self.tokens[self.pos]
            return None
        
        def consume(self, expected_type=None, expected_val=None):
            tok = self.peek()
            if tok is None:
                raise ValueError("Unexpected end of input")
            if expected_type is not None and tok[0] != expected_type:
                raise ValueError(f"Expected {expected_type}, got {tok[0]}: {tok[1]!r}")
            if expected_val is not None and tok[1] != expected_val:
                raise ValueError(f"Expected {expected_val!r}, got {tok[1]!r}")
            self.pos += 1
            return tok
        
        def at_end(self):
            return self.pos >= len(self.tokens)
        
        def parse_expression(self):
            """expression := term (('+' | '-') term)*"""
            result = self.parse_term()
            while not self.at_end():
                tok = self.peek()
                if tok[0] == 'OP' and tok[1] in ('+', '-'):
                    op = tok[1]
                    self.consume()
                    right = self.parse_term()
                    if op == '+':
                        result += right
                    else:
                        result -= right
                else:
                    break
            return result
        
        def parse_term(self):
            """term := unary (('*' | '/' | '%') unary)*"""
            result = self.parse_unary()
            while not self.at_end():
                tok = self.peek()
                if tok[0] == 'OP' and tok[1] in ('*', '/', '%'):
                    op = tok[1]
                    self.consume()
                    right = self.parse_unary()
                    if op == '*':
                        result *= right
                    elif op == '/':
                        if right == 0:
                            raise ZeroDivisionError("division by zero")
                        # Integer division truncating toward zero
                        if (result < 0) != (right < 0):
                            result = -((-result) // right)
                        else:
                            result = result // right
                    else:  # '%'
                        if right == 0:
                            raise ZeroDivisionError("modulo by zero")
                        # Remainder with sign following dividend
                        # (a/b)*b + a%b == a
                        q = result // right if (result < 0) == (right < 0) else -((-result) // right)
                        result = result - q * right
                else:
                    break
            return result
        
        def parse_unary(self):
            """unary := ('+' | '-')* power"""
            sign = 1
            while not self.at_end():
                tok = self.peek()
                if tok[0] == 'OP' and tok[1] in ('+', '-'):
                    op = tok[1]
                    self.consume()
                    if op == '-':
                        sign = -sign
                    # if '+', sign stays the same
                else:
                    break
            result = self.parse_power()
            return sign * result
        
        def parse_power(self):
            """power := atom ('**' unary)?"""
            base = self.parse_atom()
            if not self.at_end():
                tok = self.peek()
                if tok[0] == 'OP' and tok[1] == '**':
                    self.consume()
                    # Exponent is a unary expression (allows 2**+3, 2**-3, etc.)
                    exponent = self.parse_unary()
                    if exponent < 0:
                        raise ValueError("Negative exponent")
                    result = base ** exponent
                    return result
            return base
        
        def parse_atom(self):
            """atom := NUMBER | '(' expression ')'"""
            tok = self.peek()
            if tok is None:
                raise ValueError("Unexpected end of input, expected operand")
            
            if tok[0] == 'NUMBER':
                self.consume()
                return tok[1]
            
            if tok[0] == 'OP' and tok[1] == '(':
                self.consume()
                result = self.parse_expression()
                # Expect closing paren
                tok2 = self.peek()
                if tok2 is None:
                    raise ValueError("Missing closing parenthesis")
                if tok2[0] != 'OP' or tok2[1] != ')':
                    raise ValueError("Missing closing parenthesis")
                self.consume()
                return result
            
            # If we got here, it's an operator where an operand was expected
            raise ValueError(f"Expected operand, got {tok[1]!r}")
    
    parser = Parser(tokens)
    result = parser.parse_expression()
    
    # Ensure all tokens were consumed
    if not parser.at_end():
        raise ValueError(f"Unexpected token: {parser.peek()[1]!r}")
    
    return result
