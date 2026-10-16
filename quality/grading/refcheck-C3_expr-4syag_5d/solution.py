"""Reference solution for C3 (evaluate). Not shown to the models."""


def _tokens(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c in " \t\r\n":
            i += 1
        elif c.isdigit() and c in "0123456789":
            j = i
            while j < len(s) and s[j] in "0123456789":
                j += 1
            out.append(("num", int(s[i:j]))); i = j
        elif s.startswith("**", i):
            out.append(("op", "**")); i += 2
        elif c in "+-*/%()":
            out.append(("op", c)); i += 1
        else:
            raise ValueError(f"bad character {c!r}")
    return out


def _tdiv(a, b):
    if b == 0:
        raise ZeroDivisionError("division by zero")
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def evaluate(expr):
    toks = _tokens(expr)
    pos = 0

    def peek():
        return toks[pos][1] if pos < len(toks) and toks[pos][0] == "op" else None

    def take():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def expr_():
        v = term()
        while peek() in ("+", "-"):
            op = take()[1]
            r = term()
            v = v + r if op == "+" else v - r
        return v

    def term():
        v = unary()
        while peek() in ("*", "/", "%"):
            op = take()[1]
            r = unary()
            if op == "*":
                v = v * r
            elif op == "/":
                v = _tdiv(v, r)
            else:
                v = v - r * _tdiv(v, r)
        return v

    def unary():
        if peek() in ("+", "-"):
            op = take()[1]
            v = unary()
            return -v if op == "-" else v
        return power()

    def power():
        base = atom()
        if peek() == "**":
            take()
            e = unary()
            if e < 0:
                raise ValueError("negative exponent")
            return base ** e
        return base

    def atom():
        if pos >= len(toks):
            raise ValueError("unexpected end")
        kind, val = take()
        if kind == "num":
            return val
        if val == "(":
            v = expr_()
            if peek() != ")":
                raise ValueError("missing )")
            take()
            return v
        raise ValueError(f"unexpected {val!r}")

    if not toks:
        raise ValueError("empty")
    v = expr_()
    if pos != len(toks):
        raise ValueError("trailing tokens")
    return v
