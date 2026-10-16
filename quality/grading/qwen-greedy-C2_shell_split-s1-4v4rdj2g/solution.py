def shell_split(s: str) -> list[str]:
    words = []
    current = []
    in_word = False
    i = 0
    n = len(s)
    
    while i < n:
        c = s[i]
        
        if c in ' \t\r\n':
            if in_word:
                words.append(''.join(current))
                current = []
                in_word = False
            i += 1
        elif c == '\\':
            if i + 1 >= n:
                raise ValueError("backslash at end of input")
            next_c = s[i + 1]
            current.append(next_c)
            in_word = True
            i += 2
        elif c == "'":
            # Single quote: everything up to next single quote is literal
            j = i + 1
            while j < n:
                if s[j] == "'":
                    break
                current.append(s[j])
                j += 1
            if j >= n:
                raise ValueError("unterminated single quote")
            in_word = True
            i = j + 1
        elif c == '"':
            # Double quote: everything up to closing double quote, with backslash rules
            j = i + 1
            while j < n:
                if s[j] == '"':
                    break
                if s[j] == '\\':
                    if j + 1 < n and s[j + 1] in ('"', '\\'):
                        current.append(s[j + 1])
                        j += 2
                    else:
                        current.append(s[j])
                        j += 1
                else:
                    current.append(s[j])
                    j += 1
            if j >= n:
                raise ValueError("unterminated double quote")
            in_word = True
            i = j + 1
        else:
            current.append(c)
            in_word = True
            i += 1
    
    if in_word:
        words.append(''.join(current))
    
    return words
