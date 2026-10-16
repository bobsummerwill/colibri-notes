def shell_split(s: str) -> list[str]:
    words = []
    current = []
    in_word = False
    i = 0
    n = len(s)
    
    while i < n:
        c = s[i]
        
        # Whitespace outside quotes separates words
        if c in ' \t\r\n':
            if in_word:
                words.append(''.join(current))
                current = []
                in_word = False
            i += 1
            continue
        
        # Backslash outside quotes: escape next character
        if c == '\\':
            if i + 1 >= n:
                raise ValueError("Backslash at end of input")
            next_c = s[i + 1]
            current.append(next_c)
            in_word = True
            i += 2
            continue
        
        # Single quotes: everything up to next single quote is literal
        if c == "'":
            i += 1
            in_word = True
            while i < n:
                if s[i] == "'":
                    i += 1
                    break
                current.append(s[i])
                i += 1
            else:
                raise ValueError("Unterminated single quote")
            continue
        
        # Double quotes: everything up to closing double quote, with backslash rules
        if c == '"':
            i += 1
            in_word = True
            while i < n:
                if s[i] == '"':
                    i += 1
                    break
                if s[i] == '\\':
                    if i + 1 >= n:
                        raise ValueError("Backslash at end of input")
                    next_c = s[i + 1]
                    if next_c == '"' or next_c == '\\':
                        current.append(next_c)
                        i += 2
                    else:
                        current.append('\\')
                        current.append(next_c)
                        i += 2
                else:
                    current.append(s[i])
                    i += 1
            else:
                raise ValueError("Unterminated double quote")
            continue
        
        # Ordinary character
        current.append(c)
        in_word = True
        i += 1
    
    if in_word:
        words.append(''.join(current))
    
    return words
