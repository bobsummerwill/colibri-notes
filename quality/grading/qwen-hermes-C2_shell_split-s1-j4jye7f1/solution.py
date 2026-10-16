def shell_split(s: str) -> list[str]:
    words = []
    current = []
    i = 0
    n = len(s)
    in_word = False  # whether we are currently building a non-empty word (or have seen a quote)
    
    while i < n:
        c = s[i]
        
        if c in ' \t\r\n':
            # Whitespace: if we are in a word, finalize it
            if in_word:
                words.append(''.join(current))
                current = []
                in_word = False
            i += 1
        elif c == '\\':
            # Backslash outside quotes
            if i + 1 >= n:
                raise ValueError("Backslash at end of input")
            next_c = s[i + 1]
            current.append(next_c)
            in_word = True
            i += 2
        elif c == "'":
            # Single quote
            in_word = True
            i += 1
            while i < n:
                if s[i] == "'":
                    i += 1
                    break
                current.append(s[i])
                i += 1
            else:
                raise ValueError("Unterminated single quote")
        elif c == '"':
            # Double quote
            in_word = True
            i += 1
            while i < n:
                if s[i] == '"':
                    i += 1
                    break
                if s[i] == '\\' and i + 1 < n and s[i + 1] in ('"', '\\'):
                    current.append(s[i + 1])
                    i += 2
                else:
                    current.append(s[i])
                    i += 1
            else:
                raise ValueError("Unterminated double quote")
        else:
            # Ordinary character
            current.append(c)
            in_word = True
            i += 1
    
    # If we ended while building a word, finalize it
    if in_word:
        words.append(''.join(current))
    
    return words
