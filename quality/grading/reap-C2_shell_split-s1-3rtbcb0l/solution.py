def shell_split(s: str) -> list[str]:
    words = []
    current = []
    in_word = False
    i = 0
    n = len(s)
    
    while i < n:
        c = s[i]
        
        # Outside quotes
        if c in ' \t\r\n':
            if in_word:
                words.append(''.join(current))
                current = []
                in_word = False
            i += 1
            continue
        
        if c == '\\':
            if i + 1 >= n:
                raise ValueError("Trailing backslash")
            current.append(s[i+1])
            in_word = True
            i += 2
            continue
        
        if c == "'":
            in_word = True
            i += 1
            # Find closing single quote
            start = i
            while i < n and s[i] != "'":
                i += 1
            if i >= n:
                raise ValueError("Unterminated single quote")
            current.append(s[start:i])
            i += 1  # skip closing quote
            continue
        
        if c == '"':
            in_word = True
            i += 1
            # Process double-quoted string
            while i < n:
                ch = s[i]
                if ch == '"':
                    i += 1
                    break
                elif ch == '\\':
                    if i + 1 >= n:
                        raise ValueError("Trailing backslash")
                    nxt = s[i+1]
                    if nxt == '"' or nxt == '\\':
                        current.append(nxt)
                        i += 2
                    else:
                        current.append('\\')
                        current.append(nxt)
                        i += 2
                else:
                    current.append(ch)
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
