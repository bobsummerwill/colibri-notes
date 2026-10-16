def shell_split(s: str) -> list[str]:
    """Split a command line into words using POSIX-like quoting rules."""
    words = []
    current_word = []
    in_word = False  # whether we're currently building a word (have seen any non-whitespace or quoted content)
    
    i = 0
    n = len(s)
    
    while i < n:
        c = s[i]
        
        # Check for whitespace outside quotes
        if c in (' ', '\t', '\r', '\n'):
            if in_word:
                words.append(''.join(current_word))
                current_word = []
                in_word = False
            i += 1
            continue
        
        # Backslash outside quotes
        if c == '\\':
            if i + 1 >= n:
                raise ValueError("backslash at end of input")
            i += 1
            current_word.append(s[i])
            in_word = True
            i += 1
            continue
        
        # Single quote
        if c == "'":
            in_word = True
            i += 1
            while i < n:
                if s[i] == "'":
                    i += 1
                    break
                current_word.append(s[i])
                i += 1
            else:
                raise ValueError("unterminated single quote")
            continue
        
        # Double quote
        if c == '"':
            in_word = True
            i += 1
            while i < n:
                if s[i] == '"':
                    i += 1
                    break
                if s[i] == '\\':
                    if i + 1 >= n:
                        # Backslash at end of input inside double quotes
                        # The spec says raise for backslash that is last character of input
                        raise ValueError("backslash at end of input")
                    next_c = s[i + 1]
                    if next_c == '"' or next_c == '\\':
                        current_word.append(next_c)
                        i += 2
                    else:
                        current_word.append('\\')
                        current_word.append(next_c)
                        i += 2
                else:
                    current_word.append(s[i])
                    i += 1
            else:
                raise ValueError("unterminated double quote")
            continue
        
        # Regular character
        current_word.append(c)
        in_word = True
        i += 1
    
    # If we ended up with a word being built, append it
    if in_word:
        words.append(''.join(current_word))
    
    return words
