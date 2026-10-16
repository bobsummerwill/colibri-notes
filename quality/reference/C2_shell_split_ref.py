"""Reference solution for C2 (shell_split). Not shown to the models."""


def shell_split(s):
    words, cur, have = [], [], False
    i, n = 0, len(s)
    ws = " \t\r\n"
    while i < n:
        c = s[i]
        if c in ws:
            if have:
                words.append("".join(cur))
                cur, have = [], False
            i += 1
        elif c == "\\":
            if i + 1 >= n:
                raise ValueError("trailing backslash")
            cur.append(s[i + 1]); have = True; i += 2
        elif c == "'":
            j = s.find("'", i + 1)
            if j < 0:
                raise ValueError("unterminated single quote")
            cur.append(s[i + 1:j]); have = True; i = j + 1
        elif c == '"':
            i += 1; have = True
            while True:
                if i >= n:
                    raise ValueError("unterminated double quote")
                c = s[i]
                if c == '"':
                    i += 1; break
                if c == "\\" and i + 1 < n and s[i + 1] in '"\\':
                    cur.append(s[i + 1]); i += 2
                else:
                    cur.append(c); i += 1
        else:
            cur.append(c); have = True; i += 1
    if have:
        words.append("".join(cur))
    return words
