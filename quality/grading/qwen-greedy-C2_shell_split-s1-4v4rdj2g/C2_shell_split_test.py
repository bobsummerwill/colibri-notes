"""Hidden tests for C2 shell_split. Expected values equal shlex.split(s) (posix, no comments).
The shlex module is blocked before the solution is imported (a solution that imports it fails
its tests; grade.py also scores it 0 by a static check)."""
import sys

sys.path.insert(0, ".")
sys.modules["shlex"] = None  # `import shlex` in the solution raises ImportError
from _runner import raises, run  # noqa: E402

try:
    from solution import shell_split
except BaseException as e:
    def shell_split(*a, _e=e):
        raise RuntimeError(f"import failed: {_e!r}")

CASES = [
    ("simple", "ls -la /tmp", ["ls", "-la", "/tmp"]),
    ("runs_of_ws", "  a \t b\n\nc  ", ["a", "b", "c"]),
    ("empty_input", "", []),
    ("only_ws", " \t\n ", []),
    ("single_quotes", "echo 'hello   world'", ["echo", "hello   world"]),
    ("single_quote_backslash_literal", r"printf 'a\nb\'", ["printf", "a\\nb\\"]),
    ("double_quotes", 'say "two words"', ["say", "two words"]),
    ("dq_escaped_quote_and_backslash", r'"a\"b\\c"', ['a"b\\c']),
    ("dq_other_backslash_kept", r'"a\$b\nc"', ["a\\$b\\nc"]),
    ("adjacent_parts_concatenate", "a'b c'\"d\"e", ["ab cde"]),
    ("empty_quotes_alone", "a '' b \"\"", ["a", "", "b", ""]),
    ("empty_quotes_attached", "x''y", ["xy"]),
    ("backslash_space", r"a\ b c", ["a b", "c"]),
    ("backslash_quote_outside", r"\'x\" \\", ["'x\"", "\\"]),
    ("backslash_newline_outside_kept", "a\\\nb", ["a\nb"]),
    ("special_chars_ordinary", "a;b | c&&d > e $HOME #x `y` (z)", ["a;b", "|", "c&&d", ">", "e", "$HOME", "#x", "`y`", "(z)"]),
    ("option_with_quoted_value", '--opt="hello world" -x', ["--opt=hello world", "-x"]),
    ("quote_inside_other_quote", "\"it's\" 'say \"hi\"'", ["it's", 'say "hi"']),
    ("unicode", "café 'naïve x' \"日本\"", ["café", "naïve x", "日本"]),
    ("carriage_return_ws", "a\r\nb", ["a", "b"]),
]

ERRORS = [
    ("unterminated_single", "echo 'oops"),
    ("unterminated_double", 'echo "oops'),
    ("trailing_backslash", "echo \\"),
    ("dq_ends_with_escaped_quote", '"abc\\"'),
]


def _mk(s, want):
    def t():
        got = shell_split(s)
        assert got == want, f"{s!r}: got {got!r}, want {want!r}"
    return t


def _mk_err(s):
    def t():
        raises(ValueError, shell_split, s)
    return t


TESTS = [(f"case_{n}", _mk(s, w)) for n, s, w in CASES] + \
        [(f"error_{n}", _mk_err(s)) for n, s in ERRORS]
# v2: no "no_shlex" test any more. It gave 1/25 to a solution that imports shlex lazily inside the
# function; grade.py now finds any shlex import with ast.parse and scores the task 0 instead.

if __name__ == "__main__":
    run(TESTS)
