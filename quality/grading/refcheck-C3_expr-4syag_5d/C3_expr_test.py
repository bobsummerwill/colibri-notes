"""Hidden tests for C3 evaluate (integer expression evaluator with C-style / and %)."""
import sys

sys.path.insert(0, ".")
from _runner import raises, run  # noqa: E402

try:
    from solution import evaluate
except BaseException as e:
    def evaluate(*a, _e=e):
        raise RuntimeError(f"import failed: {_e!r}")

VALUES = [
    ("int", "42", 42),
    ("leading_zeros", "007 + 1", 8),
    ("precedence", "2+3*4**2", 50),
    ("left_assoc_sub", "10-4-3", 3),
    ("left_assoc_div", "100/7/2", 7),
    ("left_assoc_mixed", "2*3%4", 2),
    ("parens", "(1+2)*(3+4)", 21),
    ("pow_right_assoc", "2**3**2", 512),
    ("unary_below_pow", "-2**2", -4),
    ("paren_neg_pow", "(-2)**2", 4),
    ("unary_chain", "-+-3", 3),
    ("double_minus", "2--3", 5),
    ("mul_neg", "2*-3", -6),
    ("unary_plus", "+5", 5),
    ("neg_paren", "- (2+3)", -5),
    ("trunc_div_neg_divisor", "7/-2", -3),
    ("trunc_div_neg_dividend", "-7/2", -3),
    ("trunc_div_paren", "(-8)/3", -2),
    ("mod_neg_divisor", "7%-2", 1),
    ("mod_neg_dividend", "-7%2", -1),
    ("mod_both_neg", "(-7)%(-2)", -1),
    ("mod_paren", "(-8)%3", -2),
    ("zero_pow_zero", "0**0", 1),
    ("big_int", "2**100 + 1", 2 ** 100 + 1),
    ("pow_of_unary_exponent", "2**+3", 8),
    ("whitespace", " \t1 +\n2 ", 3),
    ("nested", "((((3))))*-(-(2))", 6),
    ("div_mod_identity", "(-17/5)*5 + (-17%5)", -17),
]

ERRORS = [
    ("empty", ""), ("blank", "   "), ("trailing_op", "1+"), ("leading_binary", "*2"),
    ("unclosed", "(1+2"), ("unopened", "1+2)"), ("empty_parens", "()"), ("adjacent_numbers", "3 4"),
    ("double_binary", "1 + * 2"), ("negative_exponent", "2 ** -1"), ("float", "1.5"), ("caret", "2 ^ 3"),
    ("letters", "abc"), ("spaced_pow", "2 * * 3"), ("neg_exp_after_unary", "-2**-2"),
]

ZERO = [("div_zero", "1/0"), ("mod_zero", "5%0"), ("div_zero_expr", "1/(2-2)")]


def _v(s, want):
    def t():
        got = evaluate(s)
        assert type(got) is int and got == want, f"{s!r}: got {got!r}, want {want!r}"
    return t


def _e(s, exc):
    def t():
        raises(exc, evaluate, s)
    return t


TESTS = [(f"value_{n}", _v(s, w)) for n, s, w in VALUES] + \
        [(f"error_{n}", _e(s, ValueError)) for n, s in ERRORS] + \
        [(f"zero_{n}", _e(s, ZeroDivisionError)) for n, s in ZERO]

if __name__ == "__main__":
    run(TESTS)
