"""Hidden tests for C1 next_cron_times. 20 tests."""
import datetime as dt
import sys

sys.path.insert(0, ".")
from _runner import raises, run  # noqa: E402

try:
    from solution import next_cron_times
except BaseException as e:  # import failure -> every test fails
    def next_cron_times(*a, _e=e):
        raise RuntimeError(f"import failed: {_e!r}")

D = dt.datetime


def t_every_15():
    assert next_cron_times("*/15 * * * *", D(2024, 3, 10, 10, 7), 4) == [
        D(2024, 3, 10, 10, 15), D(2024, 3, 10, 10, 30), D(2024, 3, 10, 10, 45), D(2024, 3, 10, 11, 0)]


def t_strictly_after_exact():
    assert next_cron_times("0 * * * *", D(2024, 1, 1, 10, 0), 1) == [D(2024, 1, 1, 11, 0)]


def t_strictly_after_seconds():
    assert next_cron_times("0 * * * *", D(2024, 1, 1, 10, 0, 30), 1) == [D(2024, 1, 1, 11, 0)]


def t_seconds_zeroed():
    r = next_cron_times("* * * * *", D(2024, 1, 1, 0, 0, 59, 999999), 2)
    assert r == [D(2024, 1, 1, 0, 1), D(2024, 1, 1, 0, 2)]
    assert all(x.second == 0 and x.microsecond == 0 for x in r)
    assert isinstance(r, list)


def t_weekdays():
    # 2024-05-03 is a Friday
    assert next_cron_times("30 9 * * 1-5", D(2024, 5, 3, 9, 30), 2) == [
        D(2024, 5, 6, 9, 30), D(2024, 5, 7, 9, 30)]


def t_sunday_7():
    # 2024-06-01 is a Saturday
    assert next_cron_times("0 12 * * 7", D(2024, 6, 1), 2) == [D(2024, 6, 2, 12, 0), D(2024, 6, 9, 12, 0)]


def t_sunday_0():
    assert next_cron_times("0 12 * * 0", D(2024, 6, 1), 1) == [D(2024, 6, 2, 12, 0)]


def t_dow_range_to_7():
    # 2024-06-06 is a Thursday; 5-7 = Fri, Sat, Sun
    assert next_cron_times("0 8 * * 5-7", D(2024, 6, 6), 4) == [
        D(2024, 6, 7, 8, 0), D(2024, 6, 8, 8, 0), D(2024, 6, 9, 8, 0), D(2024, 6, 14, 8, 0)]


def t_dom_or_dow():
    # day 13 OR Friday; 2024-09-01 is a Sunday
    got = next_cron_times("0 0 13 * 5", D(2024, 9, 1), 8)
    want = [D(2024, 9, d) for d in (6, 13, 20, 27)] + [D(2024, 10, d) for d in (4, 11, 13, 18)]
    assert got == want, got


def t_dom_only_31():
    assert next_cron_times("0 0 31 * *", D(2024, 1, 31), 3) == [
        D(2024, 3, 31), D(2024, 5, 31), D(2024, 7, 31)]


def t_restricted_star_step_is_or():
    # "*/10" is not exactly "*", so day-of-month and day-of-week are ORed:
    # days 1,11,21,31 OR Mondays. 2024-07-01 is a Monday.
    got = next_cron_times("0 0 */10 * 1", D(2024, 7, 1), 6)
    want = [D(2024, 7, 8), D(2024, 7, 11), D(2024, 7, 15), D(2024, 7, 21), D(2024, 7, 22), D(2024, 7, 29)]
    assert got == want, got


def t_lists_ranges_steps():
    got = next_cron_times("5,10-20/5,58 3 * * *", D(2024, 2, 29, 2, 59), 6)
    want = [D(2024, 2, 29, 3, m) for m in (5, 10, 15, 20, 58)] + [D(2024, 3, 1, 3, 5)]
    assert got == want, got


def t_month_step():
    assert next_cron_times("0 0 1 */3 *", D(2024, 11, 15), 3) == [
        D(2025, 1, 1), D(2025, 4, 1), D(2025, 7, 1)]


def t_year_rollover():
    assert next_cron_times("59 23 31 12 *", D(2024, 12, 31, 23, 59), 1) == [D(2025, 12, 31, 23, 59)]


def t_leap_day_skips_2100():
    assert next_cron_times("0 12 29 2 *", D(2097, 1, 1), 1) == [D(2104, 2, 29, 12, 0)]


def t_whitespace_between_fields():
    assert next_cron_times("0  12 *\t* *", D(2024, 1, 1), 1) == [D(2024, 1, 1, 12, 0)]


def t_impossible_date():
    raises(ValueError, next_cron_times, "0 0 30 2 *", D(2024, 1, 1), 1)


def t_malformed_counts():
    for e in ("* * * *", "* * * * * *", ""):
        raises(ValueError, next_cron_times, e, D(2024, 1, 1), 1)


def t_malformed_ranges():
    for e in ("60 * * * *", "* 24 * * *", "* * 0 * *", "* * 32 * *", "* * * 0 *", "* * * 13 *", "* * * * 8"):
        raises(ValueError, next_cron_times, e, D(2024, 1, 1), 1)


def t_malformed_syntax():
    for e in ("5-3 * * * *", "*/0 * * * *", "1,,2 * * * *", "a * * * *", "5/2 * * * *", "-1 * * * *", "1- * * * *"):
        raises(ValueError, next_cron_times, e, D(2024, 1, 1), 1)


TESTS = [(k, v) for k, v in list(globals().items()) if k.startswith("t_")]

if __name__ == "__main__":
    run(TESTS)
