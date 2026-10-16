
import sys, datetime as dt
sys.path.insert(0, ".")
from solution import next_cron_times
D = dt.datetime
for e in ("5-3 * * * *", "*/0 * * * *", "1,,2 * * * *", "a * * * *", "5/2 * * * *", "-1 * * * *", "1- * * * *"):
    try:
        r = next_cron_times(e, D(2024, 1, 1), 1)
        print("NO-RAISE", repr(e), r)
    except ValueError as x:
        print("ValueError", repr(e), x)
    except BaseException as x:
        print("OTHER", repr(e), type(x).__name__, x)
