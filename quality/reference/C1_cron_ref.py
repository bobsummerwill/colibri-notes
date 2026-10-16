"""Reference solution for C1 (next_cron_times). Not shown to the models."""
import datetime as _dt

_FIELDS = [("minute", 0, 59), ("hour", 0, 23), ("dom", 1, 31), ("month", 1, 12), ("dow", 0, 7)]


def _int(text):
    if not text.isdigit():
        raise ValueError(f"not an integer: {text!r}")
    return int(text)


def _parse_field(text, lo, hi, star_hi):
    values = set()
    if text == "":
        raise ValueError("empty field")
    for item in text.split(","):
        if item == "":
            raise ValueError("empty item")
        step = 1
        if "/" in item:
            base, _, step_text = item.partition("/")
            step = _int(step_text)
            if step < 1:
                raise ValueError("step must be >= 1")
            if base != "*" and "-" not in base:
                raise ValueError("step needs * or a range")
        else:
            base = item
        if base == "*":
            a, b = lo, star_hi
        elif "-" in base:
            a_text, _, b_text = base.partition("-")
            a, b = _int(a_text), _int(b_text)
            if a > b:
                raise ValueError("range start > end")
        else:
            a = b = _int(base)
        if a < lo or b > hi:
            raise ValueError("value out of range")
        values.update(range(a, b + 1, step))
    return values


def _parse(expr):
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError("need 5 fields")
    sets = []
    for text, (_, lo, hi) in zip(parts, _FIELDS):
        star_hi = 6 if hi == 7 else hi
        sets.append(_parse_field(text, lo, hi, star_hi))
    minute, hour, dom, month, dow = sets
    if 7 in dow:
        dow = (dow - {7}) | {0}
    return minute, hour, dom, month, dow, parts[2] == "*", parts[4] == "*"


def next_cron_times(expr, start, n):
    minute, hour, dom, month, dow, dom_star, dow_star = _parse(expr)
    minutes = sorted(minute)
    hours = sorted(hour)

    def day_ok(d):
        if d.month not in month:
            return False
        dom_hit = d.day in dom
        dow_hit = (d.weekday() + 1) % 7 in dow
        if dom_star and dow_star:
            return True
        if dom_star:
            return dow_hit
        if dow_star:
            return dom_hit
        return dom_hit or dow_hit

    out = []
    limit = start + _dt.timedelta(days=3000)
    day = start.date()
    while len(out) < n:
        if _dt.datetime.combine(day, _dt.time()) > limit:
            raise ValueError("not enough matches within 3000 days")
        if day_ok(day):
            for h in hours:
                for m in minutes:
                    t = _dt.datetime(day.year, day.month, day.day, h, m)
                    if t > start and t <= limit:
                        out.append(t)
                        if len(out) == n:
                            return out
        day += _dt.timedelta(days=1)
    return out
