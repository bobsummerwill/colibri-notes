import datetime
import re
from typing import List


def _parse_field(field: str, min_val: int, max_val: int, field_name: str) -> set[int]:
    """Parse a single cron field and return the set of matching values."""
    values = set()
    items = field.split(',')
    for item in items:
        if not item:
            raise ValueError(f"Empty item in {field_name} field")
        
        # Check for step
        step = 1
        if '/' in item:
            parts = item.split('/')
            if len(parts) != 2:
                raise ValueError(f"Invalid step in {field_name} field: {item}")
            base, step_str = parts
            try:
                step = int(step_str)
            except ValueError:
                raise ValueError(f"Non-integer step in {field_name} field: {item}")
            if step < 1:
                raise ValueError(f"Step must be >= 1 in {field_name} field: {item}")
            item = base
        
        # Now parse the base part
        if item == '*':
            start_val = min_val
            end_val = max_val
        elif '-' in item:
            parts = item.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid range in {field_name} field: {item}")
            try:
                start_val = int(parts[0])
                end_val = int(parts[1])
            except ValueError:
                raise ValueError(f"Non-integer value in {field_name} field: {item}")
            if start_val > end_val:
                raise ValueError(f"Range start > end in {field_name} field: {item}")
            if start_val < min_val or end_val > max_val:
                raise ValueError(f"Value out of range in {field_name} field: {item}")
        else:
            # Single value
            try:
                val = int(item)
            except ValueError:
                raise ValueError(f"Non-integer value in {field_name} field: {item}")
            if val < min_val or val > max_val:
                raise ValueError(f"Value out of range in {field_name} field: {item}")
            # Step is only allowed after * or a range, not after a single value
            if '/' in item:
                raise ValueError(f"Step after single value in {field_name} field: {item}")
            values.add(val)
            continue
        
        # For * or range, apply step
        for v in range(start_val, end_val + 1, step):
            values.add(v)
    
    return values


def _parse_cron(expr: str) -> tuple[set[int], set[int], set[int], set[int], set[int], bool, bool]:
    """Parse a cron expression and return the sets of matching values for each field,
    plus flags indicating whether day-of-month and day-of-week are restricted."""
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(f"Expected 5 fields, got {len(fields)}")
    
    minute_str, hour_str, dom_str, month_str, dow_str = fields
    
    minutes = _parse_field(minute_str, 0, 59, "minute")
    hours = _parse_field(hour_str, 0, 23, "hour")
    doms = _parse_field(dom_str, 1, 31, "day-of-month")
    months = _parse_field(month_str, 1, 12, "month")
    dows = _parse_field(dow_str, 0, 7, "day-of-week")
    
    # Normalize day-of-week: 7 means Sunday (0)
    if 7 in dows:
        dows.discard(7)
        dows.add(0)
    
    # Determine if dom and dow are restricted
    dom_restricted = dom_str != '*'
    dow_restricted = dow_str != '*'
    
    return minutes, hours, doms, months, dows, dom_restricted, dow_restricted


def _day_matches(date: datetime.date, doms: set[int], dows: set[int], dom_restricted: bool, dow_restricted: bool) -> bool:
    """Check if a date matches the day-of-month and day-of-week fields."""
    dom_match = date.day in doms
    # Python's weekday(): Monday=0, Sunday=6. Cron: Sunday=0, Monday=1, ..., Saturday=6
    cron_dow = (date.weekday() + 1) % 7  # Convert to cron format
    dow_match = cron_dow in dows
    
    if dom_restricted and dow_restricted:
        return dom_match or dow_match
    elif dom_restricted:
        return dom_match
    elif dow_restricted:
        return dow_match
    else:
        return True


def next_cron_times(expr: str, start: datetime.datetime, n: int) -> list[datetime.datetime]:
    """Return the n earliest times strictly after `start` that match the cron expression `expr`."""
    if n < 1:
        raise ValueError("n must be >= 1")
    
    minutes, hours, doms, months, dows, dom_restricted, dow_restricted = _parse_cron(expr)
    
    if not minutes or not hours or not doms or not months or not dows:
        raise ValueError("Empty field after parsing")
    
    results: list[datetime.datetime] = []
    
    # Start from the next minute after start
    # We need times strictly after start, with second=0 and microsecond=0
    # So the earliest candidate is start truncated to minute, then add 1 minute if start has seconds/microseconds
    # Actually, we want the first minute slot that is strictly after start.
    # If start is 10:30:45, the first candidate minute is 10:31:00.
    # If start is 10:30:00, the first candidate minute is 10:31:00 (since 10:30:00 is not strictly after).
    
    # Let's compute the first candidate datetime
    # Truncate start to minute
    start_minute = start.replace(second=0, microsecond=0)
    # The first candidate is start_minute + 1 minute
    candidate = start_minute + datetime.timedelta(minutes=1)
    
    # We'll scan day by day, and for each day, check if it matches the day fields.
    # If it does, we iterate through hours and minutes that match.
    
    max_days = 3000
    current_date = candidate.date()
    end_date = start.date() + datetime.timedelta(days=max_days)
    
    # Sort hours and minutes for efficient iteration
    sorted_hours = sorted(hours)
    sorted_minutes = sorted(minutes)
    
    while current_date <= end_date and len(results) < n:
        # Check if this date matches the day fields
        if _day_matches(current_date, doms, dows, dom_restricted, dow_restricted):
            # Check if the month matches
            if current_date.month in months:
                # For each hour and minute that matches, check if the datetime is after start
                for hour in sorted_hours:
                    for minute in sorted_minutes:
                        dt = datetime.datetime(current_date.year, current_date.month, current_date.day, hour, minute)
                        if dt > start:
                            results.append(dt)
                            if len(results) >= n:
                                break
                    if len(results) >= n:
                        break
        current_date += datetime.timedelta(days=1)
    
    if len(results) < n:
        raise ValueError(f"Fewer than {n} matching times found within 3000 days after start")
    
    return results
