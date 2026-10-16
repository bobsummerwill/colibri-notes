import datetime
import re
from typing import List


def next_cron_times(expr: str, start: datetime.datetime, n: int) -> list[datetime.datetime]:
    """
    Return the n earliest times strictly after `start` that match the cron expression `expr`.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    
    # Parse the cron expression
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(f"Expected 5 fields, got {len(fields)}")
    
    minute_field, hour_field, dom_field, month_field, dow_field = fields
    
    # Parse each field into a set of allowed values
    def parse_field(field: str, min_val: int, max_val: int, field_name: str) -> set:
        allowed = set()
        items = field.split(',')
        for item in items:
            if not item:
                raise ValueError(f"Empty item in {field_name} field")
            
            # Check for step
            step = None
            if '/' in item:
                parts = item.split('/')
                if len(parts) != 2:
                    raise ValueError(f"Invalid step syntax in {field_name} field: {item}")
                base, step_str = parts
                try:
                    step = int(step_str)
                except ValueError:
                    raise ValueError(f"Non-integer step in {field_name} field: {item}")
                if step < 1:
                    raise ValueError(f"Step must be >= 1 in {field_name} field: {item}")
            else:
                base = item
            
            if base == '*':
                if step is None:
                    allowed.update(range(min_val, max_val + 1))
                else:
                    allowed.update(range(min_val, max_val + 1, step))
            elif '-' in base:
                # Range
                range_parts = base.split('-')
                if len(range_parts) != 2:
                    raise ValueError(f"Invalid range in {field_name} field: {item}")
                try:
                    a = int(range_parts[0])
                    b = int(range_parts[1])
                except ValueError:
                    raise ValueError(f"Non-integer value in range in {field_name} field: {item}")
                if a < min_val or a > max_val or b < min_val or b > max_val:
                    raise ValueError(f"Value out of range in {field_name} field: {item}")
                if a > b:
                    raise ValueError(f"A > B in range in {field_name} field: {item}")
                if step is None:
                    allowed.update(range(a, b + 1))
                else:
                    allowed.update(range(a, b + 1, step))
            else:
                # Single value
                if step is not None:
                    raise ValueError(f"Step after single value in {field_name} field: {item}")
                try:
                    val = int(base)
                except ValueError:
                    raise ValueError(f"Non-integer value in {field_name} field: {item}")
                if val < min_val or val > max_val:
                    raise ValueError(f"Value out of range in {field_name} field: {item}")
                allowed.add(val)
        
        return allowed
    
    minutes = parse_field(minute_field, 0, 59, "minute")
    hours = parse_field(hour_field, 0, 23, "hour")
    doms = parse_field(dom_field, 1, 31, "day-of-month")
    months = parse_field(month_field, 1, 12, "month")
    dows = parse_field(dow_field, 0, 7, "day-of-week")
    
    # Normalize day-of-week: 7 -> 0 (Sunday)
    if 7 in dows:
        dows.discard(7)
        dows.add(0)
    # Python's weekday(): Monday=0, ..., Sunday=6
    # Cron: Sunday=0, Monday=1, ..., Saturday=6
    # Convert cron dow to Python weekday: (cron_dow + 6) % 7
    # Actually, let's just keep cron dow and compare directly
    # Cron dow: 0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat
    # Python weekday(): Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
    # So cron_dow = (python_weekday + 1) % 7
    # Or python_weekday = (cron_dow - 1) % 7
    
    # Determine if dom and dow are restricted
    dom_restricted = dom_field != '*'
    dow_restricted = dow_field != '*'
    
    def matches_date(dt: datetime.datetime) -> bool:
        """Check if the date part of dt matches the dom, month, and dow fields."""
        if dt.month not in months:
            return False
        
        # Get cron day-of-week
        # Python weekday: Mon=0, ..., Sun=6
        # Cron: Sun=0, Mon=1, ..., Sat=6
        python_wd = dt.weekday()  # Mon=0, ..., Sun=6
        cron_dow = (python_wd + 1) % 7  # Mon=1, Tue=2, ..., Sat=6, Sun=0
        
        dom_match = dt.day in doms
        dow_match = cron_dow in dows
        
        if dom_restricted and dow_restricted:
            return dom_match or dow_match
        elif dom_restricted:
            return dom_match
        elif dow_restricted:
            return dow_match
        else:
            return True
    
    # Start scanning from the next minute after start
    # We need times strictly after start
    current = start.replace(second=0, microsecond=0)
    if current <= start:
        current += datetime.timedelta(minutes=1)
    
    results = []
    max_date = start + datetime.timedelta(days=3000)
    
    # Optimize: iterate day by day, and for each matching day, iterate hours and minutes
    current_day = current.date()
    limit_day = max_date.date()
    
    while len(results) < n:
        if current_day > limit_day:
            break
        
        if matches_date(datetime.datetime.combine(current_day, datetime.time(0, 0))):
            # Find all matching times on this day that are after start (if it's the first day)
            for h in sorted(hours):
                for m in sorted(minutes):
                    dt = datetime.datetime(current_day.year, current_day.month, current_day.day, h, m)
                    if dt > start and dt <= max_date:
                        results.append(dt)
                        if len(results) >= n:
                            break
                if len(results) >= n:
                    break
        
        current_day += datetime.timedelta(days=1)
    
    if len(results) < n:
        raise ValueError(f"Fewer than {n} matching times found within 3000 days after {start}")
    
    return results[:n]
