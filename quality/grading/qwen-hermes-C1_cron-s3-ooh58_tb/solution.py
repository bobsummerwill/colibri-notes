import datetime
import re


def _parse_field(field: str, lo: int, hi: int, is_dow: bool = False) -> set[int]:
    """Parse a single cron field and return the set of matching values."""
    values = set()
    items = field.split(',')
    for item in items:
        if not item:
            raise ValueError(f"Empty item in field: '{field}'")
        
        # Check for step
        if '/' in item:
            base, step_str = item.split('/')
            try:
                step = int(step_str)
            except ValueError:
                raise ValueError(f"Invalid step: '{step_str}'")
            if step < 1:
                raise ValueError(f"Step must be >= 1, got {step}")
        else:
            base = item
            step = 1
        
        # Parse base
        if base == '*':
            start_val = lo
            end_val = hi
        elif '-' in base:
            parts = base.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid range: '{base}'")
            try:
                a = int(parts[0])
                b = int(parts[1])
            except ValueError:
                raise ValueError(f"Invalid range values: '{base}'")
            if a < lo or a > hi:
                raise ValueError(f"Value {a} out of range [{lo}, {hi}]")
            if b < lo or b > hi:
                raise ValueError(f"Value {b} out of range [{lo}, {hi}]")
            if a > b:
                raise ValueError(f"Range start {a} > end {b}")
            start_val = a
            end_val = b
        else:
            # Single value
            try:
                val = int(base)
            except ValueError:
                raise ValueError(f"Invalid value: '{base}'")
            if val < lo or val > hi:
                raise ValueError(f"Value {val} out of range [{lo}, {hi}]")
            if '/' in item:
                # Step after single value is not allowed
                raise ValueError(f"Step after single value: '{item}'")
            values.add(val)
            continue
        
        # Add stepped values
        for v in range(start_val, end_val + 1, step):
            values.add(v)
    
    if is_dow:
        # Normalize 7 to 0 (Sunday)
        normalized = set()
        for v in values:
            if v == 7:
                normalized.add(0)
            else:
                normalized.add(v)
        return normalized
    
    return values


def _parse_cron(expr: str):
    """Parse a cron expression and return (minutes, hours, dom, month, dow, dom_restricted, dow_restricted)."""
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(f"Expected 5 fields, got {len(fields)}")
    
    minute_field, hour_field, dom_field, month_field, dow_field = fields
    
    minutes = _parse_field(minute_field, 0, 59)
    hours = _parse_field(hour_field, 0, 23)
    dom = _parse_field(dom_field, 1, 31)
    months = _parse_field(month_field, 1, 12)
    dow = _parse_field(dow_field, 0, 7, is_dow=True)
    
    # Check if dom or dow is restricted (not exactly '*')
    dom_restricted = dom_field.strip() != '*'
    dow_restricted = dow_field.strip() != '*'
    
    return minutes, hours, dom, months, dow, dom_restricted, dow_restricted


def next_cron_times(expr: str, start: datetime.datetime, n: int) -> list[datetime.datetime]:
    """Return the n earliest times strictly after start that match the cron expression expr."""
    if n < 1:
        raise ValueError("n must be >= 1")
    
    minutes, hours, dom, months, dow, dom_restricted, dow_restricted = _parse_cron(expr)
    
    # Start from the next minute after start
    # We need times strictly after start, with second == 0 and microsecond == 0
    # So we start from start + 1 minute, but adjusted to minute boundary
    current = start.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
    
    # But we also need to make sure we're looking at valid minute/hour slots
    # Let's iterate through days and find matching times
    
    results = []
    max_days = 3000
    
    # We'll iterate day by day, and for each day, check if it matches the date criteria
    # Then for matching days, find all matching (hour, minute) combinations
    
    # Convert sets to sorted lists for efficient lookup
    sorted_minutes = sorted(minutes)
    sorted_hours = sorted(hours)
    
    # For each day in the range
    for day_offset in range(max_days + 1):
        if len(results) >= n:
            break
        
        day_date = start.date() + datetime.timedelta(days=day_offset)
        
        # Check month
        if day_date.month not in months:
            continue
        
        # Check day matching
        dom_match = day_date.day in dom
        # Python: Monday=0, Sunday=6; cron: Sunday=0, Monday=1, ..., Saturday=6
        python_dow = day_date.weekday()  # 0=Mon, ..., 6=Sun
        cron_dow = (python_dow + 1) % 7  # Convert to cron: Mon=1, ..., Sat=6, Sun=0
        
        dow_match = cron_dow in dow
        
        if dom_restricted and dow_restricted:
            date_matches = dom_match or dow_match
        elif dom_restricted:
            date_matches = dom_match
        elif dow_restricted:
            date_matches = dow_match
        else:
            date_matches = True
        
        if not date_matches:
            continue
        
        # For this day, find all matching (hour, minute) combinations
        for hour in sorted_hours:
            for minute in sorted_minutes:
                candidate = datetime.datetime(day_date.year, day_date.month, day_date.day, hour, minute)
                
                if candidate <= start:
                    continue
                
                results.append(candidate)
                if len(results) >= n:
                    break
            if len(results) >= n:
                break
    
    if len(results) < n:
        raise ValueError(f"Fewer than {n} matching times found within 3000 days after start")
    
    return results[:n]
