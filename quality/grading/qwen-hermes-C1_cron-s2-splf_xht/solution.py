import datetime
import re


def next_cron_times(expr: str, start: datetime.datetime, n: int) -> list[datetime.datetime]:
    """Return the n earliest times strictly after `start` that match the cron expression `expr`."""
    
    if n < 1:
        raise ValueError("n must be >= 1")
    
    # Parse the cron expression
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError(f"Expected 5 fields, got {len(fields)}")
    
    minute_str, hour_str, dom_str, month_str, dow_str = fields
    
    def parse_field(field_str: str, min_val: int, max_val: int, field_name: str) -> set:
        """Parse a single cron field and return the set of valid values."""
        values = set()
        items = field_str.split(',')
        
        for item in items:
            if not item:
                raise ValueError(f"Empty item in {field_name} field")
            
            # Check for step
            if '/' in item:
                parts = item.split('/')
                if len(parts) != 2:
                    raise ValueError(f"Invalid step syntax in {field_name} field: {item}")
                base, step_str = parts
                try:
                    step = int(step_str)
                except ValueError:
                    raise ValueError(f"Non-integer step in {field_name} field: {step_str}")
                if step < 1:
                    raise ValueError(f"Step must be >= 1 in {field_name} field: {step}")
                
                # Parse the base part
                if base == '*':
                    start_val = min_val
                    end_val = max_val
                elif '-' in base:
                    range_parts = base.split('-')
                    if len(range_parts) != 2:
                        raise ValueError(f"Invalid range in {field_name} field: {base}")
                    try:
                        a = int(range_parts[0])
                        b = int(range_parts[1])
                    except ValueError:
                        raise ValueError(f"Non-integer value in range in {field_name} field: {base}")
                    if a > b:
                        raise ValueError(f"Invalid range in {field_name} field: {a} > {b}")
                    if a < min_val or b > max_val:
                        raise ValueError(f"Value out of range in {field_name} field: {base}")
                    start_val = a
                    end_val = b
                else:
                    # Step after a single value is not allowed
                    raise ValueError(f"Step after single value not allowed in {field_name} field: {item}")
                
                for v in range(start_val, end_val + 1, step):
                    values.add(v)
            
            else:
                if item == '*':
                    for v in range(min_val, max_val + 1):
                        values.add(v)
                elif '-' in item:
                    range_parts = item.split('-')
                    if len(range_parts) != 2:
                        raise ValueError(f"Invalid range in {field_name} field: {item}")
                    try:
                        a = int(range_parts[0])
                        b = int(range_parts[1])
                    except ValueError:
                        raise ValueError(f"Non-integer value in range in {field_name} field: {item}")
                    if a > b:
                        raise ValueError(f"Invalid range in {field_name} field: {a} > {b}")
                    if a < min_val or b > max_val:
                        raise ValueError(f"Value out of range in {field_name} field: {item}")
                    for v in range(a, b + 1):
                        values.add(v)
                else:
                    try:
                        v = int(item)
                    except ValueError:
                        raise ValueError(f"Non-integer value in {field_name} field: {item}")
                    if v < min_val or v > max_val:
                        raise ValueError(f"Value out of range in {field_name} field: {v}")
                    values.add(v)
        
        return values
    
    minutes = parse_field(minute_str, 0, 59, "minute")
    hours = parse_field(hour_str, 0, 23, "hour")
    doms = parse_field(dom_str, 1, 31, "day-of-month")
    months = parse_field(month_str, 1, 12, "month")
    
    # For day-of-week, we need to handle 0 and 7 both meaning Sunday
    dow_raw = parse_field(dow_str, 0, 7, "day-of-week")
    # Normalize: map 7 to 0 (Sunday)
    dows = set()
    for d in dow_raw:
        if d == 7:
            dows.add(0)
        else:
            dows.add(d)
    
    # Determine if dom and dow are restricted
    dom_restricted = dom_str != '*'
    dow_restricted = dow_str != '*'
    
    # Start scanning from the next minute after start
    current = start.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
    
    results = []
    limit_date = start + datetime.timedelta(days=3000)
    
    while len(results) < n and current < limit_date:
        if (current.month in months and
            current.hour in hours and
            current.minute in minutes):
            # Check day rules
            dom_matches = current.day in doms
            dow_matches = ((current.weekday() + 1) % 7) in dows  # Python: Monday=0, Sunday=6; Cron: Sunday=0
            
            if dom_restricted and dow_restricted:
                day_matches = dom_matches or dow_matches
            elif dom_restricted:
                day_matches = dom_matches
            elif dow_restricted:
                day_matches = dow_matches
            else:
                day_matches = True
            
            if day_matches:
                results.append(current)
        
        current += datetime.timedelta(minutes=1)
    
    if len(results) < n:
        raise ValueError(f"Fewer than {n} matching times found within 3000 days after {start}")
    
    return results
