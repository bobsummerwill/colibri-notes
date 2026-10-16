import datetime
from datetime import datetime as dt
import re

def next_cron_times(expr: str, start: datetime.datetime, n: int) -> list:
    if n < 1:
        raise ValueError("n must be >= 1")
    
    # Parse expression
    fields = expr.split()
    if len(fields) != 5:
        raise ValueError("Expression must have exactly 5 fields")
    
    minute_field, hour_field, dom_field, month_field, dow_field = fields
    
    # Parse each field into a set of allowed values
    minutes = parse_field(minute_field, 0, 59)
    hours = parse_field(hour_field, 0, 23)
    doms = parse_field(dom_field, 1, 31)
    months = parse_field(month_field, 1, 12)
    dows = parse_field(dow_field, 0, 6)  # normalize 7 to 0
    
    # Check day rule restrictions
    dom_restricted = (dom_field != '*')
    dow_restricted = (dow_field != '*')
    
    # Start scanning from the next minute after start
    current = start.replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
    
    results = []
    end_limit = start + datetime.timedelta(days=3000)
    
    while len(results) < n and current <= end_limit:
        # Check month
        if current.month in months:
            # Check day-of-month and day-of-week with the special rule
            dom_match = current.day in doms
            dow_match = current.weekday() in dows  # weekday() returns 0=Monday, 6=Sunday
            
            # Convert to cron's convention: 0=Sunday, 1=Monday, ..., 6=Saturday
            cron_dow = (current.weekday() + 1) % 7
            dow_match = cron_dow in dows
            
            day_match = False
            if dom_restricted and dow_restricted:
                day_match = dom_match or dow_match
            elif dom_restricted:
                day_match = dom_match
            elif dow_restricted:
                day_match = dow_match
            else:
                day_match = True
            
            if day_match:
                # Check hour and minute
                if current.hour in hours and current.minute in minutes:
                    results.append(current)
        
        current += datetime.timedelta(minutes=1)
    
    if len(results) < n:
        raise ValueError("Fewer than n matching times within 3000 days")
    
    return results


def parse_field(field, min_val, max_val):
    """Parse a cron field into a set of allowed values."""
    allowed = set()
    
    # Handle day-of-week special case: 7 means Sunday (same as 0)
    if max_val == 6 and field != '*':
        # We'll handle 7 separately
        pass
    
    for item in field.split(','):
        if not item:
            raise ValueError("Empty item in field")
        
        # Check for step
        step = 1
        if '/' in item:
            base, step_str = item.split('/', 1)
            if not step_str:
                raise ValueError("Empty step")
            try:
                step = int(step_str)
            except ValueError:
                raise ValueError("Invalid step")
            if step <= 0:
                raise ValueError("Step must be positive")
        else:
            base = item
        
        if base == '*':
            start_val = min_val
            end_val = max_val
        elif '-' in base:
            parts = base.split('-')
            if len(parts) != 2:
                raise ValueError("Invalid range")
            try:
                start_val = int(parts[0])
                end_val = int(parts[1])
            except ValueError:
                raise ValueError("Invalid range values")
            if start_val > end_val:
                raise ValueError("Range start > end")
        else:
            # Single value
            try:
                val = int(base)
            except ValueError:
                raise ValueError("Invalid value")
            if val < min_val or val > max_val:
                raise ValueError("Value out of range")
            # For day-of-week, 7 is allowed and means Sunday (0)
            if max_val == 6 and val == 7:
                val = 0
            allowed.add(val)
            continue
        
        # Validate range bounds
        if start_val < min_val or end_val > max_val:
            raise ValueError("Range out of bounds")
        
        # For day-of-week, handle 7 as 0
        if max_val == 6:
            if start_val == 7:
                start_val = 0
            if end_val == 7:
                end_val = 0
            # If range crosses 7->0, handle specially
            if start_val > end_val and start_val != 0:
                # This shouldn't happen after normalization
                pass
        
        # Add values with step
        val = start_val
        while val <= end_val:
            allowed.add(val)
            val += step
    
    return allowed
