"""
Floating Point Utilities
Issue #63 Fix: Provides epsilon-based floating point comparison utilities
"""

# Issue #63 Fix: Use ε=1e-9 for all floating point comparisons
EPSILON = 1e-9


def float_equals(a: float, b: float, epsilon: float = EPSILON) -> bool:
    """
    Check if two floating point numbers are equal within epsilon tolerance.
    
    Args:
        a: First number
        b: Second number  
        epsilon: Tolerance for comparison (default: 1e-9)
        
    Returns:
        True if |a - b| <= epsilon
    """
    return abs(a - b) <= epsilon


def float_zero(value: float, epsilon: float = EPSILON) -> bool:
    """
    Check if a floating point number is effectively zero.
    
    Args:
        value: Number to check
        epsilon: Tolerance for comparison (default: 1e-9)
        
    Returns: 
        True if |value| <= epsilon
    """
    return abs(value) <= epsilon


def float_one(value: float, epsilon: float = EPSILON) -> bool:
    """
    Check if a floating point number is effectively one.
    
    Args:
        value: Number to check
        epsilon: Tolerance for comparison (default: 1e-9)
        
    Returns:
        True if |value - 1.0| <= epsilon
    """
    return abs(value - 1.0) <= epsilon


def float_greater_than(a: float, b: float, epsilon: float = EPSILON) -> bool:
    """
    Check if a floating point number is greater than another, accounting for epsilon.
    
    Args:
        a: First number
        b: Second number
        epsilon: Tolerance for comparison (default: 1e-9)
        
    Returns:
        True if a > b + epsilon
    """
    return a > b + epsilon


def float_less_than(a: float, b: float, epsilon: float = EPSILON) -> bool:
    """
    Check if a floating point number is less than another, accounting for epsilon.
    
    Args:
        a: First number
        b: Second number
        epsilon: Tolerance for comparison (default: 1e-9)
        
    Returns:
        True if a < b - epsilon
    """
    return a < b - epsilon


def float_greater_equal(a: float, b: float, epsilon: float = EPSILON) -> bool:
    """
    Check if a floating point number is greater than or equal to another.
    
    Args:
        a: First number
        b: Second number
        epsilon: Tolerance for comparison (default: 1e-9)
        
    Returns:
        True if a >= b - epsilon
    """
    return a >= b - epsilon


def float_less_equal(a: float, b: float, epsilon: float = EPSILON) -> bool:
    """
    Check if a floating point number is less than or equal to another.
    
    Args:
        a: First number
        b: Second number
        epsilon: Tolerance for comparison (default: 1e-9)
        
    Returns:
        True if a <= b + epsilon
    """
    return a <= b + epsilon


def clamp_float(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Clamp a floating point value to a range, with epsilon-aware boundary checks.
    
    Args:
        value: Value to clamp
        min_val: Minimum value (default: 0.0)
        max_val: Maximum value (default: 1.0)
        
    Returns:
        Clamped value
    """
    if float_less_than(value, min_val):
        return min_val
    elif float_greater_than(value, max_val):
        return max_val
    else:
        return value