"""Unit test for int install
"""
import math
import pytest

ROUND_OFF_ERROR = 0.001

def log_transform(x):
    if(x>=0):
        return math.log10(x+1)
    else:
        raise ValueError("Cannot log less than 0.")

def test_log_transform_1():
    """Unit test to showcase functionality of log transform of 0+1
    """
    expected_output_price = 0
    output_price = log_transform(0)
    assert math.fabs(output_price - expected_output_price) < ROUND_OFF_ERROR, \
        """Should show that the log(0+1) = 0."""
        
def test_log_transform_2():
    """Unit test to showcase functionality of log transform of 9999+1
    """
    expected_output_price = 4
    output_price = log_transform(9999)
    assert math.fabs(output_price - expected_output_price) < ROUND_OFF_ERROR, \
        """Should show that the log(10000) = 4."""
        
def test_log_transform_3():
    """Unit test to showcase functionality of log transform of -10+1
    """
    with pytest.raises(ValueError):      
        log_transform(-10)