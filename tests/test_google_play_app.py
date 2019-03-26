"""Unit test for int install
"""
import math
import pytest

ROUND_OFF_ERROR = 0.001

def int_installs(x):
    try:
        return int(x.replace(',', '').replace('+', ''))
    except:
        raise ValueError("Cannot transform to int.")
        
def test_int_install_1():
    """Unit test to showcase functionality of int of int
    """
    expected_output_price = 65000
    output_price = int_installs('65000')
    assert math.fabs(output_price - expected_output_price) < ROUND_OFF_ERROR, \
        """Should show that the installs is 65000."""

def test_int_install_2():
    """Unit test to showcase functionality of int of string with right format
    """
    expected_output_price = 65000
    output_price = int_installs('+65,000')
    assert math.fabs(output_price - expected_output_price) < ROUND_OFF_ERROR, \
        """Should show that the installs is 65000."""

def test_int_install_3():
    """Unit test to showcase functionality of int of strong with wrong format
    """
    with pytest.raises(ValueError):      
        int_installs('$65000')