
import pytest
TmSt = 1
TmEx = 2

@pytest.mark.parametrize('test', [[('stuff', 'in')]])
def test_fader(test):
    pass

def check_fader(test):
    pass

def verify_fader(test):
    pass

def verify_fader(test):
    'Hey, ho.'
    assert test.passed()

def test_calculate_fades():
    calcs = [(0, 4, 0, 0, 10, 0, 0, 6, 10), (None, 4, 0, 0, 10, 0, 0, 6, 10)]
