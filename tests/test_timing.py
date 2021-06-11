from shoc.timing import Date, TimeDelta
from astropy import time
import pytest


def test_date():
    Date('2012-12-12') - Date('2012-12-10')
    
class TestTiming:
    @pytest.mark.parametrize(
        't',
        [
            # TimeDelta(TimeDelta(1)),
            TimeDelta(time.TimeDelta(1)),
            #  TimeDelta(1) * 2,
            #  2 * TimeDelta(1),
            #  TimeDelta(1) / 2
        ])
    def test_type(self, t):
        assert type(t) is TimeDelta
        print(t)
