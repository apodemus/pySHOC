
import pytest
from pyshoc import calDB


@pytest.fixture(params=('flat', 'dark'))
def kind(request):
    return request.param


class TestCalDB:

    def test_make(self):
        calDB.make(False)

    def test_load(self, kind):
        calDB.load(kind, master=False)
