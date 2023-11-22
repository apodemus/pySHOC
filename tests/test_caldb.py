
import pytest
from pyshoc import calDB


# calDB.make(new=True)


@pytest.fixture(params=('flat', 'dark'))
def kind(request):
    return request.param


class TestCalDB:

    def test_make(self, kind):
        calDB.make(kind, False)

    def test_load(self, kind):
        calDB.load(kind, False)
