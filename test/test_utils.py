import inspect
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], "../.."))
from fl_puf.Utils.utils import Utils


class TestUtils:
    @staticmethod
    def test_rescale():
        assert round(Utils.rescale_lambda(0.2, 0, 0.3, 0, 1), 2) == 0.67
