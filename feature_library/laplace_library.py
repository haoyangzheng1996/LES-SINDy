import warnings
import numpy as np
from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r
from itertools import product as iproduct
from tools.utils import compute_derivatives

from sklearn import __version__
from sklearn.utils.validation import check_is_fitted

from pysindy.utils import AxesArray
from pysindy.feature_library.base import BaseFeatureLibrary
from pysindy.feature_library.base import x_sequence_or_item
from pysindy.differentiation import FiniteDifference


class LaplaceLibrary(BaseFeatureLibrary):

    def __init__(self):
        super(LaplaceLibrary, self).__init__()

        # We will update this once the work becomes available on arXiv.
