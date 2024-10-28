import argparse
import warnings
import numpy as np
import pysindy as ps
import distutils.util
from scipy.io import loadmat
from datetime import datetime
import matplotlib.pyplot as plt
from tools.utils import compute_derivatives
from feature_library.laplace_library import LaplaceLibrary


def main(flags):
    # We will update this once the work becomes available on arXiv.
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--noise', type=float, default=0.30)
    parser.add_argument('--basis_rate', type=float, default=0.01)
    parser.add_argument('--basis_bias', type=float, default=0.05)
    parser.add_argument('--num_basis', type=int, default=50)
    parser.add_argument('--swap', type=str, default='energy', help="")
    parser.add_argument('--burn_in', type=int, default=10, help="")
    parser.add_argument('--if', default=False, type=lambda x: bool(distutils.util.strtobool(x)), help="")

    args = parser.parse_args()
    print(args)

    now = datetime.now()
    args.time = now.strftime("%Y_%m_%d_%H_%M_%S")

    main(args)
