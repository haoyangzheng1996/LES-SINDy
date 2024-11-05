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

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Seed the random number generators for reproducibility
    np.random.seed(100)

    # Load and plot the data
    data = loadmat("data/kuramoto_sivishinky_data.mat")
    time = np.ravel(data["tt"])
    x = np.ravel(data["x"])
    u = data["uu"]
    dt = time[1] - time[0]
    dx = x[1] - x[0]

    u_deriv = [np.zeros(u.shape) for i in range(4)]
    for i in range(u.shape[1]):
        derivs_sub = compute_derivatives(u[:, i], dx, [0, 0, 0, 0], n=4)
        for j in range(4):
            u_deriv[j][:, i] = derivs_sub[j]
    u_init = [u[flags.burn_in]]
    for i in range(4 - 1):
        u_init.append(u_deriv[i][flags.burn_in])

    u = u.copy() + flags.noise * np.std(u.copy()) * np.random.randn(u.shape[0], u.shape[1])
    u = u.reshape(len(x), len(time), 1)

    # Define from Laplace library
    X, T = np.meshgrid(x, time)
    XT = np.asarray([X, T]).T
    laplace_lib = LaplaceLibrary(
        library_functions=[lambda x: x],
        function_names=[lambda x: x],
        derivative_order=4,
        spatiotemporal_grid=XT,
        is_uniform=True,
        K=flags.num_basis,
        burn_in=flags.burn_in,
        include_interaction=True,
        periodic=True,
        initial=u_init
    )

    # Run optimizers
    optimizer = ps.STLSQ(threshold=0.5, alpha=0.001, normalize_columns=True)
    model1 = ps.SINDy(feature_library=laplace_lib, optimizer=optimizer)
    model1.fit(u)

    feature_names = laplace_lib.get_feature_names()
    print("Features: ", feature_names)
    print("")

    model1.print()

    optimizer = ps.SR3(
        threshold=0.8, max_iter=1000, thresholder="l0", normalize_columns=True)
    model2 = ps.SINDy(feature_library=laplace_lib, optimizer=optimizer)
    model2.fit(u)
    model2.print()

    optimizer = ps.SR3(
        threshold=0.5, max_iter=1000, thresholder="l1", normalize_columns=True)
    model3 = ps.SINDy(feature_library=laplace_lib, optimizer=optimizer)
    model3.fit(u)
    model3.print()

    optimizer = ps.SSR(normalize_columns=True, kappa=1e-1, max_iter=20)
    model4 = ps.SINDy(feature_library=laplace_lib, optimizer=optimizer)
    model4.fit(u)
    model4.print()

    optimizer = ps.SSR(
        criteria="model_residual", normalize_columns=True, kappa=1e-1, max_iter=20)
    model5 = ps.SINDy(feature_library=laplace_lib, optimizer=optimizer)
    model5.fit(u)
    model5.print()

    optimizer = ps.FROLS(normalize_columns=True, kappa=1e-5)
    model6 = ps.SINDy(feature_library=laplace_lib, optimizer=optimizer)
    model6.fit(u)
    model6.print()

    return laplace_lib, [model1, model2, model3, model4, model5, model6]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_basis', type=int, default=10)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--burn_in', type=int, default=100, help="")
    parser.add_argument('--if', default=False, type=lambda x: bool(distutils.util.strtobool(x)), help='')
 
    args = parser.parse_args()
    print(args)

    now = datetime.now()
    args.time = now.strftime("%Y_%m_%d_%H_%M_%S")

    lib, model = main(args)
