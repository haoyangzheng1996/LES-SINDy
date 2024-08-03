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
    """
    The main framework follows WeakPDELibrary
    https://github.com/dynamicslab/pysindy/blob/master/pysindy/feature_library/weak_pde_library.py
    Will update the instruction later - Haoyang Zheng on August 03, 2024
    """

    def __init__(
            self,
            library_functions=[],
            derivative_order=0,
            spatiotemporal_grid=None,
            function_names=None,
            interaction_only=True,
            include_bias=False,
            include_interaction=True,
            basis_rate=0.05,
            basis_bias=0.5,
            K=10,
            H_xt=None,
            p=4,
            burn_in=50,
            library_ensemble=False,
            ensemble_indices=[0],
            num_pts_per_domain=None,
            implicit_terms=False,
            multiindices=None,
            differentiation_method=FiniteDifference,
            diff_kwargs={},
            is_uniform=None,
            periodic=None,
            initial=None
    ):
        super(LaplaceLibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        self.functions = library_functions
        self.derivative_order = derivative_order
        self.function_names = function_names
        self.interaction_only = interaction_only
        self.implicit_terms = implicit_terms
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.K = K
        self.H_xt = H_xt
        self.p = p
        self.num_trajectories = 1
        self.differentiation_method = differentiation_method
        self.diff_kwargs = diff_kwargs

        self.burn_in = burn_in  # from the concept of "burn-in samples" in Markov Chain Monte Carlo (MCMC)
        self.weight_scale = "normal"
        self.basis_rate = basis_rate
        self.basis_bias = basis_bias
        self.initial = initial

        if function_names and (len(library_functions) != len(function_names)):
            raise ValueError(
                "library_functions and function_names must have the same"
                " number of elements"
            )
        if library_functions is None and derivative_order == 0:
            raise ValueError(
                "No library functions were specified, and no "
                "derivatives were asked for. The library is empty."
            )
        if spatiotemporal_grid is None:
            raise ValueError(
                "Spatiotemporal grid was not passed, and at least a 1D"
                " grid is required, corresponding to the time base."
            )
        if num_pts_per_domain is not None:
            warnings.warn(
                "The parameter num_pts_per_domain is now deprecated. This "
                "value will be ignored by the library."
            )
        if is_uniform is not None or periodic is not None:
            # DeprecationWarning are ignored by default...
            warnings.warn(
                "is_uniform and periodic have been deprecated."
                "in favor of differetiation_method and diff_kwargs.",
                UserWarning,
            )

        # list of integrals
        indices = ()
        if np.array(spatiotemporal_grid).ndim == 1:
            spatiotemporal_grid = np.reshape(
                spatiotemporal_grid, (len(spatiotemporal_grid), 1)
            )
        dims = spatiotemporal_grid.shape[:-1]
        self.grid_dims = dims
        self.grid_ndim = len(dims)

        # if want to include temporal terms -> range(len(dims))
        if self.implicit_terms:
            self.ind_range = len(dims)
        else:
            self.ind_range = len(dims) - 1

        for i in range(self.ind_range):
            indices = indices + (range(derivative_order + 1),)

        if multiindices is None:
            multiindices = []
            for ind in iproduct(*indices):
                current = np.array(ind)
                if np.sum(ind) > 0 and np.sum(ind) <= derivative_order:
                    multiindices.append(current)
            multiindices = np.array(multiindices)
        num_derivatives = len(multiindices)
        if num_derivatives > 0:
            self.derivative_order = np.max(multiindices)

        self.num_derivatives = num_derivatives
        self.multiindices = multiindices
        self.spatiotemporal_grid = spatiotemporal_grid

        self.grid_dims = self.spatiotemporal_grid.shape[:-1]
        self.inds_k = np.arange(self.burn_in, self.grid_dims[0])

        self.x = self.spatiotemporal_grid[:, 0, 0]
        self.t = self.spatiotemporal_grid[0, :, 1]
        self.dx = np.mean(self.x[1:] - self.x[:-1])
        self.dt = np.mean(self.t[1:] - self.t[:-1])

        # Weak form checks and setup
        self._laplace_setup()

    def _laplace_setup(self):
        xt1, xt2 = self._get_spatial_endpoints()
        L_xt = xt2 - xt1
        if self.H_xt is not None:
            if np.isscalar(self.H_xt):
                self.H_xt = np.array(self.grid_ndim * [self.H_xt])
            if self.grid_ndim != len(self.H_xt):
                raise ValueError(
                    "The user-defined grid (spatiotemporal_grid) and "
                    "the user-defined sizes of the subdomains for the "
                    "weak form, do not have the same # of spatiotemporal "
                    "dimensions. For instance, if spatiotemporal_grid is 4D, "
                    "then H_xt should be a 4D list of the subdomain lengths."
                )
            if any(self.H_xt <= np.zeros(len(self.H_xt))):
                raise ValueError("Values in H_xt must be a positive float.")
            elif any(self.H_xt >= L_xt / 2.0):
                raise ValueError(
                    "2 * H_xt in some dimension is larger than the "
                    "corresponding grid dimension."
                )
        else:
            self.H_xt = L_xt / 20.0

        if self.spatiotemporal_grid is not None:
            if self.p < 0:
                raise ValueError("Poly degree of the spatial weights must be > 0")
            if self.p < self.derivative_order:
                self.p = self.derivative_order
        if self.K <= 0:
            raise ValueError("The number of subdomains must be > 0")

        self._set_up_weights()

    def _get_spatial_endpoints(self):
        x1 = np.zeros(self.grid_ndim)
        x2 = np.zeros(self.grid_ndim)
        for i in range(self.grid_ndim):
            inds = [slice(None)] * (self.grid_ndim + 1)
            for j in range(self.grid_ndim):
                inds[j] = 0
            x1[i] = self.spatiotemporal_grid[tuple(inds)][i]
            inds[i] = -1
            x2[i] = self.spatiotemporal_grid[tuple(inds)][i]
        return x1, x2

    def _set_up_each_weight(self, idx, scale_type='normal'):
        if scale_type == 'normal':
            return self.basis_bias + self.basis_rate * idx
        elif scale_type == 'log':
            return np.log(self.basis_bias + self.basis_rate * idx)
        elif scale_type == 'logit':
            return 1 / (1 + np.exp(-1 * self.basis_bias - self.basis_rate * idx))
        elif scale_type == 'square root':
            return np.sqrt(self.basis_bias + self.basis_rate * idx)
        else:
            raise ValueError("Unable to identify scale type. Available scale_types: normal, log, logit, square root.")

    def _set_up_weights(self):
        """
        Sets up weights needed for the weak library. Integrals over domain cells are
        approximated as dot products of weights and the input data.
        """

        # Indices for space-time points that lie in the domain cells
        self.s = []
        self.weights = []
        grid_clip = self.spatiotemporal_grid[self.burn_in:, 0, 0] - self.spatiotemporal_grid[self.burn_in, 0, 0]
        for i in range(self.K):
            self.s.append(self._set_up_each_weight(i))
            self.weights.append(np.exp(-1 * grid_clip * self.s[-1]))
        print("Assign weights done")

    @staticmethod
    def _combinations(n_features, n_args, interaction_only):
        """
        Get the combinations of features to be passed to a library function.
        """
        comb = combinations if interaction_only else combinations_w_r
        return comb(range(n_features), n_args)

    def convert_u_dot_integral(self, u):
        """
        Takes a full set of spatiotemporal fields u(x, t) and finds the weak
        form of u_dot.
        """
        K = self.K
        gdim = self.grid_ndim
        u_dot_integral = np.zeros((K * (u.shape[1] - self.burn_in), 1))
        deriv_orders = np.zeros(gdim)
        deriv_orders[-1] = 1

        ut = np.zeros(u.shape)
        for i in range(ut.shape[0]):
            ut[i] = compute_derivatives(u[i], self.dt, [0, 0, 0, 0])[0]
            # ut[i] = compute_derivatives(u[i], self.dt, [0], n=1)[0]

        ut_clip = np.squeeze(ut[self.burn_in:, self.burn_in:])

        # Extract the input features on indices in each domain cell
        for i in range(self.K):
            for j in range(ut_clip.shape[1]):
                idx = i * ut_clip.shape[1] + j
                u_dot_integral[idx] = self.trapezoidal_laplace(ut_clip[:, j], self.weights[i])

        return u_dot_integral

    def get_feature_names(self, input_features=None):
        """
        Return feature names for output features.

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        check_is_fitted(self)
        if float(__version__[:3]) >= 1.0:
            n_features = self.n_features_in_
        else:
            n_features = self.n_input_features_
        if input_features is None:
            input_features = ["x%d" % i for i in range(n_features)]
        if self.function_names is None:
            self.function_names = list(
                map(
                    lambda i: (lambda *x: "f" + str(i) + "(" + ",".join(x) + ")"),
                    range(n_features),
                )
            )
        feature_names = []

        # Include constant term
        if self.include_bias:
            feature_names.append("1")

        # Include any non-derivative terms
        for i, f in enumerate(self.functions):
            for c in self._combinations(
                    n_features, f.__code__.co_argcount, self.interaction_only
            ):
                feature_names.append(
                    self.function_names[i](*[input_features[j] for j in c])
                )

        if self.grid_ndim != 0:

            def derivative_string(multiindex):
                ret = ""
                for axis in range(self.ind_range):
                    if (axis == self.ind_range - 1) and (
                            self.ind_range == self.grid_ndim
                    ):
                        str_deriv = "t"
                    else:
                        str_deriv = str(axis + 1)
                    for i in range(multiindex[axis]):
                        ret = ret + str_deriv
                return ret

            # Include integral terms
            for k in range(self.num_derivatives):
                for j in range(n_features):
                    feature_names.append(
                        input_features[j] + "_" + derivative_string(self.multiindices[k]))
            # Include mixed non-derivative + integral terms
            if self.include_interaction:
                for k in range(self.num_derivatives):
                    for i, f in enumerate(self.functions):
                        for c in self._combinations(
                                n_features,
                                f.__code__.co_argcount,
                                self.interaction_only,
                        ):
                            for jj in range(n_features):
                                feature_names.append(
                                    self.function_names[i](*[input_features[j] for j in c]) + input_features[jj] + "_"
                                    + derivative_string(self.multiindices[k]))
        return feature_names

    @x_sequence_or_item
    def fit(self, x_full, y=None):
        """Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Measurement data.

        Returns
        -------
        self : instance
        """
        n_features = x_full[0].shape[x_full[0].ax_coord]
        if float(__version__[:3]) >= 1.0:
            self.n_features_in_ = n_features
        else:
            self.n_input_features_ = n_features

        n_output_features = 0

        # Count the number of non-derivative terms
        for f in self.functions:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_features, n_args, self.interaction_only))
            )

        if self.grid_ndim != 0:
            # Add the mixed derivative library_terms
            if self.include_interaction:
                n_output_features += (
                        n_output_features * n_features * self.num_derivatives
                )
            # Add the pure derivative library terms
            n_output_features += n_features * self.num_derivatives

        # If there is a constant term, add 1 to n_output_features
        if self.include_bias:
            n_output_features += 1
        self.n_output_features_ = n_output_features

        # required to generate the function names
        self.get_feature_names()

        return self

    @x_sequence_or_item
    def transform(self, x_full):
        """Transform data to custom features

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape (n_samples, n_output_features)
            The matrix of features, where n_output_features is the number of
            features generated from applying the custom functions
            to the inputs.
        """
        check_is_fitted(self)

        xp_full = []
        for x in x_full:
            n_features = x.shape[x.ax_coord]
            num_basis_each = x.shape[1]
            xp = np.empty((self.K * (num_basis_each - self.burn_in), self.n_output_features_), dtype=x.dtype)

            # Extract the input features on indices in each domain cell
            self.x_k = [x[self.inds_k, i] for i in range(x.shape[1])] * self.K

            # library function terms
            n_library_terms = 0
            for f in self.functions:
                for c in self._combinations(
                        n_features, f.__code__.co_argcount, self.interaction_only
                ):
                    n_library_terms += 1
            # May need modification here!!
            library_functions = np.empty((self.K * (num_basis_each - self.burn_in), n_library_terms), dtype=x.dtype)

            # Evaluate the functions on the indices of domain cells
            funcs = np.zeros((*x.shape[:-1], n_library_terms))
            func_idx = 0
            for f in self.functions:
                for c in self._combinations(n_features, f.__code__.co_argcount, self.interaction_only):
                    funcs[..., func_idx] = f(*[x[..., j] for j in c])
                    func_idx += 1
            if self.burn_in is not None:
                funcs = funcs[self.burn_in:, self.burn_in:]

            # library function terms
            for i in range(self.K):  # loop over domain cells
                for j in range(num_basis_each - self.burn_in):
                    idx = i * (num_basis_each - self.burn_in) + j
                    # calculate the integral feature by taking the dot product
                    # of the weights and functions over each axis
                    library_functions[idx] = np.array(
                        [np.trapz(self.weights[i].reshape(-1) * funcs[:, j, k], dx=self.dx)
                         for k in range(n_library_terms)])

            # # pure integral terms
            u = x_full[0]
            u_deriv = [np.zeros(u.shape) for i in range(self.derivative_order)]
            for i in range(u.shape[1]):
                derivs_sub = compute_derivatives(u[:, i], self.dx, [0, 0, 0, 0])
                for j in range(self.derivative_order):
                    u_deriv[j][:, i] = derivs_sub[j]

            if self.burn_in is not None and self.initial is None:
                self.initial = [u[self.burn_in]]
                for i in range(self.derivative_order-1):
                    self.initial.append(u_deriv[i][self.burn_in])

            u_data = np.array(x_full[0][self.burn_in:, self.burn_in:][:, :, 0])

            if self.derivative_order != 0:
                library_integrals = np.empty(
                    (self.K * (num_basis_each - self.burn_in), n_features * self.num_derivatives), dtype=x.dtype)

                # # pure integral terms
                for i in range(self.K):  # loop over domain cells
                    for j in range(num_basis_each - self.burn_in):
                        idx = i * (num_basis_each - self.burn_in) + j
                        try:
                            Fs = np.trapz(self.weights[i].reshape(-1) * u_data[:, j], dx=self.dx)
                        except IndexError:
                            print("Index Error")
                        library_idx = 0
                        for k in range(self.num_derivatives):  # loop over derivatives
                            # Calculate the integral feature by taking the dot product
                            # of the weights and data x_k over each axis.
                            # Integration by parts gives power of (-1).
                            Fs = self.s[i] * Fs - self.initial[k][j + self.burn_in]

                            library_integrals[idx, library_idx: library_idx + n_features] = Fs
                            library_idx += n_features

                # # Mixed derivative/non-derivative terms
                if self.include_interaction:
                    library_mixed_integrals = np.empty(
                        (self.K * (num_basis_each - self.burn_in),
                            n_library_terms * n_features * self.num_derivatives,), dtype=x.dtype,)

                    # Below we integrate the product of function and feature
                    # derivatives against the derivatives of phi to calculate the weak
                    # features. We cannot remove all derivatives of data in this case,
                    # but we can reduce the derivative order by half.

                    # Evaluate the functions on the indices of domain cells
                    funcs = np.zeros((*x.shape[:-1], n_library_terms))
                    func_idx = 0
                    for f in self.functions:
                        for c in self._combinations(
                                n_features, f.__code__.co_argcount, self.interaction_only
                        ):
                            funcs[..., func_idx] = f(*[x[..., j] for j in c])
                            func_idx += 1

                    # Calculate the necessary function and feature derivatives
                    # funcs_derivs = np.zeros(np.concatenate([[self.num_derivatives + 1], funcs[self.burn_in:].shape]))
                    x_derivs = np.zeros(
                        np.concatenate([[self.num_derivatives], x[self.burn_in:, self.burn_in:].shape]))
                    for i in range(self.derivative_order):
                        x_derivs[i] = u_deriv[i][self.burn_in:, self.burn_in:]
                    u = u[self.burn_in:, self.burn_in:]

                    # # mixed intergral terms
                    for i in range(self.K):  # loop over domain cells
                        for j in range(num_basis_each - self.burn_in):
                            idx = i * (num_basis_each - self.burn_in) + j
                            for k in range(self.num_derivatives):
                                for l in range(len(self.functions)):  # loop over derivatives
                                    funcs = u[:, j] ** (l + 1) * x_derivs[k, :, j]
                                    Fs = np.trapz(self.weights[i].reshape(-1) * funcs.reshape(-1), dx=self.dx)
                                    library_idx = k * len(self.functions) + l
                                    library_mixed_integrals[idx, library_idx: library_idx + 1] = Fs

            library_idx = 0
            # Constant term
            if self.include_bias:
                constants_final = np.zeros(self.K)
                for k in range(self.K):
                    constants_final[k] = np.sum(self.fullweights0[k])
                xp[:, library_idx] = constants_final
                library_idx += 1

            # library function terms
            xp[:, library_idx: library_idx + n_library_terms] = library_functions
            library_idx += n_library_terms

            if self.derivative_order != 0:
                # pure integral terms
                xp[:, library_idx: library_idx + self.num_derivatives * n_features] = library_integrals
                library_idx += self.num_derivatives * n_features

                # mixed function integral terms
                if self.include_interaction:
                    xp[:, library_idx: library_idx + n_library_terms * self.num_derivatives * n_features,] = library_mixed_integrals
                    library_idx += n_library_terms * self.num_derivatives * n_features

            xp_full = xp_full + [AxesArray(xp, {"ax_sample": 0, "ax_coord": 1})]
        if self.library_ensemble:
            xp_full = self._ensemble(xp_full)
        return xp_full

    def calc_x_dot(self, x):
        self.x_dot = self.convert_u_dot_integral(x)

    def calc_trajectory(self, diff_method, x, t):
        x_dot = self.convert_u_dot_integral(x)
        return AxesArray(x_dot, {"ax_sample": 0, "ax_coord": 1})

    def trapezoidal_laplace(self, y_t, weights):
        integrand = weights * y_t
        integral_approximation = np.trapz(integrand, dx=self.dx)
        return integral_approximation
