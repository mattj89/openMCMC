"""Distributions for the JAX pyelq implementation."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Tuple, Union

import numpy as np
from scipy import sparse, stats

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import grad, jit, vmap
from jax import random
from jax import jacfwd, jacrev, hessian
from jax.experimental import sparse as sparse_jax

from openmcmc.parameter import Parameter
from openmcmc.parameter_jax import LinearCombination_jax
from openmcmc import gmrf


@dataclass
class Distribution_jax(ABC):
    """Abstract distribution class for the JAX case.

    Attributes:
        response (str): name of the response variable in the state dictionary.
        grad_list (list): list of parameter names for which gradients will be computed.
        param_list (list): list of all parameters in the conditioning set for the distribution.
        grad_functions (dict): dictionary of gradient functions with respect to each individual parameter in
            self.grad_list
        hessian_functions (dict): dictionary of hessian functions with respect to each individual parameter in
            self.grad_list.

    """

    response: str
    grad_list: list
    param_list: list = field(init=False)
    grad_functions: dict = field(init=False)
    hessian_functions: dict = field(init=False)

    @abstractmethod
    def log_p(self, state: dict) -> jnp.ndarray:
        """Evaluate the log-posterior distribution."""

    @abstractmethod
    def grad_log_p(self, state: dict, param: str) -> jnp.ndarray:
        """Evaluate the gradient of the log-posterior distribution."""


@dataclass
class Normal_jax(Distribution_jax):
    """Normal distribution with options to jit compile and use grad functionality from JAX.

    If self.jit_compile is True, a compiled version of the log-density function is created during initialisation and
    stored in self.log_p_jit. This version is then used for all future evaluations.

    Attributes:
        mean (LinearCombination_jax): mean parameter for the Normal distribution.
        precision (LinearCombination_jax): precision matrix parameter for the Normal distribution. TODO (07/11/25): this
            is currently unused- we just use a scalar instead.
        jit_compile (bool): whether to JIT compile the log-density function.
        scalar_precision (float): scalar precision value to use in place of a full precision matrix.
        domain_response_lower (Union[float, None]): lower truncation limit for the response variable.
        domain_response_upper (Union[float, None]): upper truncation limit for the response variable.

    """
    mean: LinearCombination_jax
    precision: LinearCombination_jax
    terms_in_likelihood: list
    jit_compile: bool = True
    scalar_precision: float = 1.0
    domain_response_lower: Union[float, None] = None
    domain_response_upper: Union[float, None] = None

    def __post_init__(self):
        """Post intialisation: set up the JAX gradient."""
        self.initialise_grad()
        if self.jit_compile:
            self.log_p_jit = jit(self.log_p_internal, static_argnums=(1,))

    def log_p_internal(self, state: dict, update_index: int = None) -> Tuple[jnp.ndarray, dict]:
        """Evaluate the log-posterior distribution.

        This is the internal definition of the log-density function. If self.jit_compile is True, this function will be
        JIT compiled and stored in self.log_p_jit.

        NOTE (07/11/25): this is currently set up to simply use a scalar precision value, for speed & to avoid having
        to combine JAX and scipy.sparse operations. Should we re-do this to include JAX sparse functionality?

        Args:
            state (dict): dictionary containing current parameter information.
            update_index (int): index of the parameter to be updated. This information is passed down to the mean
                predictor, where it may inform which elements of a coupling matrix need to be updated.

        Returns:
            log_p (jnp.ndarray): log-posterior density value.
            state (dict): updated state dictionary.

        """
        mean, state = self.mean.predictor(state, update_index=update_index)
        exponent_term = jnp.vdot(state[self.response] - mean, state[self.response] - mean) * self.scalar_precision
        log_p = 0.5 * (state[self.response].shape[0] *
                        (jnp.log(self.scalar_precision) - jnp.log(2 * jnp.pi)) - exponent_term)
        if self.domain_response_lower is not None:
            norm_const = 1.0 - jsp.stats.norm.cdf(
                self.domain_response_lower, loc=mean, scale=jnp.sqrt(1.0 / self.scalar_precision)
            )
            log_p -= jnp.log(norm_const) * state[self.response].shape[0]
            # TODO (04/11/25): hard-coded for the shape of s as a test.
        return log_p, state

    def log_p(self, state: dict, update_index: bool = None) -> Tuple[jnp.ndarray, dict]:
        """Evaluate the log-posterior distribution (jit compiled if requested).

        If self.jit_compile is True, the JIT compiled version of the log-density function is used. Otherwise, the
        non-compiled function self.log_p_internal is used.

        Args:
            state (dict): dictionary containing current parameter information.
            update_index (int): index of the parameter to be updated. This information is passed down to the mean
                predictor, where it may inform which elements of a coupling matrix need to be updated.

        Returns:
            log_p (jnp.ndarray): log-posterior density value.
            state (dict): updated state dictionary.

        """
        likelihood_state = self.make_likelihood_state(state)
        if self.jit_compile:
            log_p, likelihood_state = self.log_p_jit(likelihood_state, update_index)
        else:
            log_p, likelihood_state = self.log_p_internal(likelihood_state, update_index=update_index)
        if "A" in self.terms_in_likelihood:
            state["A"] = likelihood_state["A"]
        return log_p, state

    def initialise_grad(self):
        """Initialise the JAX gradient functions of the log-likelihood.

        The traced JAX grad of a wrapper function is defined, such that we can obtain gradients of the log-likelihood
        with respect to a specific sub-set of the state parameters- this avoids the need to compute gradients wrt all
        state variables (which would result in wasted computation).

        Attaches self.grad_functions and self.hessian_functions as attributes of the class, which are dictionaries
        containing the JIT-compiled gradient and hessian functions for each parameter in self.grad_list.

        """
        self.grad_functions = {}
        self.hessian_functions = {}
        for param in self.grad_list:
            def temp_log_p(state: dict, grad_value: jnp.ndarray, update_index: int, grad_name: str = param) -> jnp.ndarray:
                state_copy = state.copy()
                state_copy[grad_name] = grad_value
                log_p, state_copy = self.log_p_internal(state_copy, update_index=update_index)
                return log_p
            self.grad_functions[param] = jit(sparse_jax.grad(temp_log_p, argnums=1), static_argnums=(2, 3))
            self.hessian_functions[param] = jit(hessian(temp_log_p, argnums=1), static_argnums=(2, 3))

    def grad_log_p(self, state: dict, param: str, update_index: int = None, hessian_required: bool = True) -> jnp.ndarray:
        """Evaluate the gradient of the log-posterior distribution.

        The gradients & hessians are calculated using the pre-computed JAX grad functions stored in self.grad_functions.
        Outputs are converted back to np.ndarray for compatibility with the rest of the codebase.

        Args:
            state (dict): dictionary containing current parameter information.
            param (str): name of the parameter with respect to which the gradient is computed.
            update_index (int): index of the parameter to be updated. This information is passed down to the mean
                predictor, where it may inform which elements of a coupling matrix need to be updated.
            hessian_required (bool): whether to also compute and return the hessian matrix.

        Returns:
            grad_log_p (np.ndarray): gradient of the log-posterior
            hess_log_p (np.ndarray, optional): hessian of the log-posterior (if hessian_required is True).

        """
        likelihood_state = self.make_likelihood_state(state)
        grad_log_p = self.grad_functions[param](likelihood_state, likelihood_state[param], update_index)
        grad_log_p = np.asarray(grad_log_p).reshape((likelihood_state[param].size, 1))
        if hessian_required:
            hess_log_p = self.hessian_functions[param](likelihood_state, likelihood_state[param], update_index)
            hess_log_p = -np.asarray(hess_log_p).reshape((likelihood_state[param].size, likelihood_state[param].size))
            # hess_log_p = np.diag(np.abs(np.diag(hess_log_p)))
            # TODO (07/11/25): is there a better solution for the Hessian? Or should we just eliminate cases where this
            # is required for now?
            return grad_log_p, hess_log_p
        else:
            return grad_log_p
        
    def make_likelihood_state(self, state: dict):
        """Prepare state for jit and grad operations."""
        likelihood_state = {}
        for param in self.terms_in_likelihood:
            likelihood_state[param] = state[param]
        if self.response == "y":
            likelihood_state["y"] = state["y"] - (state["B_bg"] @ state["bg"])
        return likelihood_state

    def conditional_precision(self, state: dict, param: str) -> np.ndarray:
        """get the conditional precision matrix for NormalNormal updates.
        TODO (17/06/25): used to do this based on the grad.

        TODO (07/11/25): is this even used any more?
        """
        precision, _ = self.precision.predictor(state)
        if isinstance(self.mean, LinearCombination_jax):
            scale_matrix = state[self.mean.form[param]]
            BtQB = scale_matrix.T @ (precision @ scale_matrix)
            if isinstance(BtQB, jax.Array):
                return np.asarray(BtQB)
            else:
                return BtQB
        else:
            return precision

    def rvs(self, state: dict, n: int = 1) -> np.ndarray:
        """Generate a random sample from the specified distribution.

        Args:
            state (dict): dictionary containing current parameter information.
            n (int): number of samples to generate.

        Returns:
            samples (np.ndarray): generated samples. shape=(p, n).

        """
        mean, _ = self.mean.predictor(state)
        precision, _ = self.precision.predictor(state)
        return gmrf.sample_normal(mu=mean, Q=precision, n=n)


@dataclass
class Uniform_jax(Distribution_jax):
    """Uniform dist."""
    domain_response_lower: Union[float, np.ndarray] = 0.0
    domain_response_upper: Union[float, np.ndarray] = 1.0

    def domain_range(self, state) -> np.ndarray:
        """Get the domain range (upper-lower) from domain_limits.

        Args:
            state (dict): dictionary with current state information.

        Returns:
            (np.ndarray): domain range. shape=(p, 1).

        """
        d = state[self.response].shape[0]
        domain_range = self.domain_response_upper - self.domain_response_lower
        if domain_range.size == 1:
            domain_range = np.ones((d, 1)) * domain_range
        return domain_range

    def log_p(self, state: dict, by_observation: bool = False) -> Union[np.ndarray, float]:
        """Evaluate log density."""
        n = state[self.response].shape[1]
        log_p = -np.sum(np.log(self.domain_range(state)))
        if by_observation:
            log_p = np.ones(n) * log_p
        else:
            log_p = n * log_p
        return log_p, state

    def grad_log_p(
        self, state: dict, param: str, hessian_required: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Evaluate trivial grad & hessian of log p."""
        grad = jnp.zeros(shape=(state[param].size, 1))
        if hessian_required:
            hess = jnp.zeros(shape=(state[param].size, state[param].size))
            return grad, hess
        return grad

    def rvs(self, state: dict, n: int = 1) -> np.ndarray:
        standard_unif = np.random.rand(state[self.response].shape[0], n)
        return jnp.array(self.domain_response_lower + self.domain_range(state) * standard_unif)