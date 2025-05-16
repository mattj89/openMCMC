"""Distributions for the JAX pyelq implementation."""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Tuple, Union

import numpy as np
from scipy import sparse, stats

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import jacfwd, jacrev, hessian
from jax.experimental import sparse as sparse_jax

from openmcmc.parameter import Parameter


@dataclass
class Distribution_jax(ABC):
    """Abstract distribution class for the JAX case."""

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

    Attributes:

    """
    mean: Parameter
    precision: Parameter
    update_index_max: int
    jit_compile: bool = True
    fixed_precision: bool = True
    log_det_precision: float = 1.0
    domain_response_lower: np.ndarray = None
    domain_response_upper: np.ndarray = None

    def __post_init__(self):
        """Post intialisation: set up the JAX gradient."""
        self.initialise_grad()
        if self.jit_compile:
            self.log_p_jit = jit(self.log_p_internal, static_argnums=(1,))

    def log_p_internal(self, state: dict, update_index: int = 0) -> Tuple[jnp.ndarray, dict]:
        """Evaluate the log-posterior distribution.

        TODO: update_index set to default of 0 at the moment. Possible to have the default as nothing getting updated?

        Args:
            state (dict): dictionary containing current parameter infotmation.
            update_state (bool): flag indicating whether a state update is required (in cases where dependent parameters
                in the state will also change).

        """
        mean, state = self.mean.predictor(state, update_index=update_index)
        precision, state = self.precision.predictor(state)
        if self.fixed_precision:
            log_det_precision = self.log_det_precision
        else:
            log_det_precision = jnp.log(jnp.linalg.det(precision))
        # exponent_term = jnp.vdot(state[self.response] - mean, precision @ (state[self.response] - mean))
        exponent_term = jnp.vdot(state[self.response] - mean, state[self.response] - mean)
        log_p = 0.5 * (log_det_precision - mean.shape[0] * jnp.log(2 * jnp.pi) - exponent_term)
        return log_p, state

    def log_p(self, state: dict, update_index: bool = 0) -> Tuple[jnp.ndarray, dict]:
        """Evaluate the log-posterior distribution (jit compiled if requested).
        """
        if self.jit_compile:
            log_p, state = self.log_p_jit(state, update_index)
        else:
            log_p, state = self.log_p_internal(state, update_index=update_index)
        return log_p, state

    def initialise_grad(self):
        """Initialise the JAX gradient functions of the log-likelihood.

        The traced JAX grad of a wrapper function is defined, such that we can obtain gradients of the log-likelihood
        with respect to a specific sub-set of the state parameters- this avoids the need to compute gradients wrt all
        state variables (which would result in wasted computation).

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

    def grad_log_p(self, state: dict, param: str, update_index: int = 0, hessian_required: bool = True) -> jnp.ndarray:
        """Evaluate the gradient of the log-posterior distribution."""
        grad_log_p = self.grad_functions[param](state, state[param], update_index)
        grad_log_p = np.asarray(grad_log_p).reshape((state[param].size, 1))
        if hessian_required:
            hess_log_p = self.hessian_functions[param](state, state[param], update_index)
            hess_log_p = -np.asarray(hess_log_p).reshape((state[param].size, state[param].size))
            hess_log_p = np.diag(np.diag(hess_log_p)) # TODO: better!
            return grad_log_p, hess_log_p
        else:
            return grad_log_p