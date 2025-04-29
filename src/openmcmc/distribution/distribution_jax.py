"""JAX version of the distribution classes."""

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

from openmcmc.parameter_jax import Parameter_jax, Identity_jax

from openmcmc.gmrf import sparse_cholesky

@dataclass
class Distribution_jax(ABC):
    """Abstract distribution class for the JAX case."""
    
    response: str
    grad_list: list
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
    """Normal distribution implemented to use JAX funcitonality.
    
    Attributes:
        mean (Parameter_jax): parameter which determines the mean vector of the multivariate Normal distribution.
        precision (Parameter_jax): parameter which determines the precision matrix of the multivariate Normal
            distribution.
        domain_response_lower (Union[float, None]): lower bound of the response domain.
        domain_response_upper (Union[float, None]): upper bound of the response domain.

        fixed_precision (bool): flag which indicates whether the precision matrix will have a fixed value for the
            purpose of likelihood evaluations.
        log_det_precision (Union[float, None]): pre-calculated log-determinant of the precision matrix.
        
        TODO (12/06/24): Think about this- may be better to implement bespoke functionality in the parameter class, so
        we can deal with determinants of scaled matrix cases which are fixed up to a multiplicative constant.

    """
    
    mean: Parameter_jax
    precision: Parameter_jax
    domain_response_lower: Union[float, None] = None
    domain_response_upper: Union[float, None] = None

    fixed_precision: bool = False
    log_det_precision: Union[float, None] = None
    jit_compile: bool = True

    def __post_init__(self):
        """Post intialisation: set up the JAX gradient."""
        self.initialise_grad()
        if self.jit_compile:
            self.log_p_jit = jit(self.log_p_internal, static_argnums=1)

    def precompute_log_det_precision(self, state: dict):
        """Pre-compute the log-determinant of the precision matrix- for use in situations where it will not affect
        changes in likelihoods, so is essentially wasted computation.
        """
        self.fixed_precision = True
        precision, state = self.precision.predictor(state, update_state=False)
        chol_precision = np.linalg.cholesky(precision)
        self.log_det_precision = jnp.sum(jnp.log(jnp.diag(chol_precision)))
    
    def log_p_internal(self, state: dict, update_state: bool = True) -> Tuple[jnp.ndarray, dict]:
        """Evaluate the log-posterior distribution.

        Args:
            state (dict): dictionary containing current parameter infotmation.
            update_state (bool): flag indicating whether a state update is required (in cases where dependent parameters
                in the state will also change).

        """
        mean, state = self.mean.predictor(state, update_state=update_state)
        precision, state = self.precision.predictor(state, update_state=update_state)
        if self.fixed_precision:
            log_det_precision = self.log_det_precision
        else:
            log_det_precision = jnp.log(jnp.linalg.det(precision))
        exponent_term = jnp.vdot(state[self.response] - mean, precision @ (state[self.response] - mean))
        log_p = 0.5 * (log_det_precision - mean.shape[0] * jnp.log(2 * jnp.pi) - exponent_term)
        return log_p, state

    def log_p(self, state: dict, update_state: bool = True) -> Tuple[jnp.ndarray, dict]:
        """"""
        if self.jit_compile:
            log_p, state = self.log_p_jit(state)
        else:
            log_p, state = self.log_p_internal(state, update_state=update_state)
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
            def temp_log_p(state: dict, grad_value: jnp.ndarray, grad_name: str = param) -> jnp.ndarray:
                state_copy = state.copy()
                state_copy[grad_name] = grad_value
                log_p, state_copy = self.log_p_internal(state_copy, update_state=True)
                return log_p
            self.grad_functions[param] = jit(sparse_jax.grad(temp_log_p, argnums=1))
            self.hessian_functions[param] = jit(hessian(temp_log_p, argnums=1))
    
    def grad_log_p(self, state: dict, param: str, hessian_required: bool = True) -> jnp.ndarray:
        """Evaluate the gradient of the log-posterior distribution."""
        grad_log_p = self.grad_functions[param](state, state[param])
        grad_log_p = np.asarray(grad_log_p).reshape((state[param].size, 1))
        if hessian_required:
            hess_log_p = self.hessian_functions[param](state, state[param])
            hess_log_p = -np.asarray(hess_log_p).reshape((state[param].size, state[param].size))
            hess_log_p = self.ensure_hessian_positive_def(hess_log_p, eig = True)
            return grad_log_p, hess_log_p
        else:
            return grad_log_p
        
    def ensure_hessian_positive_def(self, hess_log_p: np.ndarray, eig: bool = True) -> np.ndarray:
        """Enforce positive definiteness of the Hessian."""
        if eig:
            hess_eig = np.linalg.eig(hess_log_p)
            # eig_positive = np.maximum(hess_eig.eigenvalues, 1e0)
            eig_positive = np.abs(hess_eig.eigenvalues)
            hess_recon = hess_eig.eigenvectors @ np.diag(eig_positive) @ hess_eig.eigenvectors.T
        else:
            hess_diag = hess_log_p.diagonal()
            hess_recon = np.diag(np.maximum(hess_diag, 1e0))
        return hess_recon
    
    def precision_conditional(self, state: dict, param: str):
        """For the NormalNormal sampler case."""
        precision, state = self.precision.predictor(state, update_state=False)
        prefactor = state[self.mean.form[param]]
        return np.asarray(prefactor.T @ (precision @ prefactor))
    
    def rvs(self, state: dict, n: int = 1) -> jnp.ndarray:
        """Random sampling for the JAX case."""
        # TODO (25/04/25): implemented by Copilot. Not run yet- check next week.
        mean, state = self.mean.predictor(state, update_state=False)
        precision, state = self.precision.predictor(state, update_state=False)
        chol_precision = np.linalg.cholesky(precision)
        standard_normal = np.random.randn(mean.shape[0], n)
        return mean + chol_precision @ standard_normal
    

@dataclass
class Uniform_jax(Distribution_jax):
    """Uniform distribution for the JAX cases."""

    domain_response_lower: Union[float, None] = None
    domain_response_upper: Union[float, None] = None
    
    def log_p(self, state: dict, update_state: bool = True, by_observation: bool = False) -> Tuple[jnp.ndarray, dict]:
        """Dummy log-likelihood evaluation function for JAX case."""
        n = state[self.response].shape[1]
        if by_observation:
            return jnp.zeros(n), state
        else:
            return 0.0, state
    
    def grad_log_p(self, state: dict, param: str, hessian_required: bool = True) -> jnp.ndarray:
        """Dummy gradient evaluation function for JAX case."""
        grad_log_p = jnp.zeros((state[param].size, 1))
        if hessian_required:
            hess_log_p = jnp.zeros((state[param].size, state[param].size))
            return grad_log_p, hess_log_p
        else:
            return grad_log_p
        
    def domain_range(self, state) -> np.ndarray:
        """Calculate the range of the domain of the response."""
        d = state[self.response].shape[0]
        domain_range = self.domain_response_upper - self.domain_response_lower
        if domain_range.size == 1:
            domain_range = np.ones((d, 1)) * domain_range
        return domain_range
        
    def rvs(self, state: dict, n: int = 1) -> jnp.ndarray:
        """Random sampling for the JAX case."""
        standard_unif = np.random.rand(state[self.response].shape[0], n)
        return self.domain_response_lower + self.domain_range(state) * standard_unif