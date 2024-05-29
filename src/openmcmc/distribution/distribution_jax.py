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

from openmcmc.parameter_jax import Parameter_JAX, Identity_JAX

from openmcmc.gmrf import sparse_cholesky

@dataclass
class Distribution_JAX(ABC):
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
class Normal_JAX(Distribution_JAX):
    """Normal distribution class."""
    
    mean: Parameter_JAX
    precision: Parameter_JAX
    domain_response_lower: Union[float, None] = None
    domain_response_upper: Union[float, None] = None

    def __post_init__(self):
        """Post intialisation: set up the JAX gradient."""
        self.initialise_grad()
    
    def log_p(self, state: dict, update_state: bool = True) -> Tuple[jnp.ndarray, dict]:
        """Evaluate the log-posterior distribution.
        TODO (16/05/24): check whether we can throw in a standard sparse matrix and not affect the JAX autograd.
        """
        mean, state = self.mean.predictor(state, update_state=update_state)
        precision, state = self.precision.predictor(state, update_state=update_state)
        # chol_precision = jnp.linalg.cholesky(precision)
        exponent_term = jnp.vdot(state[self.response] - mean, precision @ (state[self.response] - mean))
        # exponent_term = jnp.sum(jnp.power(chol_precision @ (state[self.response] - mean), 2.0))
        log_det_precision = 1.0 # TODO (28/05/24): solve this
        # log_det_precision = 2 * jnp.sum(jnp.log(chol_precision.diagonal()))
        log_p = 0.5 * (log_det_precision - mean.shape[0] * jnp.log(2 * jnp.pi) - exponent_term)
        return log_p, state
    
    def initialise_grad(self):
        """Initialise the gradient function.
        
        TODO (23/05/24): this works, but we can for sure design the function better if we decide to commit
        to the JAX backend.
        """
        self.grad_functions = {}
        self.hessian_functions = {}
        for param in self.grad_list:
            def temp_log_p(state: dict, grad_value: jnp.ndarray, grad_name: str = param) -> jnp.ndarray:
                state_copy = state.copy()
                state_copy[grad_name] = grad_value
                log_p, state_copy = self.log_p(state_copy, update_state=True)
                return log_p
            self.grad_functions[param] = jit(sparse_jax.grad(temp_log_p, argnums=1))
            # self.hessian_functions[param] = jit(hessian(temp_log_p, argnums=1))
            self.hessian_functions[param] = jit(sparse_jax.jacfwd(sparse_jax.jacrev(temp_log_p, argnums=1), argnums=1))
    
    def grad_log_p(self, state: dict, param: str, hessian_required: bool = True) -> jnp.ndarray:
        """Evaluate the gradient of the log-posterior distribution."""
        grad_log_p = self.grad_functions[param](state, state[param])
        grad_log_p = np.asarray(grad_log_p).reshape((state[param].size, 1))
        if hessian_required:
            hess_log_p = self.hessian_functions[param](state, state[param])
            hess_log_p = -np.asarray(hess_log_p).reshape((state[param].size, state[param].size))
            # hess_log_p = np.eye(grad_log_p.size)
            hess_log_p = self.ensure_hessian_positive_def(hess_log_p, eig = True)
            return grad_log_p, hess_log_p
        else:
            return grad_log_p
        
    def ensure_hessian_positive_def(self, hess_log_p: np.ndarray, eig: bool = True) -> np.ndarray:
        """Enforce positive definiteness of the Hessian."""
        if eig:
            hess_eig = np.linalg.eig(hess_log_p)
            eig_positive = np.maximum(hess_eig.eigenvalues, 1e0)
            hess_recon = hess_eig.eigenvectors @ np.diag(eig_positive) @ hess_eig.eigenvectors.T
        else:
            hess_diag = hess_log_p.diagonal()
            hess_diag = 1e5 * np.ones_like(hess_diag)
            hess_recon = np.diag(np.maximum(hess_diag, 1e0))
        return hess_recon
    
    def precision_conditional(self, state: dict, param: str):
        """For the NormalNormal sampler case."""
        precision, state = self.precision.predictor(state, update_state=False)
        prefactor = state[self.mean.form[param]]
        return np.asarray(prefactor.T @ (precision @ prefactor))