"""Alternative version of the parameter class, which uses JAX API to take advantage of automatic differentiation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy import sparse

import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import jacfwd, jacrev


@dataclass
class Parameter_JAX(ABC):
    """JAX version of the abstract parameter class."""

    @abstractmethod
    def predictor(self, state: dict) -> jnp.ndarray:
        """Function which evaluates the predictor function."""
    

@dataclass
class Identity(Parameter_JAX):
    """Un-transformed parameter case."""
    form: str

    def predictor(self, state: dict) -> jnp.ndarray:
        return state[self.form]
    

@dataclass
class LinearCombination(Parameter_JAX):
    """Matrix-vector multiplication parameter class."""
    form: dict

    def predictor(self, state: dict) -> jnp.ndarray:
        return self.predictor_conditional(state)
    
    def predictor_conditional(self, state: dict, term_to_exclude: Union[str, list] = None) -> jnp.ndarray:
        if term_to_exclude is None:
            term_to_exclude = []

        if isinstance(term_to_exclude, str):
            term_to_exclude = [term_to_exclude]

        sum_terms = 0
        for prm, prefactor in self.form.items():
            if prm not in term_to_exclude:
                sum_terms += state[prefactor] @ state[prm]
        return sum_terms
    

@dataclass
class Transformed(Parameter_JAX):
    """Class for a generic functional transformation of the state."""
    func: callable # TODO (16/05/24): should be a function with jax.numpy components

    def predictor(self, state: dict) -> jnp.ndarray:
        return self.func(state)
    
