"""Alternative version of the parameter class, which uses JAX API to take advantage of automatic differentiation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Tuple

import jax.numpy as jnp


@dataclass
class Parameter_jax(ABC):
    """JAX version of the abstract parameter class."""

    @abstractmethod
    def predictor(self, state: dict, update_state: bool = False) -> Tuple[jnp.ndarray, dict]:
        """Function which evaluates the predictor function."""
    

@dataclass
class Identity_jax(Parameter_jax):
    """Un-transformed parameter case."""
    form: str

    def predictor(self, state: dict, update_state: bool = False) -> Tuple[jnp.ndarray, dict]:
        return state[self.form], state
    

@dataclass
class LinearCombination_jax(Parameter_jax):
    """Matrix-vector multiplication parameter class."""
    form: dict

    def predictor(self, state: dict, update_state: bool = False) -> jnp.ndarray:
        return self.predictor_conditional(state), state
    
    def predictor_conditional(self, state: dict, term_to_exclude: Union[str, list] = None) -> jnp.ndarray:
        """Blah."""

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
class LinearCombinationDependent_jax(LinearCombination_jax):
    """Matrix-vector multiplication parameter class, where the prefactor matrix is dependent on other components of the
    state.

    This class is intended for cases where some of the parameters feature linearly in the predictor, and thus can be
    updated using conjugate sampling (e.g. NormalNormal), whereas others feature non-linearliy and are updated with
    Metropolis-Hastings-type methods (e.g. RandomWalk, ManifoldMALA).

    The user has a choice whether or not to update the other elements of the state when calculating the predictor. 
    
    """
    
    def predictor(self, state: dict, update_state: bool = True) -> Tuple[jnp.ndarray, dict]:
        """Evaluate the predictor function.
        
        Args:
            state: Current state of the model.
            update_state: Whether to update the state with information calculated in the function
        
        """
        if update_state:
            state = self.update_prefactors(state)
        return self.predictor_conditional(state), state
    
    @abstractmethod
    def update_prefactors(self, state: dict, **kwargs) -> dict:
        """Update the prefactor matrices."""


@dataclass
class Transformed(Parameter_jax):
    """Class for a generic functional transformation of the state."""

    def predictor(self, state: dict, update_state: bool = True) -> jnp.ndarray:
        """Predictor function which applies a generic transformation (implemented using JAX functionality in
        self.transformation()).
        
        Args:
            state (dict): Current parameter state.

        Returns:
            jnp.ndarray: Predictor as calculated by the transformation function.
        
        """
        return self.transformation(state)
    
    @abstractmethod
    def transformation(self, state: dict) -> jnp.ndarray:
        """Transformation to be applied to the state to get the predictor.
        
        Should be implemented in terms of jax.numpy functions, to enable grad() and jit() wrappers.

        Args:
            state (dict): Current parameter state.
        """