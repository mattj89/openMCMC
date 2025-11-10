"""Alternative versions of the parameter class to be used with JAX pyELQ implementation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Tuple

import jax.numpy as jnp

from openmcmc.parameter import LinearCombination

@dataclass
class LinearCombination_jax(LinearCombination):
    """Matrix-vector multiplication parameter class."""
    form: dict

    def predictor(self, state: dict, update_index: int = None) -> Tuple[jnp.ndarray, dict]:
        """Predictor method with the option to update the prefactor matrices in the state.

        Args:
            state (dict): state dictionary.
            update_index (int): index of the parameter to update. Defaults to None.

        Returns:
            jnp.ndarray: predictor evaluated using the information in state.
            dict: state dictionary with updated prefactor matrices.

        """
        state = self.update_prefactors(state, update_index=update_index)
        return self.predictor_conditional(state), state

    def predictor_conditional(self, state, term_to_exclude = None):
        """Overloaded version, to take account of the fact that the terms are being screened in/out by the RJ
        indicator.

        TODO (21/10/21): do we even need to overload this now?

        Args:
            state (dict): state dictionary.
            term_to_exclude (Union[str, list], optional): term(s) to exclude from the linear combination.
                Defaults to None.

        Returns:
            sum_terms (jnp.ndarray): linear combination evaluated using the information in state.

        """
        if term_to_exclude is None:
            term_to_exclude = []

        if isinstance(term_to_exclude, str):
            term_to_exclude = [term_to_exclude]

        sum_terms = 0
        for prm, prefactor in self.form.items():
            if prm not in term_to_exclude:
                sum_terms += state[prefactor] @ state[prm]
        return sum_terms

    @abstractmethod
    def update_prefactors(self, state: dict, update_index: int) -> dict:
        """Method to update the prefactor matrices of the linear combincation.

        Args:
            state (dict): state dictionary.
            update_index (int): indices of e.g. the coupling matrix columns to update.

        Returns:
            state (dict): state dictionary with updated prefactor matrices.

        """
