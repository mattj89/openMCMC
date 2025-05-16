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
            jnp.ndarray:
            dict:
        """
        if update_index is not None:
            state = self.update_prefactors(state, update_index=update_index)
        return self.predictor_conditional(state), state

    @abstractmethod
    def update_prefactors(self, state: dict) -> None:
        """Method to update the prefactor matrices of the linear combincation.

        Args:
            state (dict): state dictionary.

        Returns:
            state (dict): state dictionary with updated prefactor matrices.

        """
