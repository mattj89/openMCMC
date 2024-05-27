"""Run script for testing the new JAX distribution."""

import numpy as np
from scipy import sparse
from typing import Union
from dataclasses import dataclass

import jax.numpy as jnp
from jax.experimental import sparse as sparse_jax
from jax import grad, jit, vmap
from jax import random
from jax import jacfwd, jacrev
from jax.test_util import check_grads

from openmcmc.parameter_jax import Parameter_JAX, Identity
from openmcmc.distribution.distribution_jax import Distribution_JAX, Normal_JAX
from openmcmc.model import Model
from openmcmc.sampler.sampler import NormalNormal
from openmcmc.sampler.metropolis_hastings import ManifoldMALA
from openmcmc.mcmc import MCMC

"""
Set up a test parameter class.
"""

@dataclass
class TestParameter(Parameter_JAX):
    """Test parameter class with a generic transformation function."""

    form: dict

    def predictor(self, state: dict) -> jnp.ndarray:
        """Evaluate predictor."""
        return self.plume(z=state["z"], x=state["x"], gam=state["gam"], wsp=state["wsp"]) @ state["s"]
    
    def predictor_conditional(self, state: dict, term_to_exclude: Union[str, list] = None) -> jnp.ndarray:
        """For the linear Gaussian case (NormalNormal sampler). Turn off terms in the predictor.
        
        TODO (22/05/24): work out a way to combine this sort of functionality with the existing LinearCombination case.
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

    @staticmethod
    def plume(z: jnp.ndarray, x: jnp.ndarray, gam: jnp.ndarray, wsp: jnp.ndarray) -> jnp.ndarray:
        """Claculate a plume coupling."""
        dx = x[:, [0]] - z[:, [0]].T
        dy = x[:, [1]] - z[:, [1]].T
        dz = x[:, [2]] - z[:, [2]].T
        sig = jnp.tan(gam * 180.0 / jnp.pi) * dx
        A = 1.0 / (2.0 * jnp.pi * sig * wsp) * jnp.exp(- 0.5 * jnp.power(dy / sig, 2)) * \
            jnp.exp(- 0.5 * jnp.power(dz / sig, 2))
        return A

"""
Initialise the state for the sampling
"""

# limits for source domain
lim = [[-10, 10],
       [-10, 10],
       [0, 5]]

# set up the problem
state = {}

# source locations
n_source = 2
z = np.zeros(shape=(n_source, 3))
for i in range(3):
    z[:, i] = np.random.uniform(lim[i][0], lim[i][1], n_source)
state["z"] = jnp.array(z)

# sensor locations
n_sensor = 200
x = np.concatenate((20.0 * np.ones((n_sensor, 1)),
                    np.atleast_2d(np.linspace(-10, 10, n_sensor)).T,
                    np.zeros((n_sensor, 1))), axis=1)
state["x"] = jnp.array(x)

# wind sigma parameter
state["gam"] = jnp.array([[1.0]])

# wind speed
state["wsp"] = jnp.array([[3.0]])

# initialise parameter class
max_emis = 10.0
state["s"] = jnp.array(np.random.uniform(0, max_emis, size=(n_source, 1)))

# initialise the precision matrix
std_msr = 0.00001
state["Q"] = ((1.0 / std_msr) ** 2) * jnp.eye(n_sensor)

# prior for mu
state["mu_s"] = jnp.zeros(shape=(n_source, 1))
state["P_s"] = 0.0001 * jnp.eye(n_source)

"""
Create the parameter and the distribution
"""

# create the parameter class
param = TestParameter(form={"s": "A"})

# add the plume matrix to the state
state["A"] = param.plume(state["z"], state["x"], state["gam"], state["wsp"])

# data
key = random.PRNGKey(0)
state["y"] = param.predictor(state) +  std_msr * random.normal(key, shape=(n_sensor, 1))

# create the distribution
dist = Normal_JAX(response="y", grad_list=["z", "s"], mean=param, precision=Identity(form="Q"))
dist.param_list = ["z", "s"]

# dL_dz, d2L_dz2 = dist.grad_log_p(state, param="z", hessian_required=True)

# np_hess = np.asarray(d2L_dz2)
# np.linalg.det(-np_hess.reshape((n_source * 3, n_source * 3)))

# dL_ds, d2L_ds2 = dist.grad_log_p(state, param="s", hessian_required=True)
# A = param.plume(state["z"], state["x"], state["gam"], state["wsp"])
# real_Hs = A.T @ state["Q"] @ A

"""
Set up the model and samplers.
"""

# initialise model
prior_s = Normal_JAX(response="s", grad_list=["s"], mean=Identity(form="mu_s"), precision=Identity(form="P_s"))
prior_s.param_list = ["s"]
mdl = Model([dist, prior_s])

# set up the samplers
sampler = [ManifoldMALA("z", mdl),
           NormalNormal("s", mdl)]
sampler[0].max_variable_size = (n_source, 3)


"""
Checking that the JAX derivative agrees with finite difference.
"""

grad_z, hess_z = mdl["y"].grad_log_p(state, "z")

# do the grad explicitly in numpy
state_numpy = state.copy()
for key, val in state_numpy.items():
    state_numpy[key] = np.asarray(val).copy()

step = 1e-4
grad_z_fd = np.zeros((n_source * 3, 1))
for i in range(n_source * 3):
    state_copy = state_numpy.copy()
    state_copy["z"] = state_copy["z"].copy()
    state_copy["z"][i // 3, i % 3] += step
    grad_z_fd[i] = (np.asarray(mdl["y"].log_p(state_copy)) - np.asarray(mdl["y"].log_p(state_numpy))) / step

grad_fun = grad(mdl["y"].log_p, argnums=0)
grad_dict = grad_fun(state)

# def temp_log_p(state: dict, grad_value: jnp.ndarray) -> jnp.ndarray:
#     state_copy = state.copy()
#     state_copy["z"] = grad_value
#     return mdl["y"].log_p(state_copy)
# check_grads(temp_log_p, (state, state["z"]), order=1)
# step = 1e-3
# grad_diff = np.zeros((n_source * 3, 1))
# for i in range(n_source * 3):
#     grad_plus = temp_log_p(state=state, grad_value=state["z"].at[i % n_source, i // n_source].add(step))
#     grad_centre = temp_log_p(state=state, grad_value=state["z"])
#     grad_diff[i] = (grad_plus - grad_centre) / step

# grad_z_fd = np.zeros((n_source * 3, 1))
# for i in range(n_source * 3):
#     state_copy = state.copy()
#     state_copy["z"] = state["z"].at[i % n_source, i // n_source].add(1e-4)
#     grad_z_fd[i] = (mdl["y"].log_p(state_copy) - mdl["y"].log_p(state)) / 1e-4
#     # for j in range(n_source * 3):

# set up the MCMC object
# mcmc = MCMC(state, sampler, model=mdl, n_burn=100, n_iter=100)
# mcmc.run_mcmc()

# TODO (16/05/24): NormalNormal case runs up to predictor_conditional- need to implement this bit, and then run again.

# NOTE (16/05/24): Don't seem to have functionality for sparse matrix cholesky factorization in JAX yet. Check whether
# everything still works when we just use scipy.sparse matrices instead.

# mcmc.state["z"]