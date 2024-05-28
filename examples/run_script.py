"""Run script for testing the new JAX distribution."""

import numpy as np
from scipy import sparse
from typing import Union
from dataclasses import dataclass
from copy import deepcopy

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

import matplotlib.pyplot as plt

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
        sig = jnp.tan(gam * jnp.pi / 180.0) * dx
        A = 1e6 / (2.0 * jnp.pi * sig * wsp * 0.67) * jnp.exp(- 0.5 * jnp.power(dy / sig, 2)) * \
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
n_sensor = 20
x = np.concatenate((20.0 * np.ones((n_sensor, 1)),
                    np.atleast_2d(np.linspace(-10, 10, n_sensor)).T,
                    np.random.uniform(lim[2][0], lim[2][1], size=(n_sensor, 1))), axis=1)
state["x"] = jnp.array(x)

# wind sigma parameter
state["gam"] = jnp.array([[5.0]])

# wind speed
state["wsp"] = jnp.array([[3.0]])

# initialise parameter class
max_emis = 10.0
state["s"] = jnp.array(np.random.uniform(0, max_emis, size=(n_source, 1)))

# initialise the precision matrix
std_msr = 100.0
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
# sampler = [ManifoldMALA("z", mdl),
#            NormalNormal("s", mdl)]
sampler = [ManifoldMALA("z", mdl)]
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

"""
Run the MCMC.
"""

# copy the state and move the initial source locations
state_init = deepcopy(state)
# state_init["z"] = jnp.array(np.random.uniform(-10, 10, size=(n_source, 3)))
state_init["z"] += jnp.array(np.concatenate((np.random.normal(0, 1.0, size=(n_source, 1)),
                                             np.random.normal(0, 1.0, size=(n_source, 1)),
                                             np.random.normal(0, 0.5, size=(n_source, 1))), axis=1))

# change the step size for the mMALA
sampler[0].step = 2.0e-1

# set up the MCMC object
mcmc = MCMC(state_init, sampler, model=mdl, n_burn=10, n_iter=10000)
mcmc.run_mcmc()

# NOTE (16/05/24): Don't seem to have functionality for sparse matrix cholesky factorization in JAX yet. Check whether
# everything still works when we just use scipy.sparse matrices instead.

# plot the results for the source locations
state["z"]
mcmc.state["z"]
state_init["z"]

plt.figure()
for i in range(n_source):
    plt.plot(mcmc.store["z"][i, 0, :].flatten(), mcmc.store["z"][i, 1, :].flatten(), label="Sampled")
    plt.plot(state["z"][i, 0], state["z"][i, 1], "rx", label="True")
plt.grid()
plt.show()

"""
Notes (27/05/24):
    - In general, the approach seems to work (and JAX seems to be pretty speedy at computing the gradients).
    - There is certainly an issue with doing m-MALA:
        * Hessians are commonly not positive definite (-ve definite).
        * Regularization by dropping -ve eigenvalues is relatively straightforward.
        * But even after this, the sampler is SUPER sensitive to step size.
    - MALA is better, but still sensitive to step size.
    - Maybe we should eventually consider adaptive step-size.

For the JAX distributions/parameters:
    - We need to set up a version of the LinearCombination case that can separately handle the Gaussian and
        non-Gaussian cases.

IDEA (think about whether it works):
    - Push anything to do with the state udating in the sampler down to the parameter layer.
    - When we evaluate dist.log_p(), we have an input flag that allows us to update the state.
    - When we evalues param.predictot(), the above flag also gets passed down.
    - Then in the parameter class, we can implement bespoke functionality for what happens when we evaluate the density
        -> i.e. we update the values of any associated parameters.
    - Can we handle the reversible jump situation also in this way? Would need to pass some information down about
        columns to add/delete if so.
    
"""