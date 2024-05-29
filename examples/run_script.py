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

from openmcmc.parameter_jax import Parameter_JAX, LinearCombinationDependent_JAX, Identity_JAX
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
class TestParameter(LinearCombinationDependent_JAX):
    """Test parameter class with a generic transformation function."""
    
    def update_prefactors(self, state: dict) -> dict:
        """Update the A matrix in the state, based on the current information."""
        state["A"] = self.plume(z=state["z"], x=state["x"], gam=state["gam"], wsp=state["wsp"])
        return state

    @staticmethod
    def plume(z: jnp.ndarray, x: jnp.ndarray, gam: jnp.ndarray, wsp: jnp.ndarray) -> jnp.ndarray:
        """Claculate a plume coupling."""
        dx = x[:, [0]] - z[:, [0]].T
        dy = x[:, [1]] - z[:, [1]].T
        dz = x[:, [2]] - z[:, [2]].T
        sig = jnp.tan(gam * jnp.pi / 180.0) * dx
        A = 1e6 / (2.0 * jnp.pi * sig * wsp * 0.67 * 3600) * jnp.exp(- 0.5 * jnp.power(dy / sig, 2)) * \
            jnp.exp(- 0.5 * jnp.power(dz / sig, 2))
        A = jnp.where(dx > 0.0, A, 0.0)
        A = jnp.where(dz > 0.0, A, 0.0)
        return A

"""
Initialise the state for the sampling
"""

# limits for source domain
lim = [[-0.01, 0.01],
       [-0.01, 0.01],
       [0, 1]]

# set up the problem
state = {}

# source locations
n_source = 1
z = np.zeros(shape=(n_source, 3))
for i in range(3):
    z[:, i] = np.random.uniform(lim[i][0], lim[i][1], n_source)
z[:, 2] = np.zeros(n_source)
state["z"] = jnp.array(z)

# sensor locations
n_sensor = 100
deg = np.atleast_2d(np.linspace(0, 2 * np.pi, n_sensor)).T
dist = 20.0 # distance in m from the centre
x = np.concatenate((dist * np.cos(deg), dist * np.sin(deg),
                    np.random.uniform(lim[2][0], lim[2][1], size=(n_sensor, 1))), axis=1)
state["x"] = jnp.array(x)

# wind sigma parameter
state["gam"] = jnp.array([[10.0]])

# wind speed
state["wsp"] = jnp.array([[3.0]])

# initialise parameter class
max_emis = 10.0
state["s"] = jnp.array(np.random.uniform(0, max_emis, size=(n_source, 1)))

# initialise the precision matrix
std_msr = 0.1
# state["Q"] = ((1.0 / std_msr) ** 2) * sparse_jax.eye(n_sensor)
state["Q"] = ((1.0 / std_msr) ** 2) * jnp.eye(n_sensor)

# prior for mu
state["mu_s"] = jnp.zeros(shape=(n_source, 1))
state["P_s"] = 0.0001 * jnp.eye(n_source)

"""
Create the parameter and the distribution
"""

# create the parameter class
param = TestParameter(form={"s": "A"})

# data
key = random.PRNGKey(0)
y_hat, state = param.predictor(state, update_state=True)
state["y"] = y_hat +  0.0 * random.normal(key, shape=(n_sensor, 1))

# plot the data
plt.figure()
plt.plot(deg * 180 / np.pi, y_hat, "-b")
plt.grid()
plt.show()

# create the distribution
dist = Normal_JAX(response="y", grad_list=["z", "s"], mean=param, precision=Identity_JAX(form="Q"))
dist.param_list = ["z", "s"]

"""
Set up the model and samplers.
"""

# initialise model
prior_s = Normal_JAX(response="s", grad_list=["s"], mean=Identity_JAX(form="mu_s"), precision=Identity_JAX(form="P_s"))
prior_s.param_list = ["s"]
mdl = Model([dist, prior_s])

# set up the samplers
sampler = [ManifoldMALA("z", mdl),
           NormalNormal("s", mdl)]
# sampler = [ManifoldMALA("z", mdl)]
sampler[0].max_variable_size = (n_source, 3)


"""
Checking that the JAX derivative agrees with finite difference.
"""

grad_z, hess_z = mdl["y"].grad_log_p(state, "z")

# do the grad explicitly in numpy
state_numpy = state.copy()
for key, val in state_numpy.items():
    state_numpy[key] = np.asarray(val).copy()
state_numpy["Q"] = ((1.0 / std_msr) ** 2) * np.eye(n_sensor)
logp_numpy, _ = mdl["y"].log_p(state_numpy)

step = 1e-5
grad_z_fd = np.zeros((n_source * 3, 1))
for i in range(n_source * 3):
    state_copy = state_numpy.copy()
    state_copy["z"] = state_copy["z"].copy()
    state_copy["z"][i // 3, i % 3] += step
    logp_copy, _ = mdl["y"].log_p(state_copy)
    grad_z_fd[i] = (logp_copy - logp_numpy) / step

def temp_grad_wrapper(state_in):
    log_p, state_in = mdl["y"].log_p(state_in, update_state=True)
    return log_p
grad_fun = sparse_jax.grad(temp_grad_wrapper, argnums=0)
state_copy = deepcopy(state)
# state_copy["Q"] = ((1.0 / std_msr) ** 2) * jnp.eye(n_sensor)
grad_dict = grad_fun(state_copy)
# NOTE (28/05/24): this call with a dict input didn't work when Q was still a sparse matrix. Thus it seems that the JAX
# standard grad won't work with the sparse arrays.

"""
Run the MCMC.
"""

# copy the state and move the initial source locations
state_init = deepcopy(state)
# state_init["z"] = jnp.array(np.random.uniform(-10, 10, size=(n_source, 3)))
# state_init["z"] += jnp.array(np.concatenate((np.random.normal(0, 1.0, size=(n_source, 1)),
#                                              np.random.normal(0, 1.0, size=(n_source, 1)),
#                                              np.random.normal(0, 0.1, size=(n_source, 1))), axis=1))
state_init["z"] = jnp.array(np.array([-1, -1, 0.01], ndmin=2))
# state_init["z"] = jnp.array(np.array([-5, -5, 0.01], ndmin=2))

# change the step size for the mMALA
sampler[0].step = 5.0e-1

# set up the MCMC object
mcmc = MCMC(state_init, sampler, model=mdl, n_burn=500, n_iter=1500)
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


"""
Next: try Maximum likelihood, see how that goes.
"""

state_ml = state_init.copy()

step = 0.01
diff = 1
count = 1
n_iter_max = 10000
stop = False
log_p_tracking = np.full(shape=(n_iter_max, ), fill_value=np.nan)
log_p_tracking[0], _ = mdl["y"].log_p(state_ml, update_state=True)
while not stop:
    grad_z = mdl["y"].grad_functions["z"](state_ml, state_ml["z"])
    grad_z = np.asarray(grad_z).reshape((grad_z.size, 1))
    hess_z = mdl["y"].hessian_functions["z"](state_ml, state_ml["z"])
    hess_z = - np.asarray(hess_z).reshape((grad_z.size, grad_z.size))
    # search_dir = np.linalg.solve(hess_z, grad_z).reshape(state_ml["z"].shape)
    search_dir = grad_z.reshape(state_ml["z"].shape) / 1e4
    state_ml["z"] = state_ml["z"] + step * jnp.array(search_dir)
    log_p, state_ml = mdl["y"].log_p(state_ml, update_state=True)
    log_p_tracking[count] = log_p
    diff = np.max(step * np.abs(search_dir))
    state_ml["s"] = jnp.linalg.solve(state_ml["A"].T @ state_ml["A"], state_ml["A"].T @ state_ml["y"])
    
    if np.abs(log_p_tracking[count] - log_p_tracking[count-1]) < 1e-3:
        stop = True

    count = count + 1

"""
What the hell! Make plots of the log-likelihood
"""

plot_lims = [[-10, 10],
             [-10, 10],
             [-2, 2]]

n_plot = 500
y_range = np.linspace(plot_lims[1][0], plot_lims[1][1], n_plot)
state_plot = deepcopy(state)
log_p_plot = np.full(shape=(n_plot, ), fill_value=np.nan)
grad_log_p_plot = np.full(shape=(3, n_plot), fill_value=np.nan)
for i in range(n_plot):
    state_plot["z"] = state_plot["z"].at[0, 1].set(y_range[i])
    log_p_plot[i], state_plot = mdl["y"].log_p(state_plot, update_state=True)
    grad_log_p_plot[:, i] = mdl["y"].grad_functions["z"](state_plot, state_plot["z"]).reshape((3, ))


plt.figure()
plt.plot(y_range, log_p_plot, "-r")
plt.grid()
plt.show()

plt.figure()
plt.plot(y_range, grad_log_p_plot[1, :], "-r")
plt.grid()
plt.show()

x_range = np.linspace(plot_lims[1][0], plot_lims[1][1], n_plot)
state_plot = deepcopy(state)
log_p_plot = np.full(shape=(n_plot, ), fill_value=np.nan)
grad_log_p_plot = np.full(shape=(3, n_plot), fill_value=np.nan)
for i in range(n_plot):
    state_plot["z"] = state_plot["z"].at[0, 0].set(x_range[i])
    log_p_plot[i], state_plot = mdl["y"].log_p(state_plot, update_state=True)
    grad_log_p_plot[:, i] = mdl["y"].grad_functions["z"](state_plot, state_plot["z"]).reshape((3, ))

plt.figure()
plt.plot(x_range, log_p_plot, "-r")
plt.grid()
plt.show()

plt.figure()
plt.plot(x_range, grad_log_p_plot[0, :], "-r")
plt.grid()
plt.show()

z_range = np.linspace(plot_lims[2][0], plot_lims[2][1], n_plot)
state_plot = deepcopy(state_init)
log_p_plot = np.full(shape=(n_plot, ), fill_value=np.nan)
grad_log_p_plot = np.full(shape=(3, n_plot), fill_value=np.nan)
for i in range(n_plot):
    state_plot["z"] = state_plot["z"].at[0, 2].set(z_range[i])
    log_p_plot[i], state_plot = mdl["y"].log_p(state_plot, update_state=True)
    grad_log_p_plot[:, i] = mdl["y"].grad_functions["z"](state_plot, state_plot["z"]).reshape((3, ))


plt.figure()
plt.plot(z_range, grad_log_p_plot[0, :], "-r")
plt.grid()
plt.show()
