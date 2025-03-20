"""Compute the reference balanced state of the shallow water model."""
from __future__ import annotations

import os

import fridom.shallowwater as sw
import numpy as np
from mpi4py import MPI
from sweepexp import SweepExpMPI, log

import obta_paper

# constant
RESOLUTION_FACTOR = 9
INERTIAL_PERIOD = 2 * np.pi

# Set only one gpu per rank visible
n_gpus_per_node = 4
gpu_id = MPI.COMM_WORLD.Get_rank() % n_gpus_per_node
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Set the log level of the sweep to debug
log.setLevel("DEBUG")

# Define the experiment function

def compute_reference_balanced_states(initial_condition: str,
                                      rossby_number: float) -> dict:
    """
    Projection error of equidistant time averaging.

    Parameters
    ----------
    initial_condition : str
        The initial condition to use. Either "jet" or "random".
    rossby_number : float
        The advection scale of the shallow water model.

    """
    # compute the grid resolution
    lx = ly = 2 * np.pi
    nx = ny = 2**RESOLUTION_FACTOR - 1
    dt = 1 / nx

    # construct grid and modelsettings
    grid = sw.grid.cartesian.Grid(N=(nx, ny), L=(lx, ly), periodic_bounds=(True, True))
    mset = sw.ModelSettings(grid, f0=1.0, beta=0.0, csqr=1.0, Ro=rossby_number).setup()
    mset.time_stepper.dt = dt

    # construct balancing methods
    geo_proj = sw.projection.GeostrophicSpectral(mset, use_discrete=True)
    optimal_balance = sw.projection.OptimalBalance(
        mset=mset,
        base_proj=geo_proj,
        ramp_period=INERTIAL_PERIOD/rossby_number,
        ramp_type="exp",
        max_it=10,
        stop_criterion=0,  # make sure to reach the maximum number of iterations
    )

    # create the initial condition
    if initial_condition == "jet":
        z_ini = obta_paper.initial_conditions.ShallowWaterJet(mset)
    elif initial_condition == "random":
        z_ini = obta_paper.initial_conditions.ShallowWaterRandom(mset)
    else:
        msg = "Unknown initial condition."
        raise ValueError(msg)

    # compute the balanced state
    z_bal = optimal_balance(z_ini)

    z_bal.to_netcdf(f"data/reference/balanced_sw_{initial_condition}_{rossby_number:.2f}.nc")

    return {}

# Define the sweep
sweep = SweepExpMPI(
    func = compute_reference_balanced_states,
    parameters = {
        "initial_condition": ["jet", "random"],
        "rossby_number": [0.05, 0.1, 0.3],
    },
    return_values = {},
)

sweep.timeit = True
sweep.enable_priorities = True

sweep.priority.data += (1/sweep.parameters["rossby_number"]).astype(int)

# Run the sweep
sweep.run()
