"""Test the sensitivity of the ramp period to the diagnosed imbalance in sw model."""
from __future__ import annotations

import os

import fridom.shallowwater as sw
import numpy as np
from mpi4py import MPI
from sweepexp import SweepExpMPI, log

import obta_paper

# constant
RESOLUTION_FACTOR = 9

# Set only one gpu per rank visible
n_gpus_per_node = 4
gpu_id = MPI.COMM_WORLD.Get_rank() % n_gpus_per_node
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Set the log level of the sweep to debug
log.setLevel("DEBUG")

# Define the experiment function

def compute_imbalance(initial_condition: str,
              ramp_period: float,
              rossby_number: float) -> dict:
    """
    Diagnosed imbalance vs ramp period in the shallow water model.

    Parameters
    ----------
    initial_condition : str
        The initial condition to use. Either "jet" or "random".
    ramp_period : float
        The ramp period of the optimal balance.
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
        ramp_period=ramp_period,
        ramp_type="exp",
        max_it=3,
    )

    # create the initial condition
    if initial_condition == "jet":
        z_ini = obta_paper.initial_conditions.ShallowWaterJet(mset)
    elif initial_condition == "random":
        z_ini = obta_paper.initial_conditions.ShallowWaterRandom(mset)
    else:
        msg = "Unknown initial condition."
        raise ValueError(msg)

    imbalance = obta_paper.diagnosed_imbalance(
        mset=mset,
        state=z_ini,
        projector=optimal_balance,
        diagnosing_period=10.0/rossby_number,
    )

    return {"imbalance": imbalance}

# Define the sweep
sweep = SweepExpMPI(
    func = compute_imbalance,
    parameters = {
        "initial_condition": ["jet", "random"],
        "ramp_period": np.linspace(1, 10, 10),
        "rossby_number": [0.05, 0.1, 0.3],
    },
    return_values = {"imbalance": float},
    save_path = "data/ramp_period_sw.nc",
)

sweep.timeit = True
sweep.auto_save = True

# Run the sweep
sweep.run()
