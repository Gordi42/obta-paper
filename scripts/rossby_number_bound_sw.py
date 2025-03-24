"""Compute the imbalance vs rossby number in the nonperiodic shallow water model."""
from __future__ import annotations

import os

import fridom.shallowwater as sw
import numpy as np
from mpi4py import MPI
from sweepexp import SweepExpMPI, log

import obta_paper

# constant
RESOLUTION_FACTOR = 9
INERTIAL_PERIOD = 2.0 * np.pi
RAMP_PERIOD = INERTIAL_PERIOD
DIAGNOSING_PERIOD = 2 * INERTIAL_PERIOD


# Set only one gpu per rank visible
n_gpus_per_node = 4
gpu_id = MPI.COMM_WORLD.Get_rank() % n_gpus_per_node
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Set the log level of the sweep to debug
log.setLevel("DEBUG")

# Define the experiment function

def compute_imbalance(rossby_number: float) -> dict:
    """
    Diagnosed imbalance vs ramp period in the shallow water model.

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
    grid = sw.grid.cartesian.Grid(N=(nx, ny), L=(lx, ly), periodic_bounds=(True, False))
    mset = sw.ModelSettings(grid, f0=1.0, beta=0.0, csqr=1.0, Ro=rossby_number).setup()
    mset.time_stepper.dt = dt

    # construct balancing methods
    averaging_proj = sw.projection.GeostrophicTimeAverage(
        mset = mset,
        n_ave = 3,
        equidistant_chunks = True,
        max_period = INERTIAL_PERIOD,
    )

    optimal_balance = sw.projection.OptimalBalance(
        mset=mset,
        base_proj=averaging_proj,
        update_base_point=True,
        ramp_period=RAMP_PERIOD/rossby_number,
        ramp_type="exp",
        max_it=3,
        stop_criterion=0,  # make sure to reach the maximum number of iterations
    )

    # create the initial condition
    z_ini = obta_paper.initial_conditions.ShallowWaterRandom(mset)

    imbalance = obta_paper.diagnosed_imbalance(
        mset=mset,
        state=z_ini,
        projector=optimal_balance,
        diagnosing_period=DIAGNOSING_PERIOD/rossby_number,
    )

    return {"imbalance": imbalance}

# Define the sweep
sweep = SweepExpMPI(
    func = compute_imbalance,
    parameters = {
        "rossby_number": np.logspace(np.log10(0.05), np.log10(0.5), 12),
    },
    return_values = {"imbalance": float},
    save_path = "data/rossby_number_bound_sw.nc",
)

sweep.timeit = True
sweep.auto_save = True
sweep.enable_priorities = True

# Run experiments with smallest rossby number first since they are the most expensive
sweep.priority.data += (1.0/sweep.parameters["rossby_number"]).astype(int)

# Run the sweep
sweep.run()
