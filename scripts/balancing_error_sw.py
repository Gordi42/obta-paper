"""Compute the balancing error vs rossby number in the shallow water model."""
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

def compute_imbalance(initial_condition: str,
                      rossby_number: float,
                      number_chunks: int) -> dict:
    """
    Diagnosed imbalance vs ramp period in the shallow water model.

    Parameters
    ----------
    initial_condition : str
        The initial condition to use. Either "jet" or "random".
    rossby_number : float
        The advection scale of the shallow water model.
    number_chunks : int
        The total number of time averaging chunks.

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
    spectral_proj = sw.projection.GeostrophicSpectral(mset, use_discrete=True)
    averaging_proj = sw.projection.GeostrophicTimeAverage(
        mset = mset,
        n_ave = number_chunks,
        equidistant_chunks = True,
        max_period = INERTIAL_PERIOD,
    )

    ob = sw.projection.OptimalBalance(
        mset=mset,
        base_proj=spectral_proj,
        update_base_point=False,
        ramp_period=RAMP_PERIOD/rossby_number,
        ramp_type="exp",
        max_it=3,
        stop_criterion=0,  # make sure to reach the maximum number of iterations
    )

    obta = sw.projection.OptimalBalance(
        mset=mset,
        base_proj=averaging_proj,
        update_base_point=True,
        ramp_period=RAMP_PERIOD/rossby_number,
        ramp_type="exp",
        max_it=3,
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

    # compute the reference balanced state
    z_ref = ob(z_ini)

    # compute the balanced state with obta
    z_obta = obta(z_ini)

    # compute the norm of difference between the two
    balancing_error = z_ref.norm_of_diff(z_obta)

    return {"balancing_error": balancing_error}

# Define the sweep
sweep = SweepExpMPI(
    func = compute_imbalance,
    parameters = {
        "initial_condition": ["jet", "random"],
        "number_chunks": [1, 2, 3],
        "rossby_number": np.logspace(np.log10(0.05), np.log10(0.5), 12),
    },
    return_values = {"balancing_error": float},
    save_path = "data/balancing_error_sw.nc",
)

sweep.timeit = True
sweep.auto_save = True
sweep.enable_priorities = True

# Run experiments with smallest rossby number first since they are the most expensive
sweep.priority.data += (1.0/sweep.parameters["rossby_number"]).astype(int)

# Run the sweep
sweep.run()
