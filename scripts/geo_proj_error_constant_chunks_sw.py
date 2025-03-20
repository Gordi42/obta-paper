"""Compute the geostrophic projection error of time averaging (constant chunks)."""
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

def compute_equidistant_averaging_error(initial_condition: str,
                                        number_chunks: int,
                                        total_averaging_time: float) -> dict:
    """
    Projection error of equidistant time averaging.

    Parameters
    ----------
    initial_condition : str
        The initial condition to use. Either "jet" or "random".
    number_chunks : int
        The total number of time averaging chunks.
    total_averaging_time : float
        The total averaging period.

    """
    # compute the grid resolution
    lx = ly = 2 * np.pi
    nx = ny = 2**RESOLUTION_FACTOR - 1
    dt = 1 / nx

    # construct grid and modelsettings
    grid = sw.grid.cartesian.Grid(N=(nx, ny), L=(lx, ly), periodic_bounds=(True, True))
    mset = sw.ModelSettings(grid, f0=1.0, beta=0.0, csqr=1.0).setup()
    mset.time_stepper.dt = dt

    # construct balancing methods
    spectral_proj = sw.projection.GeostrophicSpectral(mset, use_discrete=True)
    averaging_proj = sw.projection.GeostrophicTimeAverage(
        mset = mset,
        n_ave = number_chunks,
        equidistant_chunks = False,
        max_period = total_averaging_time / number_chunks,
    )

    # create the initial condition
    if initial_condition == "jet":
        z_ini = obta_paper.initial_conditions.ShallowWaterJet(mset)
    elif initial_condition == "random":
        z_ini = obta_paper.initial_conditions.ShallowWaterRandom(mset)
    else:
        msg = "Unknown initial condition."
        raise ValueError(msg)

    averaging_error = obta_paper.projection_deviations(
        state = z_ini,
        projector_1 = spectral_proj,
        projector_2 = averaging_proj,
    )

    return {"averaging_error": averaging_error}

# compute the total averaging periods
n_chunks = np.arange(1, 18)
total_averaging_time = (3 * n_chunks + 1) / 4 * INERTIAL_PERIOD

# Define the sweep
sweep = SweepExpMPI(
    func = compute_equidistant_averaging_error,
    parameters = {
        "initial_condition": ["jet", "random"],
        "number_chunks": np.arange(1, 14, 3),
        "total_averaging_time": total_averaging_time,
    },
    return_values = {"averaging_error": float},
    save_path = "data/geo_proj_error_constant_sw.nc",
)

sweep.timeit = True
sweep.auto_save = True
sweep.enable_priorities = True

sweep.priority.data += sweep.parameters["total_averaging_time"].astype(int)

# Run the sweep
sweep.run()
