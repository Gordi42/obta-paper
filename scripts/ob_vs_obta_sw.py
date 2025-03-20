"""Compute the difference between ob and obta balancing."""
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

def compute_balancing_deviation(initial_condition: str,
                                rossby_number: float,
                                number_chunks: int,
                                number_iterations: int,
                                balancing_method: str) -> dict:
    """
    Projection error of equidistant time averaging.

    Parameters
    ----------
    initial_condition : str
        The initial condition to use. Either "jet" or "random".
    rossby_number : float
        The advection scale of the shallow water model.
    number_chunks : int
        The total number of time averaging chunks.
    number_iterations : int
        The number of iterations for the optimal balance.
    balancing_method : str
        "obta" for optimal balance with time averaging
        "obta2_5" for obta without recalculating base point
        "ob" for optimal balance without time averaging

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

    # create the initial condition
    if initial_condition == "jet":
        z_ini = obta_paper.initial_conditions.ShallowWaterJet(mset)
    elif initial_condition == "random":
        z_ini = obta_paper.initial_conditions.ShallowWaterRandom(mset)
    else:
        msg = "Unknown initial condition."
        raise ValueError(msg)

    if balancing_method == "obta":
        base_proj = averaging_proj
        update_base = True
    elif balancing_method == "obta2_5":
        base_proj = averaging_proj
        update_base = False
    elif balancing_method == "ob":
        base_proj = spectral_proj
        update_base = False

    optimal_balance = sw.projection.OptimalBalance(
        mset=mset,
        base_proj=base_proj,
        ramp_period=INERTIAL_PERIOD/rossby_number,
        ramp_type="exp",
        update_base_point=update_base,
        max_it=number_iterations,
        stop_criterion=0,  # make sure to reach the maximum number of iterations
    )

    # load the reference balanced state
    f_name = f"data/reference/balanced_sw_{initial_condition}_{rossby_number:.2f}.nc"
    z_ref = sw.State.from_netcdf(mset, path=f_name)

    # balance the initial condition
    z_bal = optimal_balance(z_ini)

    # compute the norm of differences
    deviation = z_ref.norm_of_diff(z_bal)

    return {"deviation": deviation}


# Define the sweep
sweep = SweepExpMPI(
    func = compute_balancing_deviation,
    parameters = {
        "initial_condition": ["jet", "random"],
        "rossby_number": [0.05, 0.1, 0.3],
        "number_chunks": [1, 2, 3],
        "number_iterations": list(range(1, 9)),
        "balancing_method": ["obta", "obta2_5", "ob"],
    },
    return_values = {"deviation": float},
    save_path = "data/ob_vs_obta.nc",
)

sweep.timeit = True
sweep.auto_save = True
sweep.enable_priorities = True

# Run experiments with smallest rossby number first since they are the most expensive
sweep.priority.data += (1.0/sweep.parameters["rossby_number"]).astype(int)

# Skip experiments of obta2_5 and ob with more than 1 averaging chunk
sweep.status.loc[{"balancing_method": ["obta2_5", "ob"],
                  "number_chunks": [2, 3]}] = "S"

sweep.run()
