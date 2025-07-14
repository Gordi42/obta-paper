"""Compute the imbalance vs rossby number in the nonhydrostatic model."""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Literal

import fridom.nonhydro as nh
import numpy as np
from mpi4py import MPI
from sweepexp import SweepExpMPI, log

import obta_paper

# constant
RESOLUTION_FACTOR = 8
INERTIAL_PERIOD = 2.0 * np.pi
RAMP_PERIOD = INERTIAL_PERIOD
DIAGNOSING_PERIOD = 3 * INERTIAL_PERIOD


# Set only one gpu per rank visible
n_gpus_per_node = 4
gpu_id = MPI.COMM_WORLD.Get_rank() % n_gpus_per_node
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Set the log level of the sweep to debug
log.setLevel("DEBUG")

# Define the experiment function

def compute_imbalance(rossby_number: float,
                      closure_ramping: Literal["No Viscosity",
                                               "Constant Viscosity",
                                               "Ramping Viscosity"]) -> dict:
    """
    Diagnosed imbalance vs ramp period in the shallow water model.

    Parameters
    ----------
    rossby_number : float
        The advection scale of the shallow water model.
    closure_ramping : str
        "No Viscosity" No viscosity during optimal balance ramping
        "Constant Viscosity" Constant viscosity during optimal balance ramping
        "Ramping Viscosity" Ramping viscosity during optimal balance ramping

    """
    # Scale parameters
    height_scale = 1e3
    length_scale = 5e4
    dsqr = (height_scale / length_scale)**2

    # ================================================================
    #  Setup
    # ================================================================

    lx = ly = 2 * np.pi
    nx = ny = 2**RESOLUTION_FACTOR - 1
    lz = 1
    nz = 2**(RESOLUTION_FACTOR - 2) - 1

    # construct grid and modelsettings
    grid = nh.grid.cartesian.Grid(N=(nx, ny, nz), L=(lx, ly, lz),
                                periodic_bounds=(True, True, False))
    mset = nh.ModelSettings(grid, f0=1.0, beta=0.0, N2=1.0,
                            dsqr=dsqr, Ro=rossby_number, halo=2).setup()

    # Set background stratification
    obta_paper.set_background_stratification(mset, strat_type="reference")

    # ----------------------------------------------------------------
    #  Diffusion coefficient for numerical stability
    # ----------------------------------------------------------------
    dx, dy, dz = grid.dx

    kh_max = np.pi / dx
    hor_velocity_scale = 1.0
    hor_diff_coeff = hor_velocity_scale * rossby_number / kh_max**3

    kv_max = np.pi / dz
    ver_velocity_scale = 0.01
    ver_diff_coeff = ver_velocity_scale * rossby_number / kv_max**3

    # ----------------------------------------------------------------
    #  Choose the time step size
    # ----------------------------------------------------------------
    # Wave stability condition
    max_strat = mset.N2_field.max().arr.item()
    max_eigenvalue_wave = (max_strat / mset.dsqr)**0.5
    # Advection stability condition
    max_eigenvalue_adv = rossby_number * max(
        hor_velocity_scale*kh_max, ver_velocity_scale*kv_max)
    # Diffusion stability condition
    max_eigenvalue_diff = max(hor_diff_coeff * kh_max**4, ver_diff_coeff * kv_max**4)
    # Compute the maximum eigenvalue
    max_eigenvalue = max(max_eigenvalue_wave, max_eigenvalue_adv, max_eigenvalue_diff)
    # Set the time step size to 50% of the maximum eigenvalue
    mset.time_stepper.dt = 0.5 / max_eigenvalue

    # ----------------------------------------------------------------
    #  Create a model settings for the viscous model
    # ----------------------------------------------------------------
    mset_inviscid = deepcopy(mset)

    mset.tendencies.add_module(obta_paper.BiharmonicClosure(
        kh=hor_diff_coeff, kv=ver_diff_coeff))

    # ----------------------------------------------------------------
    #  Projection operators
    # ----------------------------------------------------------------
    def update_parameters(mset:
                          nh.ModelSettings,
                          ramped_value: float,
                          mode: str) -> None:
        mset.tendencies.advection.scaling = ramped_value * mset.Ro
        sign = 1.0 if mode == "forward" else -1.0
        rho = ramped_value if closure_ramping == "Ramping Viscosity" else 1.0
        closure = mset.tendencies.get("Biharmonic Closure")[0]
        closure.kh = sign * hor_diff_coeff * rho
        closure.kv = sign * ver_diff_coeff * rho

    base_proj = nh.projection.GeostrophicSpectral(mset)

    if closure_ramping == "No Viscosity":
        optimal_balance = nh.projection.OptimalBalance(
            mset=mset_inviscid,
            base_proj=base_proj,
            update_base_point=False,
            ramp_period=RAMP_PERIOD/rossby_number,
            ramp_type="exp",
            max_it=3,
            stop_criterion=0,  # make sure to reach the maximum number of iterations
        )
    else:
        optimal_balance = nh.projection.OptimalBalance(
            mset=mset,
            base_proj=base_proj,
            update_base_point=False,
            ramp_period=RAMP_PERIOD/rossby_number,
            ramp_type="exp",
            max_it=3,
            stop_criterion=0,  # make sure to reach the maximum number of iterations
            update_parameters=update_parameters,
        )

    # ----------------------------------------------------------------
    #  Create the initial condition and compute the imbalance
    # ----------------------------------------------------------------

    z_ini = obta_paper.initial_conditions.NonhydroJet(mset)

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
        "closure_ramping": ["No Viscosity", "Constant Viscosity", "Ramping Viscosity"],
        "rossby_number": np.logspace(np.log10(0.05), np.log10(0.5), 12),
    },
    return_values = {"imbalance": float},
    save_path = "data/closure_ramping_nh.nc",
)

sweep.timeit = True
sweep.auto_save = True
sweep.enable_priorities = True

# Run experiments with smallest rossby number first since they are the most expensive
sweep.priority.data += (1.0/sweep.parameters["rossby_number"]).astype(int)

# Run the sweep
sweep.run()
