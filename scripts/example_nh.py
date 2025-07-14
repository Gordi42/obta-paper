"""Show the fields of the initial condition for the shallow water jet setup."""
from __future__ import annotations

from copy import deepcopy

import fridom.nonhydro as nh
import numpy as np

import obta_paper

# ----------------------------------------------------------------
#  Constants
# ----------------------------------------------------------------

RESOLUTION_FACTOR = 8
ROSSBY_NUMBER = 0.1  # advection scale
INERTIAL_PERIOD = 2.0 * np.pi
RAMP_PERIOD = INERTIAL_PERIOD / ROSSBY_NUMBER
DIAGNOSING_PERIOD = 3 * INERTIAL_PERIOD / ROSSBY_NUMBER

# ================================================================
#  Setup
# ================================================================
# set the log level to verbose
nh.log.setLevel("VERBOSE")

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
                        dsqr=dsqr, Ro=ROSSBY_NUMBER, halo=2).setup()

# ----------------------------------------------------------------
#  Diffusion coefficient for numerical stability
# ----------------------------------------------------------------
dx, dy, dz = grid.dx

kh_max = np.pi / dx
hor_velocity_scale = 1.0
hor_diff_coeff = hor_velocity_scale * ROSSBY_NUMBER / kh_max**3

kv_max = np.pi / dz
ver_velocity_scale = 0.01
ver_diff_coeff = ver_velocity_scale * ROSSBY_NUMBER / kv_max**3

# ----------------------------------------------------------------
#  Choose the time step size
# ----------------------------------------------------------------
# Wave stability condition
max_strat = mset.N2_field.max().arr.item()
max_eigenvalue_wave = (max_strat / mset.dsqr)**0.5
# Advection stability condition
max_eigenvalue_adv = ROSSBY_NUMBER * max(
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
mset_viscous = deepcopy(mset)
mset_viscous.tendencies.add_module(obta_paper.BiharmonicClosure(
    kh=hor_diff_coeff, kv=ver_diff_coeff))

# ----------------------------------------------------------------
#  Projection operators
# ----------------------------------------------------------------

spectral_proj = nh.projection.GeostrophicSpectral(mset)

optimal_balance = nh.projection.OptimalBalance(
    mset=mset,
    base_proj=spectral_proj,
    update_base_point=False,
    ramp_period=RAMP_PERIOD,
    ramp_type="exp",
    max_it=3,
    stop_criterion=0,  # make sure to reach the maximum number of iterations
)

# ----------------------------------------------------------------
#  Create the initial condition and compute the imbalance
# ----------------------------------------------------------------

z_ini = obta_paper.initial_conditions.NonhydroJet(mset)

# balance the initial condition
z_ini_bal = optimal_balance(z_ini)

# ================================================================
#  Run the model
# ================================================================
nh.log.setLevel("INFO")

model = nh.Model(mset_viscous)
model.z = z_ini_bal
model.run(runlen=DIAGNOSING_PERIOD)

# get the evolved state and balance it again
nh.log.setLevel("VERBOSE")
z_evo = model.z
z_evo_bal = optimal_balance(z_evo)

# ================================================================
#  Save the results
# ================================================================
data_path = "../data/examples/nh_jet/"
z_ini.to_netcdf(data_path + "z_ini.nc")
z_ini_bal.to_netcdf(data_path + "z_ini_bal.nc")
z_evo.to_netcdf(data_path + "z_evo.nc")
z_evo_bal.to_netcdf(data_path + "z_evo_bal.nc")
