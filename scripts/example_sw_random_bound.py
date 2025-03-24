"""Show the fields of the initial condition for the shallow water random setup."""
from __future__ import annotations

import fridom.shallowwater as sw
import numpy as np

import obta_paper

# ----------------------------------------------------------------
#  Constants
# ----------------------------------------------------------------

RESOLUTION_FACTOR = 9
ROSSBY_NUMBER = 0.1  # advection scale
INERTIAL_PERIOD = 2.0 * np.pi
RAMP_PERIOD = INERTIAL_PERIOD / ROSSBY_NUMBER
DIAGNOSING_PERIOD = 2 * INERTIAL_PERIOD / ROSSBY_NUMBER

# ================================================================
#  Setup
# ================================================================
# set the log level to verbose
sw.log.setLevel("VERBOSE")

lx = ly = 2 * np.pi
nx = ny = 2**RESOLUTION_FACTOR - 1
dt = 1 / nx

# construct grid and modelsettings
grid = sw.grid.cartesian.Grid(N=(nx, ny), L=(lx, ly), periodic_bounds=(True, False))
mset = sw.ModelSettings(grid, f0=1.0, beta=0.0, csqr=1.0, Ro=ROSSBY_NUMBER).setup()
mset.time_stepper.dt = dt

# construct balancing methods
geo_proj = sw.projection.GeostrophicTimeAverage(
    mset,
    max_period=INERTIAL_PERIOD,
    n_ave=3,
    equidistant_chunks=True,
)

optimal_balance = sw.projection.OptimalBalance(
    mset=mset,
    base_proj=geo_proj,
    update_base_point=True,
    ramp_period=RAMP_PERIOD,
    ramp_type="exp",
    max_it=3,
    stop_criterion=0,  # make sure to reach the maximum number of iterations
)

# create the initial condition
z_ini = obta_paper.initial_conditions.ShallowWaterRandom(mset)

# balance the initial condition
z_ini_bal = optimal_balance(z_ini)

# ================================================================
#  Run the model
# ================================================================
sw.log.setLevel("INFO")

model = sw.Model(mset)
model.z = z_ini_bal
model.run(runlen=DIAGNOSING_PERIOD)

# get the evolved state and balance it again
sw.log.setLevel("VERBOSE")
z_evo = model.z
z_evo_bal = optimal_balance(z_evo)

# ================================================================
#  Save the results
# ================================================================
data_path = "../data/examples/shallow_water_random_bound/"
z_ini.to_netcdf(data_path + "z_ini.nc")
z_ini_bal.to_netcdf(data_path + "z_ini_bal.nc")
z_evo.to_netcdf(data_path + "z_evo.nc")
z_evo_bal.to_netcdf(data_path + "z_evo_bal.nc")
