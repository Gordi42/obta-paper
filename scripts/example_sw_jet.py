"""Show the fields of the initial condition for the shallow water jet setup."""
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
grid = sw.grid.cartesian.Grid(N=(nx, ny), L=(lx, ly), periodic_bounds=(True, True))
mset = sw.ModelSettings(grid, f0=1.0, beta=0.0, csqr=1.0, Ro=ROSSBY_NUMBER).setup()
mset.time_stepper.dt = dt

# construct balancing methods
geo_proj = sw.projection.GeostrophicSpectral(mset, use_discrete=True)
optimal_balance = sw.projection.OptimalBalance(
    mset=mset,
    base_proj=geo_proj,
    ramp_period=RAMP_PERIOD,
    ramp_type="exp",
    max_it=3,
)

# create the initial condition
z_ini = obta_paper.initial_conditions.ShallowWaterJet(mset)

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
data_path = "../data/examples/shallow_water_jet/"
z_ini.to_netcdf(data_path + "z_ini.nc")
z_ini_bal.to_netcdf(data_path + "z_ini_bal.nc")
z_evo.to_netcdf(data_path + "z_evo.nc")
z_evo_bal.to_netcdf(data_path + "z_evo_bal.nc")
