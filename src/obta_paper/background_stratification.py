"""Different stratification configurations for the nonhydrostatic model."""
from __future__ import annotations

import fridom.nonhydro as nh


def _unscaled_exp_stratification(z: float) -> float:
    """Exponential stratification based on Saez et al. (2023)."""
    ncp = nh.config.ncp
    return 2.5e-3 + 12.5e-3 * ncp.exp((z + 50) / 100)

def set_background_stratification(mset: nh.ModelSettings, strat_type: str) -> None:
    r"""
    Set the background stratification of the model.

    Description
    -----------
    This function sets the burger number of the model to a given value. The
    stratification can be related to the burger number by the following
    relation:

    .. math::
        N^2 = \frac{f^2 L^2}{H^2} Bu

    The options for the Burger number are:
    - "reference": a constant value of 1.0
    - "strong_constant": a constant value of 2.0
    - "strong_top_weak_bottom": a value of 1.0 in the bottom half and 2.0 in the top
    - "weak_top_strong_bottom": a value of 2.0 in the bottom half and 1.0 in the top
    - "mixed_layer": a value of 1.0 in the bottom half and 0.0 in the top
    - "cos": a cosine

    """
    ncp = nh.config.ncp
    x, y, z = mset.N2_field.get_mesh()
    lx, ly, lz = mset.grid.L

    # get the background stratification
    if strat_type == "reference":
        strat = z * 0 + 1.0
    elif strat_type == "strong_constant":
        strat = z * 0 + 2.0
    elif strat_type == "strong_top_weak_bottom":
        strat = 1.0 + 1.0 * (z > lz/2)
    elif strat_type == "weak_top_strong_bottom":
        strat = 1.0 + 1.0 * (z < lz/2)
    elif strat_type == "mixed_layer":
        strat = 1.0 * (z < lz/2)
    elif strat_type == "cos":
        strat = 1.5 - 0.5 * ncp.cos(z * ncp.pi / lz)
    else:
        msg = f"Unknown stratification type: {strat_type}"
        raise ValueError(msg)

    # set the stratification
    mset.N2_field.arr = strat
