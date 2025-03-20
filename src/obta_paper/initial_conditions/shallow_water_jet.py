"""The unstable jet initial condition for the shallow water model."""
from __future__ import annotations

import fridom.shallowwater as sw


class ShallowWaterJet(sw.State):

    r"""
    Two opposing unstable jets.

    Description
    -----------
    An instable jet setup with a small pressure perturbation
    on top of it. The jet is given by:

    .. math::
        u = 2.5 \exp\left(-\left(\frac{y - p_2 L_y}{\sigma \pi}\right)^2\right)
            - 2.5 \exp\left(-\left(\frac{y - p_1 L_y}{\sigma \pi}\right)^2\right)

    where :math:`L_y` is the domain length in the y-direction,
    :math:`p_i` are the relative positions of the jets in the y-direction,
    and :math:`\\sigma` is the width of the jet. The perturbation
    is given by:

    .. math::
        p = A \\sin \\left( \\frac{2 \\pi}{L_x} k_p x \\right)

    where :math:`A` is the amplitude of the perturbation and :math:`k_p` is the
    wavenumber of the perturbation.

    Parameters
    ----------
    mset : ModelSettings
        The model settings.
    wavenum : int
        The relative wavenumber of the perturbation.
    waveamp : float
        The amplitude of the perturbation.
    pos : tuple(float)
        The relative positions of the jet in the y-direction
    width : float
        The width of the jet.

    """

    def __init__(self,
                 mset: sw.ModelSettings,
                 wavenum: int = 5,
                 waveamp: float = 1e-3,
                 pos: tuple[float] = (0.25, 0.75),
                 width: float = 0.04) -> None:
        super().__init__(mset)
        # Shortcuts
        ncp = sw.config.ncp
        lx, ly = self.grid.L

        # Construct the zonal jets
        state = sw.State(mset)
        x, y = state.u.get_mesh()
        state.u.arr = (+ ncp.exp(- ((y - pos[1] * ly)/(width * ncp.pi))**2)
                       - ncp.exp(- ((y - pos[0] * ly)/(width * ncp.pi))**2) )

        # Construct the perturbation
        kx_p = 2 * ncp.pi / lx * wavenum
        x, y = state.p.get_mesh()
        state.p.arr = waveamp * ncp.sin(kx_p * x)

        self.fields = state.fields
