"""Biharmonic closure for the nonhydrostatic model."""
from __future__ import annotations

from functools import partial

import fridom.framework as fr
import fridom.nonhydro as nh


@partial(fr.utils.jaxify, dynamic=("kh", "kv"))
class BiharmonicClosure(fr.modules.Module):

    r"""
    Biharmonic friction + mixing for the nonhydrostatic model.

    Parameters
    ----------
    kh : float
        horizontal diffusion coefficient
    kv : float
        vertical diffusion coefficient

    """

    name = "Biharmonic Closure"
    def __init__(self, kh: float, kv: float) -> None:
        super().__init__()
        self.kh = kh
        self.kv = kv
        self.required_halo = 2

    @fr.utils.jaxjit
    def _compute_tendency(self, z: nh.State, dz: nh.State) -> nh.State:
        diff = self.diff_module.diff
        for f in z:
            # first two derivatives
            f_hor = (diff(f, axis=0, order=2) +
                          diff(f, axis=1, order=2) )
            f_ver = diff(f, axis=2, order=2)

            # multiply diffusion coefficients
            f_hor *= self.kh
            f_ver *= self.kv

            # second two derivatives
            dz[f.name] -= (diff(f_hor, axis=0, order=2) +
                           diff(f_hor, axis=1, order=2) +
                           diff(f_ver, axis=2, order=2) )
        return dz

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        mz.dz = self._compute_tendency(mz.z, mz.dz)
        return mz

@fr.utils.jaxify
class HorizontalBiharmonic(fr.modules.Module):

    r"""
    Biharmonic friction + mixing for the nonhydrostatic model.

    Parameters
    ----------
    kh : float
        horizontal diffusion coefficient

    """

    name = "Biharmonic Closure"
    def __init__(self, kh: float) -> None:
        super().__init__()
        self._kh = kh
        self.required_halo = 2

    @fr.utils.jaxjit
    def _compute_tendency(self, z: nh.State, dz: nh.State) -> nh.State:
        diff = self.diff_module.diff
        for f in z:
            # first two derivatives
            f_hor = diff(f, axis=0, order=2) + diff(f, axis=1, order=2)

            # multiply diffusion coefficient
            f_hor *= self._kh

            # second two derivatives
            dz[f.name] -= diff(f_hor, axis=0, order=2) + diff(f_hor, axis=1, order=2)
        return dz

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        mz.dz = self._compute_tendency(mz.z, mz.dz)
        return mz
