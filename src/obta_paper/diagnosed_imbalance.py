"""Contains a function to compute the diagnosed imbalance."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import fridom.framework as fr


def diagnosed_imbalance(
        mset: fr.ModelSettingsBase,
        state: fr.VectorField,
        projector: fr.projection.Projection,
        diagnosing_period: float,
        ) -> float:
    """
    Compute the imbalance diagnosed by a projection.

    Description
    -----------
    To compute the diagnosed imbalance, a state is first projected onto a
    given subspace using a projector. This projected state is then given as
    an initial condition to a model, which is integrated for a given period.
    Finally, the obtained evolved state is projected again onto the same
    subspace. The imbalance is then computed as the norm of the difference
    between the final evolved state and the projected final evolved state.

    Parameters
    ----------
    mset : fr.ModelSettingsBase
        The model settings of the state.
    state : fr.VectorField
        The state to diagnose the imbalance with.
    projector : fr.projection.Projection
        The projector of which the diagnosed imbalance is computed.
    diagnosing_period : float
        The period over which the imbalance is diagnosed.

    """
    # TODO(Silvano): Implement this function.
