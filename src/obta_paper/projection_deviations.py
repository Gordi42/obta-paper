"""Contains a function to compute deviations between two projectors."""
import fridom.framework as fr


def projection_deviations(
        state: fr.VectorField,
        projector_1: fr.projection.Projection,
        projector_2: fr.projection.Projection,
        ) -> float:
    """
    Compute the norm of difference between two projections.

    Parameters
    ----------
    state : fr.VectorField
        The state to test the projection with.
    projector_1 : fr.projection.Projection
        The first projector.
    projector_2 : fr.projection.Projection
        The second projector.

    """
    # TODO(Silvano): Implement this function.
