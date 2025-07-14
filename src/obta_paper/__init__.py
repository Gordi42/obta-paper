"""
Optimal Balance with Time Averaging.

====================================

Description
-----------
Experiments for the paper that introduces the optimal balance with
time averaging method.

For more information, visit the project's GitHub repository:
https://github.com/Gordi42/obta_paper
"""
__author__ = """Silvano Gordian Rosenau"""
__email__ = "silvano.rosenau@uni-hamburg.de"
__version__ = "0.1.0"

from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    from . import initial_conditions
    from .background_stratification import set_background_stratification
    from .biharmonic_closure import BiharmonicClosure, HorizontalBiharmonic
    from .diagnosed_imbalance import diagnosed_imbalance
    from .projection_deviations import projection_deviations

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {
    "obta_paper": ["initial_conditions"],
}

all_imports_by_origin = {
    "obta_paper.background_stratification": ["set_background_stratification"],
    "obta_paper.biharmonic_closure": ["BiharmonicClosure", "HorizontalBiharmonic"],
    "obta_paper.diagnosed_imbalance": ["diagnosed_imbalance"],
    "obta_paper.projection_deviations": ["projection_deviations"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
