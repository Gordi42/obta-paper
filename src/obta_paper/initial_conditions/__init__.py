"""Initial conditions used in the OBTA paper."""

from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from .shallow_water_jet import ShallowWaterJet
    from .shallow_water_random import ShallowWaterRandom

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {
}

base = "obta_paper.initial_conditions"

all_imports_by_origin = {
    f"{base}.shallow_water_jet": ["ShallowWaterJet"],
    f"{base}.shallow_water_random": ["ShallowWaterRandom"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
