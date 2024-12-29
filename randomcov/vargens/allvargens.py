# vargens/allvargens.py

from enum import Enum
from randomcov.vargens.lognormalvar import lognormal_var
from randomcov.vargens.unitvar import unit_var

class VarMethod(str, Enum):
    LOGNORMAL = "lognormal"
    UNIT = "unit"

# Direct mapping of VarMethod to generator functions
VAR_GENERATORS = {
    VarMethod.LOGNORMAL: lognormal_var,
    VarMethod.UNIT: unit_var
}
