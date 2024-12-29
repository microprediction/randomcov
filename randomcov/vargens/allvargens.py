# vargens/allvargens.py

from enum import Enum
from randomcov.vargens.lognormalvar import lognormal_var

class VarMethod(str, Enum):
    LOGNORMAL = "lognormal"

# Direct mapping of VarMethod to generator functions
VAR_GENERATORS = {
    VarMethod.LOGNORMAL: lognormal_var
}
