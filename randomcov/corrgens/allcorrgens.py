from randomcov.corrgens.lkj import lkj_corr
from randomcov.corrgens.wishart import wishart_corr
from randomcov.corrgens.residuals import residuals_corr
from randomcov.corrgens.walk import walk_corr
from randomcov.corrgens.animals import animals_corr
from enum import Enum

class CorrMethod(str, Enum):
    LKJ = "lkj"
    RESIDUALS = "residuals"
    WALK = "walk"
    WISHART = "wishart"
    ANIMALS = "animals"

CORR_GENERATORS = {CorrMethod.LKJ:lkj_corr,
                   CorrMethod.WISHART:wishart_corr,
                   CorrMethod.RESIDUALS:residuals_corr,
                   CorrMethod.WALK:walk_corr,
                   CorrMethod.ANIMALS:animals_corr}
