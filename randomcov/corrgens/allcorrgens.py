from randomcov.corrgens.lkj import lkj_corr
from randomcov.corrgens.wishart import wishart_corr
from randomcov.corrgens.residuals import residuals_corr
from randomcov.corrgens.walk import walk_corr
from randomcov.corrgens.animals import animals_corr

CORR_GENERATORS = [lkj_corr, wishart_corr, residuals_corr, walk_corr, animals_corr]
