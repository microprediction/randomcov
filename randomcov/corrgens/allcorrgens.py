from randomcov.corrgens.lkj import lkj_corr
from randomcov.corrgens.wishart import wishart_corr
from randomcov.corrgens.residuals import residuals_corr

CORR_GENERATORS = [lkj_corr, wishart_corr, residuals_corr]