from randomcov.corrgens.lkj import lkj_corr
from randomcov.corrgens.wishart import wishart_corr
from randomcov.corrgens.residuals import residuals_corr
from randomcov.corrgens.walk import walk_corr
from randomcov.corrgens.animals import animals_corr
from randomcov.corrgens.onion import onion_corr
from randomcov.corrgens.vine import vine_corr
from randomcov.corrgens.archakovhansen import archakov_hansen_corr
from randomcov.corrgens.spectrum import spectrum_corr
from randomcov.corrgens.factor import factor_corr
from randomcov.corrgens.hierarchical import hierarchical_corr
from randomcov.corrgens.blockequi import block_equicorr
from randomcov.corrgens.ar1 import ar1_corr
from randomcov.corrgens.kernelcorr import kernel_corr
from randomcov.corrgens.sparseprecision import sparse_precision_corr
from enum import Enum

class CorrMethod(str, Enum):
    LKJ = "lkj"
    RESIDUALS = "residuals"
    WALK = "walk"
    WISHART = "wishart"
    ANIMALS = "animals"
    ONION = "onion"
    VINE = "vine"
    ARCHAKOV_HANSEN = "archakov_hansen"
    SPECTRUM = "spectrum"
    FACTOR = "factor"
    HIERARCHICAL = "hierarchical"
    BLOCK_EQUICORR = "block_equicorr"
    AR1 = "ar1"
    KERNEL = "kernel"
    SPARSE_PRECISION = "sparse_precision"

CORR_GENERATORS = {CorrMethod.LKJ: lkj_corr,
                   CorrMethod.WISHART: wishart_corr,
                   CorrMethod.RESIDUALS: residuals_corr,
                   CorrMethod.WALK: walk_corr,
                   CorrMethod.ANIMALS: animals_corr,
                   CorrMethod.ONION: onion_corr,
                   CorrMethod.VINE: vine_corr,
                   CorrMethod.ARCHAKOV_HANSEN: archakov_hansen_corr,
                   CorrMethod.SPECTRUM: spectrum_corr,
                   CorrMethod.FACTOR: factor_corr,
                   CorrMethod.HIERARCHICAL: hierarchical_corr,
                   CorrMethod.BLOCK_EQUICORR: block_equicorr,
                   CorrMethod.AR1: ar1_corr,
                   CorrMethod.KERNEL: kernel_corr,
                   CorrMethod.SPARSE_PRECISION: sparse_precision_corr}
