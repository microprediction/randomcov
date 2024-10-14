from randomcov.corrgens.wishart import wishart_corr
from randomcov.corrgensutil.isvalidcorr import is_valid_corr


def test_wishart_corr_small():
    c = wishart_corr(n=2)
    assert is_valid_corr(c)


def test_wishart_corr_large():
    c = wishart_corr(n=5000)
    assert is_valid_corr(c)


