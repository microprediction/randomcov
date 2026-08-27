# randomcov
Generating random covariance and correlation matrices with beautiful geodesic interpolation visualizations! 🎯


### Install 

    pip install randomcov 

or for latest

    pip install git+https://github.com/microprediction/randomcov.git
    
### Example

    from randomcov import random_covariance_matrix
    cov = random_covariance_matrix(n=50, corr_method='residuals', var_method='lognormal')


### Motivation

To collect standard but also novel correlation and covariance generation methods, in order to better understand when some estimation methods work better than others in different contexts: such as the construction of machine learning model ensembles, combinations of forecasts, or financial portfolios.  

The geodesic interpolation capabilities enable smooth transformations between covariance structures while preserving mathematical properties, making it ideal for portfolio optimization and risk management applications.




## Generator catalog

Every method: `random_correlation_matrix(n, corr_method="<name>", corr_kwargs={...})`.
All generators accept `rng=` (int seed or numpy Generator) for reproducibility.

| method | measure / structure | reference |
|---|---|---|
| `lkj` | LKJ(eta) via Cholesky factor; eta=1 uniform on the elliptope | Lewandowski, Kurowicka & Joe 2009 |
| `onion` | extended onion, exact LKJ(eta) | Ghosh & Henderson 2003; LKJ 2009 |
| `vine` | C-vine partial correlations, Beta margins (LKJ-exact) | Joe 2006 |
| `archakov_hansen` | Gaussian in matrix-log space + unit-diagonal fixed point | Archakov & Hansen 2021 |
| `spectrum` | prescribed random eigenvalues (exp / dirichlet / marchenko_pastur / spiked), Haar frame, Givens diagonal restoration | Bendel & Mickey 1978; Davies & Higham 2000; Johnstone 2001 |
| `wishart` | normalized Wishart sample correlation | classical |
| `residuals` | sample correlation of residual-driven paths | this package |
| `factor` | k factors + idiosyncratic, optional sparse links | approximate-factor literature |
| `hierarchical` | ultrametric / cophenetic from a random dendrogram | Tumminello, Lillo & Mantegna |
| `block_equicorr` | constant within/between blocks (DECO, block form) | Engle & Kelly 2012 |
| `ar1` | Toeplitz rho^|i-j| | Kac-Murdock-Szego |
| `kernel` | RBF / Matern field on a random point cloud | spatial statistics |
| `sparse_precision` | random Gaussian graphical model (sparse inverse) | graphical-model literature |
| `walk` | perturbation random walk on the elliptope, nearest-corr projected | this package |
| `animals` | agent-based interacting sizes (emergent correlation) | this package |

There is no canonical "uniform" measure over covariance matrices, and the
elliptope-uniform (LKJ eta=1) concentrates on weak correlations as n grows;
robust conclusions require batteries across NAMED ensembles, which is what
this catalog is for.

## Correlation Inflation

An example of the kind of thing I wish to test against generative models. 

**[Correlation Inflation: A Working Paper](https://github.com/microprediction/home/blob/main/workingpapers/CorrelationInflation.pdf)**

*The transformation preserves geometric properties while smoothly interpolating towards perfect correlation structure using differential geometry on the manifold of positive definite matrices.*

![Geodesic Interpolation Towards Perfect Correlation](https://github.com/microprediction/randomcov/blob/main/geodesic_interpolation_beautiful.gif)


