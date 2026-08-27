# The audit docket

Prominent covariance/correlation-method claims collected across fields as
candidates for the ensemble audit (see `main.tex`). Each entry: the claim,
canonical source, and whether it is ensemble-relative (auditable) or an
ensemble-invariant theorem (null control). Citation counts are approximate
and should be re-verified before quoting. Completed audits live as
`*_sweep.py` scripts beside this file.

## Completed

| field | claim | source | script |
|---|---|---|---|
| statistics | LW shrinkage intensity is "typically moderate" | Ledoit-Wolf 2004 | `lw_sweep.py` |
| finance | HRP beats minimum variance out of sample | Lopez de Prado 2016 | `hrp_sweep.py` |
| econophysics | MP eigenvalue clipping cleans the correlation matrix | Laloux et al. 1999 | `mp_clip_sweep.py` |
| geostatistics | a few nearest neighbors suffice for kriging (screening) | Stein 2002 | `screening_sweep.py` |
| psychometrics | parallel analysis / Kaiser rule find the number of factors | Horn 1965; Kaiser 1960 | `pa_sweep.py` |
| data assimilation (+cosmology) | Gaspari-Cohn localization repairs member-starved covariances (= Paz-Sanchez tapering) | Houtekamer-Mitchell 2001; Hamill et al. 2001; Paz-Sanchez 2015 | `localization_sweep.py` |
| signal processing | diagonal loading robustifies the MVDR beamformer | Cox-Zeskind-Owen 1987; Carlson 1988 | `mvdr_sweep.py` |
| graphical modeling (+genomics, neuro, cosmology) | sparsity prior beats shrinkage for the precision matrix | Friedman et al. 2008; Schafer-Strimmer 2005; Smith et al. 2011; Padmanabhan et al. 2016 | `glasso_sweep.py` |
| statistics (minor) | constant-correlation vs identity shrinkage target | Ledoit-Wolf 2003 vs 2004 | `target_sweep.py` |
| evolutionary biology | random skewers detect matrix similarity | Cheverud 1996; Rohlf 2017 critique | `skewers_sweep.py` |
| climatology | North's rule certifies EOF separation | North et al. 1982 | `north_sweep.py` |
| ecology | Mantel test holds its nominal level | Mantel 1967; Guillot-Rousset 2013 | `mantel_sweep.py` |
| quantitative genetics | estimated G has ~1-2 effective dimensions | Kirkpatrick 2009 | `gdim_sweep.py` |

Also complete: the Hartlap null control (`hartlap_sweep.py` — corrected
ratio 0.98-1.01 under every ensemble, pipeline validated), the T=120
regime check (`robustness_sweep.py`), and the seventeen-estimator
leaderboard over scikit-learn and precise (`ranking_sweep.py` ->
`ranking_results.json`, figure via `ranking_matrix.py`; five distinct
winners across the fifteen ensembles).

TO RUN ON A BIGGER MACHINE: the regime sweep (`regime_sweep.py`) re-runs
six audits at T in {15, 30, 60, 120, 240} (T/n from 0.5 to 8) under every
ensemble — `python papers/measure_audit/regime_sweep.py` (~20-40 min,
graphical-lasso-dominated; `REPS=1` for a smoke test), then
`python papers/measure_audit/regime_matrix.py` renders `regime.pdf`/`.png`.
The code path is smoke-tested; the paper's regime section should be
written from the full-rep results.

The full results assemble into the audit matrix (`audit_matrix.py` ->
`matrix.pdf`/`matrix.png`): claims x ensembles, colored by verdict, with
home-turf cells outlined. Related work now lives in the paper's
Precedents section (Morris et al. 2019; Boulesteix et al. 2013; Niessl
et al. 2022; Hooker 1995; Wolpert-Macready 1997; Hardin et al. 2013;
Marti 2020 CorrGAN, whose HRP experiments anticipated one audit;
Papenbrock et al. 2021; Alvarez et al. 2014).

## Open — ensemble-relative candidates

### Astronomy / cosmology
- **Pope-Szapudi 2008** (MNRAS): shrinkage "should always be employed" for
  power-spectrum covariances. Audit: shrunk vs raw+Hartlap on precision-matrix
  loss; expect flips on factor/equicorr ensembles.
- **Joachimi 2017** (MNRAS Lett): NERCOME nonlinear shrinkage needs "50x fewer
  mocks". Audit: implied mock-saving factor per ensemble (expect ~1 to >>50).
- **Norberg et al. 2009** (MNRAS, ~600 cites): jackknife/bootstrap covariance
  bias verdicts. Audit needs cross-region correlation design; can reproduce the
  Norberg-vs-Shirasaki sign disagreement in one experiment.
- **Heavens-Jimenez-Lahav 2000** (MOPED): "lossless" compression — lossless
  only with true C; estimated-C information loss is ensemble-dependent.
- **Dodelson-Schneider 2013 / Percival et al. 2014**: the inflation *formula*
  is invariant (control); the "you need thousands of mocks" *practice* is
  ensemble-relative (structured estimators change the requirement).

### Climate / meteorology / oceanography
- **Rule N** (Preisendorfer-Mobley 1988; Wilks 2016 repair): retained modes vs
  true dimensionality; over-retention on smooth-spectrum ensembles.
- **Ribes et al. 2009/2013** regularized optimal fingerprinting: shrinkage vs
  EOF truncation ranking and CI calibration for attribution beta.
- **Bretherton et al. 1999** effective spatial degrees of freedom: plug-in
  N_eff bias by spectral shape.

### Ecology / evolution
- **Pollock et al. 2014** JSDM residual correlations as species interactions:
  sign-recovery AUC by residual-structure ensemble.

### Genomics / neuroscience / chemometrics
- **Zhang-Horvath 2005** WGCNA scale-free topology criterion: fraction of
  draws where R^2 > 0.8 is achievable at all, and module recovery at the
  criterion-selected power, by ensemble.
- **Smith et al. 2011** netsim: partial correlation beats full correlation for
  edge detection — demonstrated on sparse DCM networks; expect flips on
  factor/equicorr ensembles. (Partially covered by `glasso_sweep.py`.)
- **Bulik-Sullivan et al. 2015** LD score regression intercept: non-unit
  intercepts without confounding under block/hierarchical LD (stylized audit
  only; far from GWAS scale).
- **van den Heuvel et al. 2017** proportional thresholding of connectomes:
  spurious-edge rate by mass of the correlation distribution near zero.
- **Haaland-Thomas 1988 / Frank-Friedman 1993** PLS vs PCR vs ridge: winner
  depends on (Sigma_X, beta) joint ensemble.
- **Engemann-Gramfort 2015** automated covariance model selection (MNE):
  the CV winner table reshuffles by ensemble (invariant core: regularized
  beats empirical).

### ML / statistics / econometrics / engineering
- **Ledoit-Wolf 2012/2020** nonlinear shrinkage beats linear: edge depends on
  spectral dispersion; can lose at n=30 on near-flat spectra.
- **Fan-Liao-Mincheva 2013** POET: PC-removal subtracts noise-fit factors on
  no-factor ensembles; chosen K distribution per ensemble.
- **Bickel-Levina 2008 / Cai-Zhang-Zhou 2010** banding/tapering: authors state
  the bandable condition (honest-conditional case); audit the cost when
  practitioners drop it, incl. permutation of variable order.
- **Engle 2002** DCC beats CCC: overfitting penalty under constant-correlation
  truth is ensemble-dependent.
- **Chen et al. 2010** OAS dominates LW: demonstrated largely on AR(1)-like
  truths; ranking vs sphericity by ensemble.
- **Johnstone-Lu 2009** sparse PCA: needs a sparse-spike ensemble as positive
  control; fails on all 15 standard ensembles (none produce sparse
  eigenvectors).
- **Won et al. 2013** condition-number regularization: right when true
  condition numbers are bounded, wrong when the ensemble genuinely produces
  huge ones.
- **Higham 2002** nearest correlation matrix: the *repair-is-harmless* reading
  is ensemble-relative (near-boundary ensembles concentrate repair in the
  smallest eigenvalues, which dominate inverse uses).

## Null controls — ensemble-invariant theorems (audit pipeline sanity checks)
- **Hartlap-Simon-Schneider 2007** debiasing factor: exact inverse-Wishart
  identity, holds for every true C. Finding ensemble-dependence = bug.
  DONE: `hartlap_sweep.py` confirms invariance.
- **Dodelson-Schneider 2013** inflation formula (see above).
- **Sellentin-Heavens 2016** t-likelihood (invariant for Wishart estimates;
  relative when fed shrunk/tapered estimates).
- **Lowry et al. 1992** MEWMA noncentrality invariance (known Sigma).
- **Kessy-Lewin-Strimmer 2018** ZCA-cor optimality (exact identity for fixed
  Sigma; the plug-in whitening practice is relative).

Taxonomy: claims imposing *structure* on the truth are ensemble-relative;
theorems about *Wishart sampling noise* are invariant. Classify before
auditing; the most instructive failures are invariant theorems wrapped in
ensemble-relative plug-in practice.
