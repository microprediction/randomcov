# Citation audit

Every reference in `refs.bib` was checked twice in September 2026: once for
bibliographic metadata against Crossref, OpenAlex and publisher records, and
once for whether the cited work actually says what `main.tex` attributes to
it. Eighteen claim-level errors were found and fixed (commits `aa87f34` and
`94ff86d`); metadata was correct everywhere except Preisendorfer, where
Mobley had been listed as a co-author rather than as the posthumous editor.

This file records how far each check got, because "we read the abstract" and
"we read the paper" are different evidence and the difference should not be
invisible. Paywalled sources were pursued through author homepages,
institutional repositories, government technical reports and the Internet
Archive. No pirate mirror was used, and no paywall or bot protection was
bypassed.

## Verification level

**PRIMARY** — the full text of the cited work was read.

`hartlap2007`, `dodelson2013`, `sellentin2016`, `norberg2009`, `archakov2021`,
`morris2019`, `boulesteix2013`, `niessl2022`, `laloux1999`, `ledoit2004`,
`ledoit2012`, `marti2020`, `alvarez2014`, `hardin2013`, `hamill2001`,
`gaspari1999`, `paz2015`, `bickel2008`, `friedman2008`, `schafer2005`,
`smith2011`, `padmanabhan2016`, `fan2013`, `frank1993`, `guillot2013`,
`hooker1995`, `wolpert1997`, `zhang2005`, `kirkpatrick2009`, `mantel1967`,
`horn1965`, `lopezdeprado2016`, `north1982`, `cheverud2007`.

Four of these took a non-obvious route: `kirkpatrick2009` from a Wayback
snapshot of the author's retired UT lab page, `zhang2005` from CiteSeerX
after both UCLA hosts went dead, `mantel1967` from a UCLA statistics course
mirror, `lopezdeprado2016` from the free SSRN working-paper version.
`north1982` is an image scan with no text layer, read page by page in the
browser and transcribed.

**PRIMARY (substitute)** — the cited work itself is closed, but the same
authors state the same result in an obtainable primary source.

- `preisendorfer1988` (Rule N) — via Overland & Preisendorfer, *Monthly
  Weather Review* 110(1):1–4 (1982), which states Rule N by name: terminate
  at the largest `m` such that the sample eigenvalue fraction exceeds the
  95th percentile of the null spectrum simulated from uncorrelated Gaussian
  data of the same shape. Confirms it is a per-index resampling comparison,
  not a statement about eigenvalue spacing. Rule N originates with
  Preisendorfer & Barnett (1977); citing the 1988 monograph is shorthand.
- `cheverud1996` (random skewers) — via Cheverud & Marroig, *Genetics and
  Molecular Biology* 30(2):461–469 (2007), open access, which restates the
  procedure and its null, and concedes that in practice the null "is nearly
  always rejected."
- `stein2002` (screening) — via Stein's own 2011 Rietz lecture
  (arXiv:1203.1801), which restates the spectral-density conditions.

**ABSTRACT** — the publisher abstract plus explicit secondary restatements;
the body could not be obtained legitimately.

`lowry1992`, `lkj2009`, `joe2006`, `cox1987`, `carlson1988`, `rohlf2017`,
`kaiser1960`, `houtekamer2001`, `papenbrock2021`, `cheverud1996`.

For `lkj2009` and `joe2006` the obstacle is technical rather than economic:
Unpaywall reports both as bronze OA at Elsevier, but ScienceDirect refuses
non-browser clients.

## Claims that rest on an abstract

One only: **Lowry et al. 1992**, for the MEWMA ARL depending on the mean and
covariance solely through a noncentrality parameter. The published abstract
states it verbatim and unconditionally. Secondary sources add that the result
holds for equal smoothing constants; `main.tex` does not assert that
condition, so nothing in the paper depends on it. If the condition is ever
added to the text, get the body by interlibrary loan rather than quoting a
secondary source for it.

## New references added during the audit

`laloux2000`, `ledoit2004honey`, `glorfeld1995`, `kaufman2008`, `marti2021`,
`cheverud2007`. Metadata for each was verified against Crossref before it was
added — a citation audit should not introduce unchecked citations. `ledoit2003`
was removed: it was cited for a constant-correlation shrinkage target that is
not in it, and nothing else cited it.

## What changed, in one line each

See the two commit messages for evidence and quotations. The corrections that
altered an argument rather than an attribution:

- Schäfer & Strimmer moved from proponent to baseline in the graphical-lasso
  audit — they propose the Ledoit–Wolf shrinkage estimator column 7 measures
  against, and had already run that comparison themselves.
- The Marchenko–Pastur clipping recipe reattributed from Laloux et al. 1999
  (which prescribes nothing) to Laloux et al. 2000.
- Paz & Sánchez no longer described as presuming banded truth, which they
  explicitly decline to assume.
- Rule N redescribed as a resampling test rather than an eigenvalue spacing
  law, which had conflated it with North et al.
- North et al.'s corollary — a stand-out eigenvalue implies small EOF sampling
  error — identified as the thing the audit scores. An intermediate revision
  claimed they never assert this; they do, and that revision was wrong.
