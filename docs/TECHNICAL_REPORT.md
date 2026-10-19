# LunaLens: Intelligent Lunar Satellite Observation Planning

**Technical report — executed prototype, 19 October 2026**  
**Version:** 0.1.0 · **Implementation:** Python, CPU only · **Status:** research prototype

## Abstract

LunaLens investigates whether selecting observations from available coarse
imagery can improve a lunar image map with fewer high-resolution acquisitions.
It implements image loading, interpretable feature extraction, four competing
policies, a budget-enforcing simulator, evaluation, an interactive dashboard,
and reproducible experiments. The planner never receives unrevealed reference
pixels or evaluation errors through its supported interface.

The executed benchmark contains three real LROC image derivatives and three
explicitly synthetic fixtures. At 64 observations from 256 regions, feature
selection reduced mean real-image MSE from 0.004443659 for random selection to
0.002315224: a 47.90% reduction in the ratio of group means. Halving each real
image's initial MSE required 70.0 observations on average, versus 131.6 for
random selection. The outcome was conditional: feature selection was 26.42%
worse than random on an artificial hidden-detail fixture, and the custom
diversity heuristic did not consistently improve the simpler baseline.

These measurements support a useful software experiment, not general lunar
mapping superiority or spacecraft readiness. Source data, exact configurations,
full histories, failures, and uncertainty about generalization are retained.
The repository is an AI-assisted foundation; student authorship of the supplied
implementation is not asserted.

## 1. Motivation and research formulation

The project models a decision that arises when acquiring every possible image
region is unaffordable: which available footprint should receive a better
observation next? Its computer-science focus is sequential selection under
limited information. Resource limits are represented by abstract observation
costs; orbital opportunities and spacecraft hardware are outside this model.

Let the normalized reference image be $T\in[0,1]^{H\times W}$, the initial
available map be $L$, and disjoint observation regions be $R_1,\ldots,R_N$.
For a selected set $O$, reconstruction is

$$
M_O(p)=\begin{cases}T(p),&p\in\bigcup_{i\in O}R_i,\\L(p),&\text{otherwise}.\end{cases}
$$

The experiment asks whether a policy using permitted observations can reduce
$\mathrm{MSE}(T,M_O)$ under $\sum_{i\in O}c_i\le B$, compared with random
and systematic acquisition. The reference is available to the simulator and
evaluator, but unavailable to the decision rule before the corresponding reveal.
Consequently, a planner estimates utility rather than optimizing known true
error. There is no claim of a globally optimal policy.

Variance, entropy, filtering, weighted ranking, and full-reference image
metrics are established mathematical tools. The particular seven-feature
combination, preset weights, diversity rule, degradation settings, and
experimental interface are project-specific engineering choices. They are
heuristics rather than a new physical model or validated uncertainty estimator.

## 2. Data and image preparation

### 2.1 Authentic and synthetic inputs

The three authentic samples derive from NASA publications identifying Lunar
Reconnaissance Orbiter Narrow Angle Camera imagery, credited
**NASA/GSFC/Arizona State University**. Source publications and product IDs are
listed below; processing records are in [the attribution document](../data/ATTRIBUTION.md)
and [manifest](../data/manifest.json).

| Real sample | Product | Primary source |
| --- | --- | --- |
| Bouldery crater near Mare Australe | PIA14421; frame M150062296R | [NASA publication][nasa-australe] |
| Giordano Bruno impact melt | PIA12932 | [NASA publication][nasa-bruno] |
| The floor of Tycho | PIA12941 | [NASA publication][nasa-tycho] |

The cached source JPEGs are 1000×1000 pixels. Each benchmark input is a fixed
central 512×512 grayscale crop, beginning at `(x=244, y=244)`. Source choices and
the crop rule were fixed before policy comparison. URLs, retrieval timestamps,
source-byte hashes and derived-file hashes make the transformation auditable.
These are published outreach derivatives, with JPEG and display-processing
limitations; they are not calibrated or georeferenced PDS products. Here,
“ground truth” means the reference image used for reconstruction evaluation,
not surveyed terrain geometry or radiometrically exact lunar reflectance.

Three procedural fixtures supplement the real samples: `mixed`, seed 11;
`rugged`, seed 29; and `hidden_detail`, seed 47. The [generator](../src/synthetic.py)
combines multiscale filtered noise and elliptical rim/bowl intensity patterns.
The stress fixture replaces pixels to the right of normalized x=0.55 with
alternating bright/dark checker detail, largely removed by degradation. It was
designed to expose observability failure, not to mimic a typical lunar scene.
All synthetic imagery is labelled as such. [Dataset gallery](../evidence/figures/dataset_gallery.png).

### 2.2 Loading and fixed intensity normalization

The [loader](../src/image_loader.py) accepts local paths, bytes, and binary file
objects for PNG, JPEG and TIFF. TIFF reads its first page, preserves unsigned
16-bit values, and handles planar channels; additional compressed TIFF codecs
may require `imagecodecs`. Unsupported multispectral or signed scientific data
requires explicit preprocessing. EXIF orientation is applied to PNG/JPEG.

Unsigned integer pixels are divided by their dtype maximum: 255 for uint8,
65535 for uint16. Float input must already be finite and within [0,1]. Stored
RGB intensities become grayscale through
$Y=0.2126R+0.7152G+0.0722B$; this is not radiometric linearization.
Alpha is composited on black. Optional maximum-dimension reduction uses
bilinear resizing with antialiasing and preserves aspect ratio. The result is
a detached two-dimensional float32 array. File and pixel limits bound uploads.
There is no image-dependent contrast stretch or reference-fitted intensity
normalization that would silently change the evaluation target.

### 2.3 Degradation and segmentation

The [initial measurement surrogate](../src/preprocessing.py) is

$$
L=\operatorname{clip}_{[0,1]}
\left(U_{\mathrm{bilinear}}\left(A_d\left(G_\sigma*T\right)\right)\right).
$$

Gaussian blur uses reflected boundaries and $\sigma=1$ reference-grid pixel.
Area averaging (`resize_local_mean`, `grid_mode=True`) reduces the image to
$(\lceil H/d\rceil,\lceil W/d\rceil)$, with $d=4$. Bilinear upsampling
returns it to the reference dimensions using reflected boundaries and no
additional antialiasing. Thus the benchmark begins with 128×128 coarse samples
displayed on a 512×512 grid. Factor one and sigma zero form an exact identity.
This operator is explicit and reproducible, but is not a calibrated LROC sensor.

Square tiling uses 32×32 regions in row-major order, producing 256 regions per
benchmark image. Arbitrary input shapes retain partial boundary tiles without
padding. Frozen geometry contains IDs, coordinates, dimensions and cost;
features and observation status occupy aligned arrays. Uniform cost is one
unit per tile. An optional area cost, $c_i=|R_i|/32^2$ for this tile size,
is available in the simulator/dashboard but is not used in the main benchmark.

## 3. Interpretable image analysis

All seven planner descriptors are calculated **once from the initial map**.
Full-map filtering precedes slicing, avoiding artificial boundaries between
tiles. Let $g$ be the magnitude of horizontal and vertical Sobel responses,
each divided by eight; let $q=L-G_1*L$.

| Feature | Implemented regional value | Default weight |
| --- | --- | ---: |
| Variance | Population mean squared deviation from mean intensity | 0.10 |
| Edge density | Fraction of pixels with $g>0.04$ | 0.15 |
| Gradient | Mean $g$ | 0.15 |
| Texture | $\sqrt{\operatorname{mean}(q^2)}$ | 0.20 |
| Entropy | $-\sum_b p_b\log_2p_b$, from 32 fixed intensity bins | 0.15 |
| Contrast | 95th minus 5th intensity percentile | 0.10 |
| Information | $(\mathrm{entropy}/5)(\mathrm{contrast}+\mathrm{edge\ density})/2$ | 0.15 |

Variance and contrast describe intensity spread. Edges and gradients describe
visible transitions. High-pass RMS describes surviving fine texture. Histogram
entropy describes intensity diversity, with range zero to five bits, but
discards spatial ordering. “Information” is a bounded composite proxy; it is
neither measured information gain nor calibrated uncertainty. Its reuse of
other terms makes feature dependence especially relevant to interpretation.

Each raw column is min-max normalized across the initial image:

$$
f_{ij}=\frac{x_{ij}-\min_kx_{kj}}{\max_kx_{kj}-\min_kx_{kj}}.
$$

Constant columns become zero. Normalization remains fixed during acquisition
and uses no reference-derived descriptors. It makes heterogeneous statistics
comparable within an image, but does not calibrate values between images.
The exact definitions are in [feature extraction](../src/feature_extraction.py).

Optional exploration additionally displays intensity below 0.15, Sobel
boundaries, and local high-pass RMS above the greater of 0.015 and its image
75th percentile. Canny/Hough circle candidates operate on an image bounded to
256 pixels per longest dimension, using radii 5–40 pixels where dimensions
allow, and return at most twelve peaks. These displays are not planner inputs.
Darkness is not confirmed shadow depth; texture is not physical roughness;
circles are not verified craters. None establishes slope, elevation, hazards
or landing safety.

## 4. Selection policies and permitted information

Random selection samples uniformly among currently affordable, unobserved
tiles, using a local NumPy generator. Systematic selection takes the smallest
available affordable region ID, corresponding to row-major acquisition.
Neither uses the image descriptors.

The feature baseline computes

$$
u_i=\sum_{j=1}^{7}w_jf_{ij},\qquad s_i=u_i/c_i.
$$

Weights are finite, nonnegative, and normalized to sum one; at least one must
be positive. A supplied mapping replaces defaults, with omitted names assigned
zero. Observed or unaffordable regions have score negative infinity. Maximum
finite score wins, with smallest-ID tie breaking. This score assumes visible
structure predicts the value of recovering missing detail; the experiments
explicitly test, rather than assume, that relationship.

The isolated [custom strategy](../src/custom_strategy.py) adds descriptor novelty
and spatial dispersion. For previously observed IDs $O_t$, centres $p_i$,
and image diagonal $D$,

$$
n_i=\min_{k\in O_t}\lVert f_i-f_k\rVert_2/\sqrt7,\qquad
d_i=\min_{k\in O_t}\lVert p_i-p_k\rVert_2/D,
$$

$$
s_i=(0.60u_i+0.25n_i+0.15d_i)/c_i.
$$

Both additional terms are zero before the first observation. Their coefficients
are configurable and normalized; novelty and distance update after each reveal.
Descriptors remain the initial ones. The supplied policy uses which regions
were acquired, without learning from their newly revealed residuals. Diversity
may improve sampling breadth while sacrificing immediate reconstruction gain.

Every policy implements `scores(state)` and `choose_next(state, rng)`, returning
a region ID or `None`. [PlannerState](../src/observation_planner.py) contains the
initial map, current reconstruction, normalized initial descriptors, frozen
region geometry/costs, observation mask, remaining budget and selected-ID
history. Ground truth, error maps, and evaluation records are absent. Arrays
are detached copies backed by immutable bytes, so retained snapshots cannot
change and their writable flags cannot be reenabled.

This interface prevents accidental leakage in trusted strategies. It is not a
security sandbox against a hostile Python plugin reading reference files or
inspecting execution frames. A future policy may use discrepancies inside
already observed tiles, because those pixels have been revealed; it may not
read the corresponding discrepancy elsewhere. Counterfactual and spy-policy
tests exercise precisely this boundary.

## 5. Simulation, metrics and architecture

The [simulator](../src/simulator.py) independently checks each returned ID,
repeat status, cost and remaining budget. It replaces the selected tile exactly,
records metrics, and sends a fresh permitted snapshot to the next decision.
The loop is bounded by the number of regions. Excess budgets stop at complete
coverage; fractional remainders can be unusable. Area-mode remainder arithmetic
uses integer observed pixel counts to avoid losing a final partial tile through
floating subtraction roundoff. A zero budget cannot purchase a positive-cost tile.

For $P=HW$, evaluation accumulates errors in float64:

$$
\mathrm{MAE}=P^{-1}\sum_p|T_p-M_p|,\quad
\mathrm{MSE}=P^{-1}\sum_p(T_p-M_p)^2,\quad
\mathrm{PSNR}=-10\log_{10}(\mathrm{MSE}).
$$

PSNR uses data range one and becomes positive infinity for equality. Exact
replacement makes MAE/MSE nonincreasing, but cannot guarantee monotonic SSIM.
SSIM compares local luminance, contrast and covariance; its established
formulation is described by [Wang et al.][ssim-paper]. LunaLens uses
`skimage.metrics.structural_similarity` with `data_range=1`, a **uniform 7×7
window**, `gaussian_weights=False`, and `use_sample_covariance=True`. Smaller
images use the largest valid odd window up to seven; dimensions below three
produce NaN. These settings are recorded explicitly and should not be mistaken
for a reproduction of the original paper's parameterization. See the
[official API documentation][skimage-metrics] and [metric code](../src/evaluation.py).

History includes observation zero and each reveal: count, accumulated cost,
pixel coverage, four quality metrics, $\Delta_t=\mathrm{MSE}_0-\mathrm{MSE}_t$,
relative reduction $\Delta_t/\mathrm{MSE}_0$, and cumulative gain per observation
$\Delta_t/t$. Zero denominators produce zero gain. The last quantity is not
the marginal benefit of the most recent acquisition.

The dashboard and CLI call the same simulator. Loading, features, policies,
truth-owning simulation, pure evaluation and visualization occupy separate
modules, documented in [ARCHITECTURE.md](ARCHITECTURE.md). Streamlit provides
Overview, Image Explorer, Observation Planner, Results and Algorithm Explorer;
uploads, adjustable weights, simulations, comparisons and exports operate on
actual arrays. Input fingerprints invalidate stale results, and source changes
require restarting the server to avoid mixing cached algorithms with edited code.

## 6. Experimental protocol and statistical interpretation

Every image uses the same 512-pixel dimensions, 32-pixel regions, factor-four
degradation, sigma one, uniform costs and fixed weights. Random runs use seeds
0–4. The three deterministic methods execute once per image: six images times
eight runs gives **48 full-scan runs**, each recording 257 states. Seven
single-feature-drop ablations at 64 observations add **42 runs**. Dropped
weights become zero and the remainder are renormalized. No coefficients were
tuned after seeing these results.

Matched endpoints are 0, approximately 10%, 25%, 50%, 75% and 100% of total
cost: 0, 25, 64, 128, 192 and 256 observations. Thus the nominal 10% endpoint
actually covers 9.765625%. Every intermediate observation remains available.
Current policies have budget-invariant prefixes under uniform costs, tested
against separately capped runs. Default smaller-budget endpoints can therefore
be recovered from full scans. A future rule using remaining budget to change
priorities requires `--independent-budgets`: that reruns interior endpoints at
their actual caps. Full-scan curves remain trajectories, not independently
replanned outcomes at every possible cap.

Random seeds are averaged **within each image first**. Kind-level summaries
then give each of the three real or three synthetic images equal weight.
Deterministic results are not duplicated to create artificial sample sizes.
CSV standard deviations describe their stated sampling level; three image
means or five random orders do not establish population performance or
statistical significance. No significance claim or confidence interval is made.

Mean MSE, mean per-image relative improvement and mean PSNR answer different
questions. Nonlinear ratios and logarithms must not be moved across averaging.
The following tables identify their denominator and use rounded display values;
the linked CSVs retain full precision. The complete [benchmark summary](../evidence/results/summary.json) records the executed run counts and configuration.

## 7. Executed results

### 7.1 Matched quarter-budget quality

At 64 observations, cost and spatial coverage are identical across methods:
64 units and 25%. Random entries average five seeds within each image before
averaging images. The final column averages improvement relative to **each
image's own initial MSE**. Source: [group summary](../evidence/results/summary_by_kind.csv).

| Kind | Method | Mean MAE | Mean MSE | Mean PSNR (dB) | Mean SSIM | Mean initial-MSE reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Real | random | 0.027337 | 0.004443659 | 27.267 | 0.786873 | 24.50% |
| Real | systematic | 0.027208 | 0.004291926 | 27.289 | 0.783484 | 24.87% |
| Real | feature | 0.019104 | 0.002315224 | 29.106 | 0.815390 | 49.53% |
| Real | custom | 0.019229 | 0.002347051 | 29.080 | 0.815194 | 49.31% |
| Synthetic | random | 0.027145 | 0.003863555 | 31.369 | 0.834627 | 24.07% |
| Synthetic | systematic | 0.026496 | 0.003760048 | 31.453 | 0.834648 | 25.58% |
| Synthetic | feature | 0.031383 | 0.004804113 | 32.410 | 0.808015 | 36.16% |
| Synthetic | custom | 0.031384 | 0.004803999 | 32.414 | 0.807924 | 36.20% |

For real images, $1-0.002315224358/0.004443658889\approx47.90\%$: the feature
policy reduces the ratio of mean MSE relative to random. It is also 46.06%
lower than systematic mean MSE. Averaging the three paired relative reductions
against random instead gives **33.08%**. The larger group-mean ratio reflects
Mare Australe's much larger absolute error scale, not three uniformly 48% gains.

The custom policy's real mean MSE is **1.37% higher** than the feature baseline
at this endpoint. It slightly improves Giordano Bruno and some synthetic values,
so the correct conclusion is that the added terms provide no consistent
advantage in this experiment. They are not justified by their sophistication alone.

### 7.2 Per-image outcomes and the stress failure

The table reports MSE. “Feature reduction” is
$100(1-\mathrm{MSE}_{\rm feature}/\overline{\mathrm{MSE}}_{\rm random})$;
a negative value means feature selection is worse. Source:
[image summaries](../evidence/results/summary_by_image.csv).

| Image | Random | Systematic | Feature | Custom | Feature reduction vs random |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mare Australe (real) | 0.011616800 | 0.011115822 | 0.005690774 | 0.005782435 | +51.01% |
| Giordano Bruno (real) | 0.000450088 | 0.000445129 | 0.000366963 | 0.000365429 | +18.47% |
| Tycho floor (real) | 0.001264088 | 0.001314827 | 0.000887936 | 0.000893291 | +29.76% |
| Mixed (synthetic) | 0.000143025 | 0.000143784 | 0.000080428 | 0.000080267 | +43.77% |
| Rugged (synthetic) | 0.000242714 | 0.000233727 | 0.000166108 | 0.000165927 | +31.56% |
| Hidden detail (synthetic) | 0.011204928 | 0.010902634 | 0.014165802 | 0.014165802 | -26.42% |

Feature MSE beats both conventional baselines on all three real samples, but
the Giordano Bruno SSIM result disagrees: feature **0.850266**, random
**0.851522**, systematic **0.852907**. Pixel error and local structural similarity
weight distortions differently. This measured disagreement prevents a blanket
claim of superior image quality across every metric.

The hidden-detail fixture is a substantial failure: feature MSE **0.014165802**
versus random **0.011204928**, or **26.42% worse**. Its fine checker pattern is
largely invisible in the coarse map, so visible-structure scoring spends early
observations elsewhere. This is evidence against assuming visible complexity
always predicts recoverable error.

Across all synthetic fixtures, feature mean MSE is **24.34% worse** than random,
yet its mean normalized initial-MSE reduction is **36.16%**, versus **24.07%**.
There is no mathematical contradiction: normalization gives each scene's
fractional gain equal weight, whereas absolute MSE remains strongly affected
by the stress scene's much larger initial error. Both summaries are useful;
reporting only the favorable one would conceal a material limitation.

![Executed MSE trajectories, with real and synthetic groups separated](../evidence/figures/07_error_vs_budget.png)

### 7.3 Observations needed to reach a quality target

For each trajectory, the diagnostic finds the first observation reaching at
least 50% reduction of that image's initial MSE, then averages random seeds
within image. Fractional counts are averages of integer stopping counts.
Sources: [per-image targets](../evidence/results/observations_to_targets_by_image.csv)
and [kind targets](../evidence/results/observations_to_targets_by_kind.csv).

| Image/group | Random | Systematic | Feature | Custom |
| --- | ---: | ---: | ---: | ---: |
| Mare Australe (real) | 136.2 | 126.0 | 49.0 | 50.0 |
| Giordano Bruno (real) | 128.0 | 139.0 | 93.0 | 94.0 |
| Tycho floor (real) | 130.6 | 151.0 | 68.0 | 69.0 |
| **Real mean** | **131.60** | **138.67** | **70.00** | **71.00** |
| Mixed (synthetic) | 132.6 | 136.0 | 47.0 | 47.0 |
| Rugged (synthetic) | 137.2 | 132.0 | 69.0 | 69.0 |
| Hidden detail (synthetic) | 138.2 | 128.0 | 199.0 | 199.0 |
| **Synthetic mean** | **136.00** | **132.00** | **105.00** | **105.00** |

The real-image average falls from 131.6 random observations to 70.0 feature
observations: **46.81% fewer** at this particular target. It is not a promise
that the same reduction holds for another target or image. On hidden detail,
feature selection needs 199 observations versus random's 138.2. At full
coverage all methods reproduce the reference exactly; endpoint equality is
guaranteed by the reveal model and does not demonstrate useful prioritization.

### 7.4 Why the ranking works or fails

Completed-run diagnostics correlate initial feature scores with true initial
tile MSE. The true errors are calculated **after planning** and never supplied
to policies. A high positive Spearman coefficient supports alignment of ranking
with the measured objective on that image; it does not establish causation or
generalization. The negative stress coefficient captures the opposite ordering.

| Image | Spearman rho | Measured feature MSE reduction | Evaluation-only maximum reduction |
| --- | ---: | ---: | ---: |
| Mare Australe (real) | +0.958138 | 62.67% | 67.58% |
| Giordano Bruno (real) | +0.621198 | 38.58% | 43.98% |
| Tycho floor (real) | +0.813431 | 47.34% | 56.69% |
| Mixed (synthetic) | +0.854981 | 58.46% | 62.24% |
| Rugged (synthetic) | +0.820238 | 47.37% | 53.36% |
| Hidden detail (synthetic) | -0.631643 | 2.64% | 55.78% |

The final column is an evaluation-only upper bound obtained by sorting true
per-tile MSE contributions and selecting the largest 64. It is exact for the
equal-cost, disjoint, exact-reveal MSE problem, but sees hidden truth and is not
a deployable strategy. The stress policy recovers only **2.64%** of initial MSE
at 64 observations, while the bound is **55.78%**.

Under uniform random selection of $k$ of $N$ tiles, each tile has inclusion
probability $k/N$. Summing its fixed recoverable error therefore gives an
expected fractional MSE reduction of $k/N$, when initial MSE is positive.
The exact expectation at 64/256 is 25%; five realized random orders need not
equal it. This reference helps interpret both positive and adverse outcomes.
Sources: [correlations](../evidence/results/score_error_correlations.csv),
[evaluation-only bounds](../evidence/results/evaluation_only_bounds.csv), and
[diagnostic figure](../evidence/figures/score_error_diagnostics.png).

### 7.5 Feature-drop ablations

Each cell below averages **per-image percentage MSE change relative to the
full weighted feature policy**, at 64 observations. Positive means dropping
the feature worsened MSE. This is not percentage change of the group mean MSE.
Source: [ablation summary](../evidence/results/ablations_by_kind.csv).

| Dropped feature | Real mean relative MSE change | Synthetic mean relative MSE change |
| --- | ---: | ---: |
| variance | -1.80% | +0.13% |
| edge_density | +0.32% | +0.47% |
| gradient | +0.70% | +0.54% |
| texture | +3.08% | +3.72% |
| entropy | -2.03% | -0.79% |
| contrast | -2.04% | -0.66% |
| information | -2.04% | -0.35% |

Dropping texture increased real-image MSE by 3.08% on this paired measure;
dropping entropy, contrast or information improved it by about 2.03%. Therefore
the default combination is not empirically optimal even here. Correlated terms,
renormalized weights and discrete rank changes prevent interpreting these
drops as independent causal contributions. The original defaults were retained;
selecting new weights from these outcomes would require fresh held-out images.

## 8. Verification and reproducibility

Automated checks cover loading and fixed scaling, complete segmentation,
mathematical feature examples, ranking, immutable snapshots, invalid parameters,
seeds, budget enforcement, exact reconstruction, metric values, aggregation,
prefix assumptions and real Streamlit actions/exports. Counterfactual tests
change unrevealed reference pixels while keeping permitted information equal;
decisions must match until allowed feedback differs. A spy policy checks every
received snapshot, including old snapshots retained across updates. See
[VALIDATION.md](VALIDATION.md) for the final executed installation and test record.

From the repository root, the tested workflow is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
python -m experiments.run_experiments
python -m experiments.diagnose_scores
python -m experiments.compare_algorithms
python -m streamlit run app.py
```

The delivered benchmark records Python 3.12.14 and its exact dependency versions,
48 full scans, 42 ablations, all input/source hashes, command, timestamps and
completion status in [reproducibility.json](../evidence/results/reproducibility.json).
It completed in about five minutes on the execution CPU; this is not a guaranteed
laptop timing. Full-image SSIM and copied snapshots scale with image size and
observation count; the custom minimum-distance calculation also grows with
observation history. No GPU or trained model is used.

Bundled data makes execution offline after dependencies are installed. A first
installation still needs package-index access or cached wheels. Python 3.11+
is the declared target; separate Windows, macOS and Python 3.11 executions are
not claimed. The [README](../README.md) gives platform commands, smoke runs,
offline sample regeneration and the independent-budget option. Numeric evidence
and PNGs, rather than illustrative dashboard screenshots, support this report.

## 9. Limitations and next engineering contributions

The dataset is small and intentionally heterogeneous. Three outreach crops
cannot establish performance across illumination, terrain types, acquisition
scales or instruments. No learned parameters or training split are needed for
the fixed baseline, but once a student tunes on these examples they cease to
be an untouched evaluation set. Future comparisons should hold out distinct
source scenes, not merely nearby crops.

The simulator assumes aligned, instantaneous, noiseless reveals of an unchanged
surface, with exact replacement and no overlapping footprints. It models no
orbit, visibility windows, attitude slews, illumination changes, image motion,
compression, downlink queues or storage measured in bytes. MSE measures image
reconstruction, not geological usefulness, crater-detection accuracy or safe
landing terrain. Optional visual proxies have no ground-labelled validation.

Useful next contributions are to define a defensible acquisition objective and
cost model; develop an observable predictor of missing detail; and choose an
exploration rule with a predeclared test. A small online model could learn from
already revealed tile residuals, with explicit uncertainty and leakage tests.
Later work could add labelled geological objectives, sensor noise, calibrated
data, or feasible scheduling constraints. These are proposals, not implemented
capabilities or measured improvements.

[CUSTOM_STRATEGY_GUIDE.md](CUSTOM_STRATEGY_GUIDE.md) provides equations, an
API-preserving example and concrete experiments. [PORTFOLIO_SUMMARY.md](PORTFOLIO_SUMMARY.md)
contains editable personal-contribution placeholders. The supplied foundation
was substantially AI assisted. A student should document the decisions, code,
validation and interpretation they personally perform, including negative
results, rather than claim independent authorship of the entire baseline.

## Primary references

1. NASA. [Bouldery Crater near Mare Australe][nasa-australe]. PIA14421;
   LRO/NAC; NASA/GSFC/Arizona State University.
2. NASA. [Impact Melt Flows on Giordano Bruno][nasa-bruno]. PIA12932;
   LRO/NAC; NASA/GSFC/Arizona State University.
3. NASA. [The Floor of Tycho][nasa-tycho]. PIA12941;
   LRO/NAC; NASA/GSFC/Arizona State University.
4. scikit-image. [Metrics API documentation][skimage-metrics]. Implementation
   options and explicit floating-point data-range guidance.
5. Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. (2004).
   [Image Quality Assessment: From Error Visibility to Structural Similarity][ssim-paper].
   *IEEE Transactions on Image Processing*, 13, 600–612.

[nasa-australe]: https://science.nasa.gov/photojournal/bouldery-crater-near-mare-australe/
[nasa-bruno]: https://science.nasa.gov/photojournal/impact-melt-flows-on-giordano-bruno/
[nasa-tycho]: https://science.nasa.gov/photojournal/the-floor-of-tycho/
[skimage-metrics]: https://scikit-image.org/docs/stable/api/skimage.metrics.html
[ssim-paper]: https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
