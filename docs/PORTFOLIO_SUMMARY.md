# LunaLens — portfolio summary

**Project Name:** LunaLens — Intelligent Lunar Satellite Observation Planning  
**Role:** Independent Project Developer  
**Context:** Personal computer-science engineering project for a UNSW BlueSat portfolio  
**Current stage:** Executed software prototype and baseline experimental foundation

## Project objective

Investigate whether an interpretable observation-selection algorithm can
improve a lunar image map using fewer high-resolution observations than random
selection or systematic scanning. The project treats observation budget as a
finite resource and explicitly separates information available to the planner
from reference imagery used to reveal observations and evaluate reconstruction.

## Technical approach

The Python pipeline loads PNG/JPEG/TIFF images into a fixed grayscale range,
constructs a Gaussian-blurred and downsampled initial measurement, and divides
the image into configurable observation regions. It extracts seven interpretable
image descriptors: variance, edge density, gradient magnitude, high-frequency
texture, entropy, contrast, and a defined composite information proxy.

Four implemented policies choose observations: seeded random selection,
row-major scanning, configurable weighted feature utility, and a separate
custom example combining utility with descriptor novelty and spatial
dispersion. The simulator reveals exact reference pixels only for selected
regions and enforces the observation budget independently of the policy.

Evaluation records MAE, MSE, PSNR, SSIM, cost, coverage, and improvement per
observation. An executable benchmark compares identical budgets across three
authentic LROC image crops and three explicitly synthetic fixtures. The
Streamlit dashboard uses the same simulator and exports real numeric results,
reconstructions, metadata, and figures.

## Implementation provenance

**This delivered foundation was implemented with AI assistance.** The baseline
algorithms are transparent engineering choices built from established image
processing and evaluation methods. The provided custom diversity rule is an
initial editable example, not a claimed research invention or a statement of
the student's independent contribution.

The following fields are intentionally incomplete. Replace them only after
doing the work, and retain evidence such as commits, notebooks, tests, or
experiment manifests. Do not describe this entire foundation as code you
independently wrote unless that becomes accurate through your own work.

## Personal contributions — complete accurately

| Contribution | Your description | Proof |
| --- | --- | --- |
| Problem and objective | `[Explain the objective you chose and why it matters.]` | `[Design note / decision date]` |
| Algorithm design | `[Describe your own strategy, assumptions, and trade-offs.]` | `[File and commit / derivation]` |
| Implementation | `[List the modules or changes you personally implemented.]` | `[Commits / code review]` |
| Experimental design | `[Describe new data, holdouts, budgets, metrics, and hypotheses you selected.]` | `[Experiment config / preregistered note]` |
| Validation | `[Explain a bug you found, how you proved it, and the test you added.]` | `[Failing/passing test or issue]` |
| Interpretation | `[Explain a result that surprised you and how it changed your design.]` | `[Measured table / report revision]` |

## Genuinely measured results

Default experiments use 512×512 maps, 32×32 regions, factor-four coarse
sampling, Gaussian sigma 1, uniform observation costs, and random seeds 0–4.
There are 256 regions per image; the table below uses 64 observations.
Random results are averaged within each image before image-level aggregation.

| Real lunar samples, mean across 3 images | MSE ↓ | SSIM ↑ | Reduction of each image's initial MSE, averaged ↑ |
| --- | ---: | ---: | ---: |
| Random | 0.004444 | 0.7869 | 24.50% |
| Systematic | 0.004292 | 0.7835 | 24.87% |
| Feature weighted | 0.002315 | 0.8154 | 49.53% |
| Custom diversity example | 0.002347 | 0.8152 | 49.31% |

The feature method's mean MSE was **47.90% lower than random** on the three
real examples. To reduce initial MSE by at least half, it needed **70.0
observations on average**, versus **131.6** for random and **138.7** for
systematic scanning. The custom example needed **71.0** and did not improve
overall on the feature baseline. These sample means are not a significance
claim or a general estimate of mission performance.

The failure is equally relevant: on the deliberately artificial hidden-detail
fixture, the feature and custom methods reduced initial MSE by only **2.64%**
at 25% observation budget. Random achieved **22.99%** and systematic **25.07%**.
Feature selection had **26.42% higher MSE than random** on that fixture, and
**24.34% higher mean MSE across all three synthetic fixtures**. The low-resolution
map hid the high-frequency detail most in need of observation.

On the real Giordano Bruno sample, feature selection improved MSE while its
SSIM was slightly below the baselines. This demonstrates why a portfolio
claim should name its metric and dataset. Dropping the texture feature raised
real-image MSE by an average **3.08%** relative to the all-feature method;
dropping some correlated features improved it slightly. Defaults were retained
instead of tuning them on the evaluation examples.

Sources for all measurements are the CSVs under `evidence/results/`, especially
`summary_by_image.csv`, `summary_by_kind.csv`, `ablations_by_kind.csv`, and
`observations_to_targets_by_kind.csv`. `reproduction_check.json` records identical
numeric outputs across three full benchmark executions and matching runtime
source hashes.

## Limitations

The simulator uses aligned, noiseless pixel replacement and abstract cost
units. It has no orbit propagation, access windows, attitude dynamics,
illumination changes, storage-in-bytes model, or downlink scheduling. It is
not flight software. The real samples are published JPEG-derived LROC images,
not calibrated geospatial PDS products. Three real examples are insufficient
for broad performance claims, and those examples become development data if
you use them to tune a new policy.

Entropy and texture are image statistics. The code's circle candidates,
darkness masks, and rough-looking patterns do not verify craters, physical
roughness, elevation, slope, or landing safety. The composite information
feature is not calibrated uncertainty or expected scientific information gain.
The default custom strategy changes with observation history but does not
learn a model from the revealed pixel content.

## Future work — proposed, not completed

The most useful next contribution is a policy that balances exploration with
feature ranking and learns from legitimately revealed patches. One hypothesis
is that a small, seeded exploration phase will find detail that the initial
map suppresses, allowing later priorities to adapt. Test it against the
current custom example, pure weighted scoring, and random exploration under
identical budgets.

Further extensions include an interpretable regression model for expected
reconstruction error, an uncertainty proxy that is calibrated on held-out
observations, active-learning acquisition functions, adaptive region sizes,
and realistic imaging/downlink constraints. Each requires new experiments.
A learning model may use already revealed patches; it must not read unrevealed
reference pixels or the evaluator's diagnostic files.

## Proof and demonstration assets

| Evidence | Location |
| --- | --- |
| Runnable dashboard | `app.py`; `python -m streamlit run app.py` |
| Simulator and baseline policies | `src/simulator.py`; `src/observation_planner.py` |
| Isolated personal strategy starting point | `src/custom_strategy.py` |
| Original and degraded imagery | `evidence/figures/01_original_and_degraded.png` |
| Segmentation, feature, and priority maps | `evidence/figures/02_observation_segmentation.png`; `03_information_heatmap.png`; `04_observation_priorities.png` |
| Selected observations and reconstructed maps | `evidence/figures/05_selected_observation_regions.png`; `06_reconstructed_maps.png` |
| Measured comparison and table | `evidence/figures/07_error_vs_budget.png`; `08_measured_metrics_table.png` |
| Feature ablation | `evidence/figures/09_feature_drop_ablation.png` |
| Same-quality observation counts | `evidence/figures/observations_to_half_error.png` |
| Genuine failure diagnosis | `evidence/figures/score_error_diagnostics.png` |
| Executed experiment histories and hashes | `evidence/results/`; `python -m experiments.run_experiments` |
| Automated and installation validation | `tests/`; `docs/VALIDATION.md` |
| Technical explanation | `docs/TECHNICAL_REPORT.md`; `docs/ARCHITECTURE.md`; `docs/CUSTOM_STRATEGY_GUIDE.md` |

Real imagery credit: NASA/GSFC/Arizona State University; detailed URLs and
processing records are in `data/ATTRIBUTION.md` and `data/manifest.json`.
LunaLens analysis is the project's own output. No institutional endorsement
is implied.

