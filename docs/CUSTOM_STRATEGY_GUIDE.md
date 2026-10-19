# Designing and defending your own LunaLens strategy

The implemented foundation gives you an executable comparison, an isolated
place to change the decision rule, and tests of the experimental information
boundary. Your contribution should be a reasoned decision that you implement,
test and explain. A more complicated score is not automatically a better
engineering result, and a negative result can still be useful evidence.

This guide describes the code in `src/feature_extraction.py`,
`src/observation_planner.py`, `src/custom_strategy.py`, and `src/simulator.py`.
It does not claim that any method wins. Consult the executed tables in
`evidence/results/summary_by_image.csv` and
`evidence/results/summary_by_kind.csv` for measured outcomes, and keep real and
synthetic imagery separate when interpreting them.

## 1. Follow one baseline decision from pixels to an observation

### Step 1: construct the information that the satellite initially has

The simulator degrades a full-resolution image with blur, downsampling and
resizing. It retains the full-resolution image privately as the source of
future observations and for evaluation. A supplied `initial_map` can replace
degradation for controlled experiments. You are responsible for making that
map a plausible available input; supplying the full-resolution truth as the
initial map would remove the mapping problem.

The initial image is tiled in row-major order. IDs begin at zero. A tile at an
image boundary keeps its actual height and width. The default cost is one unit
per tile. The optional area cost is `tile.area / region_size**2`. These are
experimental accounting rules, not orbital or spacecraft resource models.

### Step 2: calculate observable descriptors

Every tile receives the following seven values, calculated once from the
initial degraded map. Sobel and Gaussian filtering operate on that whole
available map before regions are sliced, so there are no artificial filter
boundaries between tiles.

| Descriptor | Implemented definition | Reason it might be useful | Important failure mode |
| --- | --- | --- | --- |
| `variance` | Mean squared deviation from the tile mean | Captures intensity variation | Broad illumination differences also create variation |
| `edge_density` | Fraction of Sobel magnitudes greater than 0.04 | Counts sharp visible transitions | The threshold depends on pixel scale and preprocessing |
| `gradient` | Mean Sobel magnitude; each directional kernel divided by eight | Measures the strength of visible transitions | Brightness gradients are not terrain slopes |
| `texture` | RMS of `I - Gaussian(I, sigma=1 pixel)` | Captures visible high-frequency structure | Noise or compression can appear textured; degradation can erase structure |
| `entropy` | Shannon entropy of a 32-bin intensity histogram, in bits | Measures diversity of intensity values | Histogram entropy does not encode spatial arrangement |
| `contrast` | 95th minus 5th intensity percentile | Measures a robust intensity range | A simple bright/dark division may have high contrast |
| `information` | `(entropy / 5) * (contrast + edge_density) / 2` | Combines diversity with contrast and boundaries | This is a correlated composite proxy, not measured information gain |

For example, histogram entropy is

\[
H_i = -\sum_{b:p_{ib}>0}p_{ib}\log_2 p_{ib}.
\]

A flat tile occupies one histogram bin and has entropy zero. A tile with equal
mass in all 32 bins has entropy five bits. These statements describe image
statistics, not a geological interpretation of the tile.

### Step 3: place each feature on a comparable scale

Within one initial image, each feature column is min-max normalized:

\[
f_{ij} = \frac{x_{ij}-\min_k x_{kj}}{\max_k x_{kj}-\min_k x_{kj}}.
\]

A constant column becomes zero. The fitted scale remains fixed throughout the
simulation. Different images have different feature ranges, so a normalized
value of 0.8 is not an absolute physical measurement or a calibrated probability.
All operations up to this point use the initial map alone.

### Step 4: calculate an interpretable utility per unit cost

For nonnegative weights, normalized to sum one, the feature baseline computes

\[
u_i=\sum_{j=1}^{7}w_j f_{ij},\qquad s_i=\frac{u_i}{c_i}.
\]

The defaults, in feature order, are `0.10, 0.15, 0.15, 0.20, 0.15, 0.10, 0.15`.
They are preset design choices, not fitted coefficients. Passing a weight
mapping replaces the defaults: omitted features receive zero weight. Unknown
names, negative or nonfinite weights, and all-zero weights are rejected.

Consider a deliberately small numeric example with only variance and entropy
enabled: `{"variance": 3, "entropy": 1}` becomes weights 0.75 and 0.25.

| Region | Normalized variance | Normalized entropy | Utility | Cost | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.20 | 0.80 | 0.350 | 1.00 | 0.350 |
| 1 | 0.80 | 0.20 | 0.650 | 2.00 | 0.325 |
| 2 | 0.40 | 0.10 | 0.325 | 0.25 | 1.300 |

Region 2 is selected when it is affordable and unobserved. With all costs equal
to one, region 1 would be selected. This illustrates why specifying the cost
model is part of the algorithm, not an incidental UI setting.

### Step 5: apply acquisition constraints and update the reconstruction

Observed and unaffordable tiles receive score negative infinity. The largest
finite score wins; ties resolve to the lowest region ID. The simulator checks
the returned ID independently, including repeats and affordability. It then
replaces exactly that tile in the reconstruction with the corresponding
full-resolution pixels. Every unobserved pixel remains equal to its initial
value. No interpolation or super-resolution is added between observations.

Metrics are calculated by the evaluator after each update and saved to result
history. They are never added to planner state. Exact replacement guarantees
nonincreasing MSE and MAE; SSIM can change nonmonotonically because its local
windows couple neighboring pixels.

## 2. Understand the information boundary before modifying the policy

The strategy protocol is deliberately small:

```python
name: str
scores(state: PlannerState) -> numpy.ndarray  # one score per region
choose_next(state: PlannerState, rng: numpy.random.Generator) -> int | None
```

An integer return value is a region ID. `None` stops acquisition. Built-in
policies continue while any affordable unobserved region remains. A supplied
strategy object is supported directly by `run_simulation`; `make_strategy`
provides the four registered names used by the dashboard and benchmark CLI.

| Available field | What you may use it for |
| --- | --- |
| `initial_map` | The original degraded observations and image dimensions |
| `current_map` | Initial pixels outside selected tiles; revealed pixels inside selected tiles |
| `regions` | Frozen IDs, positions, dimensions and positive costs |
| `features` | The normalized descriptors fitted on the initial map |
| `observed_mask` | Which regions have already been acquired; shape `(N,)`, boolean |
| `remaining_budget` | Current affordable capacity under the selected accounting rule |
| `history` | A tuple of selected IDs, in order; no metric dictionaries |

There is no `ground_truth`, error heatmap, unobserved high-resolution descriptor,
per-region true reconstruction error, or evaluation-history field in this
state. The arrays are detached copies backed by immutable bytes. Neither
ordinary assignment nor re-enabling their NumPy writable flag can modify them.
The simulator retains its own arrays and enforces constraints itself.

This is a scientific information boundary for trusted Python code. It is not
a sandbox against a malicious plug-in that deliberately inspects Python stack
frames, reads other files, or imports a hidden dataset. Keep policy modules
free of such side channels. The tests check interface separation, immutability,
and counterfactual decisions when only unrevealed truth changes.

The distinction between **observed local feedback** and **evaluation leakage**
matters for future work. After observing region `k`, you may compare
`current_map[regions[k].slices]` with the same initial-map tile: those observed
pixels are now available. You must not calculate the same discrepancy against
unrevealed truth elsewhere. A learning policy can use the observed discrepancies
as training targets without receiving the evaluator's full-image metrics.

## 3. The working custom example, exactly as implemented

`src/custom_strategy.py` isolates a policy that combines initial feature utility,
descriptor novelty and spatial spread. Let `O_t` be the set of observed regions,
`p_i` the centre of tile `i`, and `D` the image diagonal in pixels. Its extra
terms are

\[
n_i=\min_{k\in O_t}\frac{\lVert f_i-f_k\rVert_2}{\sqrt{7}},
\qquad
d_i=\min_{k\in O_t}\frac{\lVert p_i-p_k\rVert_2}{D}.
\]

Before the first observation, both terms are zero. Afterwards, novelty is high
when a tile's initial descriptor differs from every observed initial descriptor;
dispersion is high when its centre is distant from every observed centre. The
score is

\[
s_i=\frac{0.60u_i+0.25n_i+0.15d_i}{c_i}.
\]

Constructor parameters `utility_weight`, `novelty_weight` and `spatial_weight`
replace the three coefficients; they must be nonnegative, finite, and not all
zero. They are normalized to sum one. The default method recomputes novelty and
dispersion after each acquisition, but deliberately uses the same initial
descriptors throughout. It does not learn from revealed image residuals.

The policy may distribute observations more widely or reduce repeated sampling
of similar-looking tiles. That can also make it worse on a reconstruction-error
objective: the unusual or distant regions may already be accurately represented.
The implementation and documentation should preserve that possibility.

## 4. Three engineering decisions for you to own

### Decision 1: define what a useful observation means

Choose an objective before looking for winning parameters. The current benchmark
measures map reconstruction under a synthetic degradation model. Pixel MSE,
edge preservation and broad spatial coverage are related but different goals.
For each candidate objective, explain which observations should become more
valuable and what independent evaluation would show success.

Also decide whether observation cost should be uniform, proportional to area,
or eventually include movement and storage. A mean tile descriptor divided by
area cost can favor tiny boundary regions. If your objective is reduction of
whole-image MSE, a more directly aligned candidate is

\[
\widehat{\Delta\mathrm{MSE}}_i
=\frac{A_i}{A_{\mathrm{image}}}\widehat{e_i},
\qquad
s_i=\frac{\widehat{\Delta\mathrm{MSE}}_i}{c_i},
\]

where `e_i` estimates the initial tile's mean squared error from permitted
information. Derive your area factors explicitly instead of quietly changing
the cost model when reporting an improvement.

**Deliverable you can defend:** an objective statement, a cost equation, a
reason for each, and a matched-budget comparison that exposes their tradeoff.

### Decision 2: choose and validate a proxy for missing detail

Question whether visible texture predicts what degradation removed. Low-resolution
entropy may be high in a region that is already well reconstructed, while a
flat-looking region may hide a fine pattern. The `information` feature also
reuses entropy, contrast and edges, so treating all seven as independent signals
would be misleading.

Start with a hypothesis, such as: *the high-pass residual on the initial map
predicts missing fine detail better than intensity histogram entropy at a fixed
degradation scale*. Compare one-feature policies or feature drops without tuning
on the evaluation images. Repeat at several downsampling factors to see whether
the relationship survives a changed observation model.

An extension worth investigating is online prediction from acquired regions.
For each observed tile, calculate its now-known initial-map error from permitted
current-map pixels. Use descriptor similarity to estimate error for unobserved
tiles. Specify the prior used before observations, the similarity bandwidth,
and what happens far from all observed examples. Keep this an explicit proposal
until you implement, validate and run it.

**Deliverable you can defend:** a descriptor or predictor you implemented,
an explanation of why it might correlate with the target, and plots showing
both successful predictions and failures on held-out images.

### Decision 3: control the exploration tradeoff

Decide how much budget should sample novel or distant regions instead of regions
with the highest immediate feature utility. A fixed weighted sum is simple and
auditable. A changing schedule or an observed-error predictor may adapt better,
but introduces additional assumptions and parameters.

For example, you can test whether emphasizing spatial spread early and feature
utility later improves low-budget performance. State whether progress means
fraction of tiles observed, pixel coverage, budget spent, or remaining mission
capacity. These definitions differ for partial tiles and nonuniform costs.

**Deliverable you can defend:** a policy equation, an implementation in the custom
module, a predeclared parameter choice, and an ablation that removes the new term
to isolate its effect.

## 5. A small example modification that preserves the protocol

The following is an **illustrative AI-assisted starting point**, not a claim of
student authorship or a measured improvement. It schedules the coefficients
according to observed pixel coverage. Place it in a new `src/my_strategy.py`
while experimenting, leaving the baseline module intact:

```python
from src.custom_strategy import CustomStrategy


class CoverageFirstStrategy(CustomStrategy):
    name = "coverage_first"

    def scores(self, state):
        covered_pixels = sum(
            region.area
            for region, observed in zip(state.regions, state.observed_mask)
            if observed
        )
        coverage = covered_pixels / state.initial_map.size
        scheduled = CustomStrategy(
            weights=self.weights,
            utility_weight=0.4 + 0.4 * coverage,
            novelty_weight=0.2,
            spatial_weight=0.4 * (1.0 - coverage),
        )
        return scheduled.scores(state)
```

Inherited `choose_next` still applies affordability, observation status, and
deterministic tie-breaking. You can exercise an explicit strategy object with
the existing simulator:

```python
from src.image_loader import list_sample_images, load_image
from src.my_strategy import CoverageFirstStrategy
from src.simulator import SimulationConfig, run_simulation

entry = list_sample_images()[0]
image = load_image(entry["path"], max_dimension=256)
result = run_simulation(
    image,
    CoverageFirstStrategy(),
    SimulationConfig(region_size=32, budget=16, seed=0),
)
print(result.selected_ids)
print(result.history[-1])
```

The dashboard and benchmark register four named methods. To benchmark a new
method there, either apply your chosen score change inside the existing
`CustomStrategy` or deliberately extend `make_strategy` and the benchmark's
method list, with tests and clear output names. Merely creating the example
module does not register a fifth dashboard option.

### A necessary distinction for budget-aware strategies

For the four current policies with uniform benchmark costs, a run stopped after
`k` observations follows the same first `k` decisions as a full-map run with the
same seed. The benchmark tests this property and efficiently derives default
budget endpoints from complete trajectories.

A future policy that changes priorities using `state.remaining_budget` can
break this property. Its first observation under a budget of 10 may differ
from its first observation under a budget of 100. In that case, evaluate endpoint
comparisons with actual capped runs:

```bash
python -m experiments.run_experiments --independent-budgets --output-dir evidence_custom
```

That option reruns nontrivial endpoints at their actual caps. The exported
full-scan curves still describe full-scan trajectories; they do not become
independently replanned curves at every observation count. Do not label a prefix
as a valid capped-budget result for a horizon-aware policy without checking it.

## 6. Experiments that can distinguish a useful idea from a favorable example

| Question | Matched experiment | What would weaken the claim |
| --- | --- | --- |
| Does novelty help? | Custom with novelty enabled versus the same policy with its coefficient zero | Gains vanish on held-out images or reverse at the target budget |
| Does spatial spread help? | Enable/disable dispersion while retaining identical images, costs and other settings | Better coverage produces worse reconstruction at the intended operating point |
| Is the proxy robust to scale? | Repeat fixed parameters at downsampling factors 2, 4 and 8 | A gain exists only at the one scale used to choose the weights |
| Does the method handle boundary tiles? | Use nondivisible image shapes and compare uniform versus area costs explicitly | The gain comes entirely from selecting cheap partial tiles |
| Does the custom schedule matter? | Fixed coefficients versus your declared schedule | An equally simple fixed policy performs as well |
| Is apparent success due to randomness? | Repeat random selection on distinct seeds; compare paired image-level outcomes | The advantage is smaller than ordinary seed-to-seed variation |
| Does feedback teach the policy anything? | Compare an observed-error learner with a version that uses only the initial prior | Learning adds computation but no reliable improvement |

Use held-out images if you select features, schedules or coefficients after
seeing the bundled benchmark. Different crops from one source image can share
appearance and are not strong evidence of independent scene generalization.
Repeated deterministic runs are not additional independent samples. Report
per-image outcomes alongside averages, distinguish real from synthetic images,
and retain cases where a baseline wins.

The existing single-feature drops are sensitivity checks for the fixed baseline.
They do not establish feature independence or prove a causal account of
geological value. Compare source hashes and saved configuration when combining
results from different revisions.

## 7. Validation and evidence to keep with your contribution

Run the core checks before and after changing your policy:

```bash
python -m pytest tests/test_features.py tests/test_planner.py tests/test_simulation.py -q
python -m pytest -q
```

The tests cover mathematical feature checks, partial tiling, cost enforcement,
parameter failures, deterministic execution, exact selected-pixel replacement,
and two kinds of counterfactual leakage checks. A spy policy confirms that each
snapshot contains only initial or already-revealed pixels and immutable arrays.
Budget tests include a fractional-cost roundoff case and a zero budget with a
very small positive tile cost.

Add tests for your own substantive behavior, such as whether observed feedback
changes the intended score, whether a term is correctly disabled by an ablation,
and whether altered unrevealed truth leaves decisions unchanged until permitted
feedback differs. Passing an API test does not establish that a policy improves
mapping; that conclusion requires actual experiment results.

Keep a short experiment note with a hypothesis recorded before the run, the
source revision, command, dataset identifiers, parameter values, numeric outputs,
plots, failures, and your interpretation. Export to a separate output directory
so the baseline evidence remains available for comparison.

## 8. An honest contribution record

Use the following as editable placeholders; do not describe unperformed work
as completed:

- **Foundation:** This initial repository, algorithms, tests and documentation
  were implemented with substantial AI assistance and reviewed through executed
  tests and experiments.
- **Decision I made:** `[State the objective, cost model, descriptor, or exploration
  decision you personally chose, and why.]`
- **Code I implemented:** `[List modules/functions you personally wrote or changed,
  with a source revision or diff.]`
- **Validation I performed:** `[Give exact tests, commands and evidence you ran and
  inspected yourself.]`
- **Result I measured:** `[Insert actual held-out or matched-budget numbers,
  including where the method lost.]`
- **What I learned:** `[Explain a failure mode, discarded approach, or changed
  assumption and the evidence behind that conclusion.]`

For an interview, be prepared to reproduce one numerical ranking by hand,
explain why evaluation truth is separate from planner observations, describe
why your chosen objective matches the intended use, and discuss a case where
your custom method loses. Those are concrete demonstrations of engineering
understanding that a decorative dashboard cannot supply by itself.
