# LunaLens — executed validation record

Validation completed on **19 October 2026**. The delivered project was executed
on Linux x86-64 with **Python 3.12.14**, using the exact dependency versions in
`requirements-lock.txt`. The checks below concern the code and bundled
experiment. They do not establish flight readiness or performance on an
unseen population of lunar scenes.

## 1. Results at a glance

| Check | Actual outcome | Evidence |
| --- | --- | --- |
| Full automated suite in a separate project copy and virtual environment | **190 passed; 0 failures, errors, or skips**, in 23.391 seconds of pytest suite time | [pytest log](../evidence/results/pytest.log), [JUnit XML](../evidence/results/pytest_results.xml), [execution record](../evidence/results/test_execution.json) |
| Installed dependency versions | All **53 locked packages** match the lock file | [clean environment](../evidence/results/clean_environment.json) |
| Offline sample regeneration in the separate copy | All **six PNGs** regenerated from cached originals or procedural seeds with identical SHA-256 hashes | [CLI check record](../evidence/results/clean_cli_smoke.json), [CLI log](../evidence/results/clean_cli_smoke.log) |
| Small command-line benchmark and comparison | Data preparation, **5 baseline + 7 ablation runs**, and the comparison command all exited successfully | [CLI log](../evidence/results/clean_cli_smoke.log) |
| Full evidence benchmark | **48 full-scan runs + 42 feature-drop ablation runs**, with **12,336 baseline history rows** | [summary](../evidence/results/summary.json), [execution log](../evidence/results/execution.log) |
| Repeated numerical execution | Seven principal numerical CSVs were byte-identical across **three complete benchmark executions** | [reproduction check](../evidence/results/reproduction_check.json) |
| Recorded runtime provenance | All **19 source and dependency hashes** match the delivered files | [provenance](../evidence/results/reproducibility.json), [hash verification](../evidence/results/reproduction_check.json) |
| Dashboard integration | Eight Streamlit AppTest tests passed as part of the 190-test suite | [dashboard tests](../tests/test_dashboard.py), [JUnit XML](../evidence/results/pytest_results.xml) |
| Actual local server startup | Health endpoint returned **HTTP 200 / `ok`**; homepage returned **HTTP 200** with Streamlit HTML | [startup record](../evidence/results/streamlit_startup_check.json), [server log](../evidence/results/streamlit_startup.log) |

## 2. What the automated tests exercise

The suite covers image loading, fixed-range normalization, RGB and 16-bit TIFF
handling, preprocessing, synthetic reproducibility, exact region coverage and
partial boundary tiles, numerical feature definitions, score ordering, weight
validation, affordable choices, seeded determinism, and metric calculations.

Simulation tests check zero and full budgets, selected-pixel replacement,
unchanged unobserved pixels, duplicate and invalid selections, cumulative cost,
fractional-cost roundoff, and exact reconstruction when every tile is revealed.
Aggregation checks distinguish independent random seeds from repeated
deterministic runs. Budget-prefix tests verify that all four shipped policies
have the prefix behavior assumed by the default uniform-cost benchmark.

Leakage tests check the planner interface, detached immutable arrays, and
counterfactual hidden images: changing unrevealed reference pixels must not
change a decision while the permitted observation history is the same. A spy
strategy checks that snapshots contain the initial pixels outside acquired
regions and revealed pixels only inside them. The evaluator's full-image errors
are not part of planner state. These checks protect the experimental interface;
they are not a security sandbox for hostile Python plugins.

Dashboard tests exercise real Streamlit widget interactions, uploaded image
replacement, simulation, repeated-seed comparison, result invalidation, export
contents, and the source-change restart guard. They execute application logic
through Streamlit's AppTest framework. They do not inspect a desktop browser's
rendering or operate native file-download dialogs.

## 3. Clean installation and offline operation

The project, tests, and bundled input data were copied into a separate directory.
A new virtual environment was created without inheriting the project's
development environment. After dependency installation, the delivered lock was
checked against the installed versions and the full suite ran in that copy.
The absolute execution paths and exact commands are retained in the JSON logs.

This check caught an intermediate packaging defect: stale plotting-library
metadata had recorded `contourpy==1.4.0` even though the experiments imported
version 1.3.3. The stale metadata was removed, the lock corrected to 1.3.3, and
the clean environment verified against it. Both imported and distribution
versions now agree. The final full benchmark was rerun against the corrected
lock; its seven checked numerical outputs were unchanged.

All locked packages declare a Python requirement accepting version 3.11.0.
**Python 3.11 itself was not executed here**; the tested interpreter was 3.12.14.
Windows and macOS were not executed. The metadata check supports the declared
minimum but does not replace testing each platform and interpreter.

The first installation needs a package index or an existing wheel cache. Once
installed, the sample dashboard, tests, data regeneration, and experiments use
the bundled files and do not need internet access, a GPU, or an API key. The
optional `prepare_data --download` operation deliberately fetches NASA sources.

## 4. Reproduce the delivered evidence

From the extracted project directory, with its virtual environment activated:

```bash
python -m pip install -r requirements-lock.txt
python -m pytest -q
python -m experiments.run_experiments
python -m experiments.diagnose_scores
python -m experiments.compare_algorithms
```

The benchmark uses six 512×512 images, 32×32 regions, Gaussian sigma 1,
downsampling factor 4, uniform observation costs, five random seeds (0–4), and
fixed documented weights. Each of the 48 complete trajectories has 257 states,
including the initial state. The 42 ablations remove one of seven weights on
each of six images at 25% budget and renormalize the remaining weights. The
last full execution took **299.408 seconds** on the execution CPU. Other
machines can take a different amount of time.

Three complete executions verified repeatability. They are **not three new
independent datasets**, and the repeated runs are not added to statistical
sample counts. The verification record includes reference hashes for the
seven compared CSVs. It does not claim every binary PNG or timestamp-bearing
metadata file is byte-identical across machines.

The 19 provenance hashes cover the numerical source modules, experiment
commands, and dependency declarations used by the benchmark. They do not cover
every documentation or dashboard file: the benchmark does not execute
`app.py`. Dashboard exports separately record their app and numerical-source
hashes. New source changes require new validation and experiment provenance.

The score diagnostics, evaluation-only upper bounds, and observations-to-target
figures were regenerated from the final completed benchmark. The true-error
bound is retrospective evaluation; it is never supplied to the planner.

For the small end-to-end check used in the separate installation:

```bash
python -m experiments.prepare_data
python -m experiments.run_experiments --limit 1 --max-dimension 128 --seeds 0 1 --dpi 100 --output-dir evidence/smoke
python -m experiments.compare_algorithms --results evidence/smoke/results
```

This smoke benchmark has 16 regions per image. Its 5 baseline and 7 ablation
runs are installation checks; they are not mixed with the main scientific
results. The six regenerated sample hashes are recorded separately.

For policies that depend on the remaining budget, use the actual capped-budget
evaluation mode described in the [custom strategy guide](CUSTOM_STRATEGY_GUIDE.md):

```bash
python -m experiments.run_experiments --independent-budgets --output-dir evidence/local_runs/independent
```

## 5. Dashboard startup and remaining limits

The actual Streamlit server was started on an available localhost port in the
clean installation. Its health endpoint and homepage were requested over HTTP;
both passed. The check then stopped the server. No public service was deployed,
and no background server is required to use the delivered files.

The PNG evidence was rendered from actual data and inspected for legibility.
No desktop browser visual inspection or native download-dialog test was
performed. The application has automated integration coverage and a verified
server startup; visual rendering across browsers remains a manual acceptance
check for the user's laptop.

The benchmark demonstrates both success and failure on a deliberately small
dataset. Passing software tests does not validate entropy as scientific value,
establish terrain hazards, or make the three real examples representative of
the Moon. The [technical report](TECHNICAL_REPORT.md) gives the measured outcomes,
the hidden-detail failure, aggregation choices, and simulation assumptions.
