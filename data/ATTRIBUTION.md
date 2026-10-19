# Dataset attribution and processing

The authentic image sources are NASA Lunar Reconnaissance Orbiter publications
using the Lunar Reconnaissance Orbiter Camera Narrow Angle Camera (LROC NAC).
All three source pages credit **NASA/GSFC/Arizona State University**.

| Local source | Publication | Source page |
| --- | --- | --- |
| `source_images/PIA14421.jpg` | Bouldery Crater near Mare Australe, 13 April 2011; frame M150062296R | https://science.nasa.gov/photojournal/bouldery-crater-near-mare-australe/ |
| `source_images/PIA12932.jpg` | Impact Melt Flows on Giordano Bruno, 15 December 2009 | https://science.nasa.gov/photojournal/impact-melt-flows-on-giordano-bruno/ |
| `source_images/PIA12941.jpg` | The Floor of Tycho, 14 January 2010 | https://science.nasa.gov/photojournal/the-floor-of-tycho/ |

The exact assets.science.nasa.gov download URLs are retained in
`source_catalog.json` and `manifest.json`. Downloads were made on 19 October
2026; the manifest has the actual UTC timestamps and byte hashes. Source files
are 1000×1000 published JPEG renderings. Each real benchmark image is the
central 512×512 crop, at x=244 and y=244, encoded as a grayscale PNG. This crop
rule and the three source choices were fixed before algorithm comparison.

Pixels are normalized by unsigned dtype range, using 255 for the derived PNGs.
RGB input, when present, is converted with luminance coefficients
0.2126/0.7152/0.0722. No per-image histogram equalization or reference-dependent
contrast stretch is used. The images retain the limitations of the published
JPEGs, including compression, display processing, and uncertain radiometric
calibration. They have no supplied georeferencing in this repository. A pixel
in a source publication is not automatically a metrically calibrated lunar
surface area in LunaLens.

The software's plots, ranks, and measured results are **LunaLens outputs**.
The source image credit does not imply that NASA or Arizona State University
reviewed, validated, or endorsed the algorithms or their conclusions. NASA's
[images and media usage guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/)
provide the source-use context; retain credits and inspect source-specific
notices when substituting or redistributing imagery. Source imagery is separate
from any software license you later choose for your own project.

## Synthetic fixtures

| Sample | Seed | Purpose |
| --- | --- | --- |
| `synthetic_mixed.png` | 11 | Multiscale texture and procedurally drawn crater-like intensity patterns. |
| `synthetic_rugged.png` | 29 | Greater spatial texture variation and more crater-like patterns. |
| `synthetic_hidden_detail.png` | 47 | Deliberate stress case: high-frequency checker detail on the right is largely suppressed by the initial sensor surrogate. |

These are deterministic software fixtures produced by
`src/synthetic.py:generate_terrain`. They are not NASA images, actual lunar
elevation models, calibrated radiometry, or confirmed geological structures.
The hidden-detail fixture is intentionally artificial; its purpose is to expose
the information limitation of planning from a coarse image.

Run `python -m experiments.prepare_data` from repository root to regenerate
samples offline from the included original JPEGs and these seeds. Re-downloading
with `--download` may return changed publication bytes in the future; compare
the manifest hashes before claiming exact reproduction.

