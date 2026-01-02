# GradingMusic

My experiments grading vocal performance using classical feature engineering, ML baselines, and exploratory quantum-inspired representations.

This repository contains experiments from the QuantumMelody line of work: feature extraction, comparison tasks, and model-based grading.

---

## What problem this answers

Given a vocal recording, produce:
1) a set of interpretable performance features (pitch stability, timing, dynamics, timbre proxies, etc.), and  
2) a defensible scoring / comparison output (e.g., student vs teacher, artist vs artist, or category classification).

---

## What’s in this repo

You will find:
- feature extraction notebooks and scripts (audio → structured features)
- ML baselines for classification / scoring
- quantum-inspired and exploratory quantum analysis notebooks

The notebook names reflect iteration; the “recommended path” below is the stable narrative route.

---

## Recommended run order (current)

1) `dataprocessing.ipynb` — download/prepare audio and metadata
2) `01-extractfeatures.ipynb` / `.py` — compute structured features
3) Quantum analysis notebooks — exploratory representations and comparisons
4) ML notebooks — baseline models and evaluation

(As the repo stabilizes, this will be consolidated into `/notebooks/` and a single `/src/` pipeline.)

---

## Outputs

Typical outputs include:
- feature tables suitable for modeling
- evaluation summaries (cross-validation metrics where applicable)
- comparison plots and per-sample diagnostics

If you are browsing: start with the README + the earliest notebook in the recommended path.


---

## License

MIT

---

## Contact

Blue Leaf Labs — https://www.blueleaflabs.org






