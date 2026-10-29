# Physically constrained adversarial attacks on time-series foundation models

Code for the paper **"Physically Constrained Adversarial Attacks on Time-Series Foundation Models for Day-Ahead Solar and Load Forecasting"** (S. Sarp, Gebze Technical University; manuscript under review).

The study asks whether attacks that pass a physics-based bad-data check can still corrupt day-ahead forecasts from zero-shot time-series foundation models (TSFMs). It uses the public ARPA-E PERFORM data for ERCOT: 22 existing PV plants (651 MW) and nine load series (8 weather zones + ERCOT total), with 2017 for training/calibration and 2018 as the test year.

**Models**
- Chronos-Bolt (small), Chronos-2 and TimesFM 2.5, used zero-shot. TimesFM decoding is re-implemented in PyTorch so that gradients reach the input.
- Baselines: a quantile LSTM trained on 2017 (3 seeds), persistence, and the PERFORM day-ahead P50.

**Attacks**
- Sign-gradient PGD on the last 24 h of measured history: 20 steps, step 0.125ε, random start.
- Budgets ε ∈ {0.02, 0.05, 0.10} of capacity (PV) or of the 2017 peak (load).
- **UC** attacks are limited only by the ε-ball.
- **PC** attacks must also pass a physics-based bad-data check:
  - a seasonal level band;
  - a ramp limit;
  - night-time zero for PV.
  - Each rule is calibrated to ≤ 0.5 % clean false alarms on validation days.
- Two goals:
  - *dev*: move all quantiles away from the clean forecast;
  - *risk*: raise the PV forecast and lower the load forecast, so net load looks smaller.
- Extensions:
  - random-noise baseline;
  - transfer between models and to a median ensemble;
  - covariate attack on Chronos-2 with the day-ahead forecast as covariate;
  - PGD-step and attack-window ablations.

**Detection and defenses**
- the physics check;
- a forecast-residual detector (Chronos-Bolt backcast);
- input sanitization (projection onto the plausible set);
- split-conformal (CQR) recalibration of the 80 % interval.

**Statistics:** 95 % confidence intervals from a day-cluster bootstrap (2,000 resamples, paired attacked vs. clean).

## Repository layout

| File / folder | Purpose |
|---|---|
| `run_all.sh` | One-command pipeline: environment, weights, data, all stages for solar and load, figures, and paper tables |
| `common.py` | Experiment configuration (`ExpConfig`), metrics, shared paths |
| `data.py` | PERFORM download (public S3), hourly panels, physics envelope, ramp limits, windows, day-ahead covariate |
| `models.py` | Differentiable wrappers for Chronos-Bolt, Chronos-2 (± covariate), TimesFM 2.5; quantile LSTM; persistence |
| `attacks.py` | Projection onto UC/PC sets, PGD, random noise, physics-violation check |
| `run_experiments.py` | Stages: `prep`, `verify`, `lstm`, `clean`, `attack`, `transfer`, `cov`, `detect`, `defense`, `steps`, `window` |
| `make_figures.py` | Quick per-domain diagnostic figures |
| `analyze.py` | Final paper tables (T1–T5, A1–A2) and figures (Figs. 2–6) with bootstrap CIs |
| `configs/run3/` | Exact configuration and physics-check calibration of the reported run, and the wrapper-vs-official-API check |
| `results_summary/` | The paper's tables as CSV and `key_numbers.json` |

## Requirements

- Python 3.11 and PyTorch ≥ 2.4. Any of these back ends works: Apple-Silicon GPU (MPS), CUDA, or CPU.
- Packages are listed in `requirements.txt`.
- `chronos-forecasting==2.3.2` and `timesfm==3.0.2` must be installed **without dependencies**, to avoid version conflicts:

```bash
pip install -r requirements.txt
pip install --no-deps chronos-forecasting==2.3.2 timesfm==3.0.2
```

## Quick start

On a Mac with conda (miniforge or anaconda), run:

```bash
bash run_all.sh
```

`run_all.sh` performs these steps:

1. Creates a conda environment `tsfm` and installs the packages.
2. Downloads the model weights from Hugging Face: `amazon/chronos-bolt-small`, `amazon/chronos-2`, `google/timesfm-2.5-200m-pytorch`.
3. Downloads the ERCOT PERFORM data (about 3 GB).
4. Checks the differentiable wrappers against the official inference APIs.
5. Runs every stage, then writes `results/`, `figures/` and `paper_results/`.

Finished stages are skipped on re-run. The full pipeline took about 2 h 45 min on an Apple M1 Max.

You can also run the stages by hand, for `D` = `solar` and then `load`:

```bash
python data.py --download --domain D
python run_experiments.py prep --domain D
python run_experiments.py verify --domain D
python run_experiments.py lstm --domain D
python run_experiments.py clean --domain D
python run_experiments.py attack --domain D --model chronos-bolt-small   # also: lstm, chronos-2, timesfm-2.5
python run_experiments.py transfer --domain D
python run_experiments.py cov --domain D
python run_experiments.py detect --domain D
python run_experiments.py defense --domain D
python run_experiments.py steps --domain D
python run_experiments.py window --domain D
python analyze.py . paper_results
```

- The device is chosen automatically (MPS > CUDA > CPU). To override it, set `TSFM_DEVICE=cpu|mps|cuda`.
- `--synthetic` runs the code on synthetic data. It is for testing only and was never used for reported results.

## Data and licenses

- **Data:** ARPA-E PERFORM data sets, ERCOT (B. Sergi et al., Open Energy Data Initiative, NREL, 2022, doi:[10.25984/1891136](https://doi.org/10.25984/1891136)), CC BY 4.0.
  - The data are downloaded by `data.py` and are not redistributed here.
  - PV actuals in PERFORM are simulated from NSRDB irradiance with SAM; load actuals are ERCOT data.
- **Model weights:** obtained from Hugging Face under their respective licenses.
- **Code:** MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the paper (citation will be added on publication) or the repository (see `CITATION.cff`).

## Use of AI tools

The analysis code in this repository was developed with the assistance of Claude (Anthropic). The author ran all experiments, checked the results, and is responsible for the content.
