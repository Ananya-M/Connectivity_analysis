# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The SOMA connectivity analysis (sibling project `../soma_data`) applied to open EEG data: OpenNeuro **ds002680** (go/no-go rapid animal categorisation, Delorme et al.). Work happens in two notebooks; reusable code is the `bids_conn` package (`src/bids_conn/`, installed editable). No test suite, linter config, or git repo.

The user is learning Python/MNE. Unless they explicitly ask for code to be written, guide with explanations, pseudocode and targeted fixes rather than writing whole functions, and keep their function names/structure when correcting their code.

## Environment and commands

Own uv environment `conn_open/` and Jupyter kernel `conn_open`, kept separate from SOMA's `conn_anal` so the projects cannot break each other (both kernels are registered user-wide; reusing a kernel name overwrites the other). Run from this directory:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "conn_open"; uv sync          # install / repair (also after moving the folder)
$env:UV_PROJECT_ENVIRONMENT = "conn_open"; uv add <pkg>
.\conn_open\Scripts\python.exe -m ipykernel install --user --name conn_open --display-name conn_open
```

- The user's shell often has a conda env active; call `.\conn_open\Scripts\python.exe -m <module>` explicitly.
- Run a notebook headless without touching the user's copy:
  `.\conn_open\Scripts\python.exe -m nbconvert --to notebook --execute notebooks\01_pre_post_stimulus.ipynb --output-dir <scratch dir> --ExecutePreprocessor.kernel_name=conn_open`
  Setting `MPLBACKEND=Agg` for this suppresses inline figures; leave it unset to capture them.
- Notebooks start with `%autoreload 2`.

## Data

- `config/paths.yaml` → `bids_root` (default `data/ds002680`). `find_recordings` globs `*_task-gonogo*_eeg.set` at any depth and parses `sub/ses/task/run` from filenames, so a full BIDS download (`sub-*/ses-*/eeg/`) works without changes. Currently only `sub-002_ses-01_run-1` is present.
- Each recording: continuous EEGLAB `.set` (MATLAB v5, read directly by `mne.io.read_raw_eeglab`; MNE converts µV→V), 31 channels at 1000 Hz, ~206 s. Channels include non-10-20 names (`T5'`, `P3"`, `CB1`, `I`…), so standard montages don't cover them all; no montage is set.
- Events come from the BIDS `*_events.tsv` (not annotations): `trial_type == 'stimulus'` rows with `value` `animal_target` / `animal_distractor` (50 each per run), `response_time` (ms) only when a response was given. `stimulus_metadata` maps these to `condition` (`target`/`distractor`) and `correct` (target with a response = hit; distractor without = correct rejection). Onsets match the annotations exactly; stimuli are ≥1.8 s apart.
- Other runs of ds002680 may use different `value` strings (the recognition task); `condition` is derived by substring (`target`/`distractor`), and unknown values pass through unchanged. Check `meta.groupby(['value','condition'])` when adding data.

## Code architecture

- `bids_conn/bids.py`:
  - `find_recordings` → DataFrame of recordings with `rec_id`.
  - `load_raw` filters 1–40 Hz, sets the average reference, then resamples to 500 Hz, all on continuous data.
  - `stimulus_metadata` reads events.tsv.
  - `make_epochs` builds `mne.Epochs` (−0.6 to 1.0 s) from tsv onsets with metadata attached, and rejects epochs by peak-to-peak on `reject_picks` only.
- `bids_conn/connectivity.py` is SOMA's module, extended:
  - `window_connectivity(epochs, windows, pairs, bands, con_kwargs)`: `pairs=None` means all pairs; several bands are handled in one call by passing `fmin`/`fmax` tuples, so `con_kwargs` must not contain `fmin`/`fmax`.
  - `window_itc`, `crop_window` (equal-length check) and `subsample` (trial equalisation).
- `notebooks/01_pre_post_stimulus.ipynb` (part 1): homologous pairs F3–F4, C3–C4, P3–P4, O1–O2 plus ITC per hemisphere set. Windows pre/early/late, per condition.
- `notebooks/02_target_vs_distractor.ipynb` (part 2): all 28 pairs of 8 channels, pre vs post per condition. Effect = post − pre; contrast = target effect − distractor effect. Then 3×3 heatmaps per band, and Wilcoxon + FDR per edge (skipped when there are fewer than 6 subjects).
- Both notebooks: config cell → `run_recording(rec)` → loop writing `results/<part>/<rec_id>.csv` (skip if it exists unless `overwrite=True`) → concatenate → average runs within subject before any summary or stats.

## Analysis constraints (keep consistent)

- Average reference: the original "common" reference sits near the vertex (C3/C4 are ~26 µV peak-to-peak vs 50–80 µV elsewhere) and would create spurious C3–C4 coupling.
- Rejection uses the analysis channels only, at 100 µV. FP1/FP2 carry blinks (~190 µV median), and rejecting on all channels would drop over half the epochs.
- Windows at 500 Hz: pre `(-0.5, -0.01)`, early/post `(0.01, 0.5)`, late `(0.5, 0.99)`, 246 samples each. Bins are ~2 Hz apart and the 5-cycle minimum is ~10 Hz, so alpha (3 bins) is usable here, unlike SOMA. MNE's warning for fmin=8 Hz (3.9 cycles) is expected.
- `mt_bandwidth=4` (±2 Hz, 1 taper). The default for 0.49 s smooths ±8 Hz; 6 gives 2 tapers at ±3 Hz.
- Trials are equalised across target/distractor within each recording (fixed seed). Prefer `wpli2_debiased` as the primary measure; PLV reflects volume conduction.
- ITC: compare `rayleigh_z` (≈1 under no phase locking), never after `subtract_evoked()`.
- Go (target) trials contain a button release at ~250–480 ms, inside the post/early window. Target-vs-distractor differences there can be motor rather than perceptual.
