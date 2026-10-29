"""Find BIDS EEG recordings and turn them into stimulus-locked epochs."""

import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# sub-002_ses-01_task-gonogo_run-1_eeg.set -> subject, session, task, run
ENTITIES = re.compile(
    r"sub-(?P<subject>[^_]+)(?:_ses-(?P<session>[^_]+))?_task-(?P<task>[^_]+)(?:_run-(?P<run>[^_]+))?_eeg\.set$"
)


def find_recordings(bids_root: Path, task: str = "gonogo") -> pd.DataFrame:
    """One row per *_eeg.set under bids_root (any folder depth): subject, session, task, run, path, rec_id."""
    rows = []
    for p in sorted(Path(bids_root).rglob(f"*_task-{task}*_eeg.set")):
        m = ENTITIES.search(p.name)
        if m:
            rows.append({**m.groupdict(), "path": p})
    recs = pd.DataFrame(rows, columns=["subject", "session", "task", "run", "path"])
    recs["rec_id"] = recs["path"].map(lambda p: p.name.removesuffix("_eeg.set"))
    return recs


def load_raw(set_file: Path, l_freq: float = 1.0, h_freq: float = 40.0,
             sfreq: float = 500.0, reference: str | None = "average") -> mne.io.BaseRaw:
    """Read a continuous EEGLAB recording, band-pass, re-reference and resample.

    MNE converts EEGLAB's uV to V on reading. Filtering happens on the continuous data,
    before epoching, so epoch edges are not distorted.
    """
    raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
    raw.filter(l_freq, h_freq, verbose=False)
    if reference:
        raw.set_eeg_reference(reference, verbose=False)
    raw.resample(sfreq, verbose=False)
    return raw


def stimulus_metadata(set_file: Path) -> pd.DataFrame:
    """One row per stimulus from the BIDS events.tsv: onset, value, condition, rt, correct.

    condition is 'target' / 'distractor' (from the event value). A target is correct when a
    response was given (go hit); a distractor is correct when none was (correct rejection).
    """
    tsv = Path(str(set_file).replace("_eeg.set", "_events.tsv"))
    ev = pd.read_csv(tsv, sep="\t", na_values="n/a")
    stim = ev[ev["trial_type"] == "stimulus"].reset_index(drop=True)

    meta = pd.DataFrame({
        "onset": stim["onset"],
        "value": stim["value"],
        "condition": np.where(stim["value"].str.contains("target"), "target",
                              np.where(stim["value"].str.contains("distractor"), "distractor", stim["value"])),
        "rt": stim["response_time"],
        "stim_file": stim["stim_file"],
    })
    responded = meta["rt"].notna()
    meta["correct"] = np.where(meta["condition"] == "target", responded, ~responded)
    return meta


def make_epochs(raw: mne.io.BaseRaw, meta: pd.DataFrame, tmin: float = -0.6, tmax: float = 1.0,
                reject_uv: float | None = 100.0, reject_picks: list[str] | None = None) -> tuple[mne.Epochs, int]:
    """Cut stimulus-locked epochs with the stimulus metadata attached.

    Epochs whose peak-to-peak amplitude on reject_picks (default: all channels) exceeds
    reject_uv are dropped. Returns the epochs and the number rejected.
    """
    samples = np.round(meta["onset"].to_numpy() * raw.info["sfreq"]).astype(int) + raw.first_samp
    codes = {c: i + 1 for i, c in enumerate(sorted(meta["condition"].unique()))}
    events = np.column_stack([samples, np.zeros_like(samples), meta["condition"].map(codes).to_numpy()])

    epochs = mne.Epochs(raw, events, event_id=codes, tmin=tmin, tmax=tmax, baseline=None,
                        metadata=meta, preload=True, verbose=False)
    n_before = len(epochs)
    if reject_uv is not None:
        ptp = np.ptp(epochs.get_data(picks=reject_picks or epochs.ch_names), axis=2).max(axis=1)   # per epoch
        epochs = epochs[np.flatnonzero(ptp < reject_uv * 1e-6)]
    return epochs, n_before - len(epochs)
