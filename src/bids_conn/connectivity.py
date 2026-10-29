"""Channel-pair connectivity and inter-trial phase coherence for stimulus-locked windows."""

from itertools import combinations

import mne
import numpy as np
from mne_connectivity import spectral_connectivity_epochs


def pairs_to_indices(ch_names: list[str], pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert channel-name pairs to the (seeds, targets) index arrays MNE expects."""
    seeds = np.array([ch_names.index(a) for a, _ in pairs])
    targets = np.array([ch_names.index(b) for _, b in pairs])
    return seeds, targets


def crop_window(epochs: mne.BaseEpochs,
    windows: dict[str, tuple[float, float]]):
    cropped = {name: epochs.copy().crop(tmin, tmax) for name, (tmin, tmax) in windows.items()}
    lengths = {name: len(win.times) for name, win in cropped.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Windows differ in length (samples): {lengths}")
    return cropped


def subsample(epochs: mne.BaseEpochs, n: int, rng: np.random.Generator) -> mne.BaseEpochs:
    """Keep n randomly chosen epochs (in original order), e.g. to equalise trial counts."""
    keep = np.sort(rng.choice(len(epochs), size=n, replace=False))
    return epochs[keep]


def window_connectivity(
    epochs: mne.BaseEpochs,
    windows: dict[str, tuple[float, float]],
    pairs: list[tuple[str, str]] | None,
    bands: dict[str, tuple[float, float]],
    con_kwargs: dict) -> list[dict]:
    """Connectivity per window x method x band x channel pair, as long-format rows.

    pairs=None computes every channel pair. Band edges come from `bands`, so con_kwargs
    must not contain fmin/fmax. All windows must have the same number of samples.
    """
    if pairs is None:
        pairs = list(combinations(epochs.ch_names, 2))
    indices = pairs_to_indices(epochs.ch_names, pairs)
    fmin = tuple(lo for lo, _ in bands.values())
    fmax = tuple(hi for _, hi in bands.values())

    rows = []
    for win_name, win in crop_window(epochs, windows).items():
        cons = spectral_connectivity_epochs(win, indices=indices, fmin=fmin, fmax=fmax, faverage=True, **con_kwargs)
        if not isinstance(cons, list):   # a single method returns one object, not a list
            cons = [cons]
        for con in cons:
            values = con.get_data()      # (n_pairs, n_bands)
            for p_idx, (ch1, ch2) in enumerate(pairs):
                for b_idx, band in enumerate(bands):
                    rows.append({"window": win_name, "method": con.method, "band": band,
                                 "ch1": ch1, "ch2": ch2, "value": float(values[p_idx, b_idx])})
    return rows


def window_itc(
    epochs: mne.BaseEpochs,
    windows: dict[str, tuple[float, float]],
    bands: dict[str, tuple[float, float]],
    bandwidth: float = 4.0) -> list[dict]:
    """Inter-trial phase coherence per window x band x channel.

    rayleigh_z = n_trials * itc**2 is ~1 with no phase locking, whatever the trial count,
    so it is the value to compare across conditions and subjects.
    """
    n_trials = len(epochs)
    rows = []
    for win_name, win in crop_window(epochs, windows).items():
        for band, (fmin, fmax) in bands.items():
            spec = win.compute_psd(method='multitaper', fmin=fmin, fmax=fmax,
                                   bandwidth=bandwidth, output='complex', verbose=False)
            X = spec.get_data()                           # (trials, chans, tapers, freqs)
            unit = X / np.abs(X)                          # phase only
            itc = np.abs(np.mean(unit, axis=0))           # average over trials (axis 0) → (chans, tapers, freqs)
            itc = np.mean(itc, axis=1)                    # average over tapers (axis 1) → (chans, freqs)
            z = n_trials * itc**2                         # Rayleigh Z per freq, before averaging
            for ch, ch_itc, ch_z in zip(win.ch_names, np.mean(itc, axis=1), np.mean(z, axis=1)):
                rows.append({"window": win_name, "band": band, "channel": ch,
                             "itc": float(ch_itc), "rayleigh_z": float(ch_z), "n_freqs": len(spec.freqs)})
    return rows
