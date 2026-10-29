"""Time-domain Granger causality between channel pairs, pooled over trials.

For target y and source x, both (trials, times):
    restricted model: y[t] ~ y[t-1..t-p]
    full model:       y[t] ~ y[t-1..t-p] + x[t-1..t-p]
    GC(x -> y) = ln(RSS_restricted / RSS_full)      (0 = x's past adds nothing)

GC is never negative (more predictors never fit worse), so whether x adds value is decided
by a test: the F-test on the extra p coefficients, and a trial-shuffle permutation test.

On scalp EEG every channel is a mixture of the same sources (volume conduction, common
reference), and mixing alone makes most pairs "significant". GC is therefore also computed
on time-reversed data. Time-reversed GC (TRGC, Haufe et al. 2013, NeuroImage):
    net(x -> y)  = GC(x -> y) - GC(y -> x)
    trgc(x -> y) = net(x -> y) - net_reversed(x -> y)
Mixing gives trgc ~ 0; true directed influence x -> y gives trgc > 0.
"""

from itertools import permutations

import mne
import numpy as np
from scipy import stats


def prepare_for_gc(data: np.ndarray) -> np.ndarray:
    """Make epochs closer to stationary: remove the evoked response, z-score each trial and channel.

    data: (trials, chans, times). The evoked response is the same in every trial; left in, it
    would look like one channel 'predicting' another simply because both follow the stimulus.
    """
    x = data - data.mean(axis=0, keepdims=True)                                  # remove ensemble mean
    x = (x - x.mean(axis=2, keepdims=True)) / x.std(axis=2, keepdims=True)       # z-score per trial/channel
    return x


def lagged(series: np.ndarray, order: int) -> np.ndarray:
    """Past values as predictors: (trials, times) -> (trials * (times - order), order).

    Column k-1 holds the value k samples back (t-1, t-2, ..., t-order). Rows from all
    trials are stacked; no row mixes samples from two different trials.
    """
    n_times = series.shape[1]
    cols = [series[:, order - k:n_times - k] for k in range(1, order + 1)]       # each (trials, times - order)
    return np.stack(cols, axis=-1).reshape(-1, order)


def _rss(X: np.ndarray, y: np.ndarray) -> float:
    """Residual sum of squares of the least-squares fit y ~ intercept + X."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    return float(resid @ resid)


def granger_pair(target: np.ndarray, source: np.ndarray, order: int) -> dict:
    """Does the source's past improve prediction of the target beyond the target's own past?

    target, source: (trials, times). Returns gc = ln(RSS_r / RSS_f), the F statistic and its p-value.
    """
    y = target[:, order:].reshape(-1)              # values to predict, all trials stacked
    own = lagged(target, order)                    # target's own past
    other = lagged(source, order)                  # source's past

    rss_r = _rss(own, y)
    rss_f = _rss(np.hstack([own, other]), y)

    n_obs = len(y)
    df_num, df_den = order, n_obs - 2 * order - 1
    F = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
    return {"gc": float(np.log(rss_r / rss_f)), "F": float(F),
            "p_F": float(stats.f.sf(F, df_num, df_den)), "n_obs": n_obs}


def select_order(data: np.ndarray, max_order: int) -> int:
    """Model order (number of lags) with the lowest BIC, averaged over all ordered channel pairs.

    data: (trials, chans, times), already prepared. BIC = n ln(RSS/n) + k ln(n) penalises extra lags.
    """
    n_chans = data.shape[1]
    bics = []
    for order in range(1, max_order + 1):
        pair_bics = []
        for i, j in permutations(range(n_chans), 2):
            y = data[:, i, order:].reshape(-1)
            X = np.hstack([lagged(data[:, i], order), lagged(data[:, j], order)])
            n, k = len(y), 2 * order + 1
            pair_bics.append(n * np.log(_rss(X, y) / n) + k * np.log(n))
        bics.append(np.mean(pair_bics))
    return int(np.argmin(bics)) + 1


def pairwise_granger(epochs: mne.BaseEpochs, order: int | None = None, max_order: int = 15,
                     n_perm: int = 0, rng: np.random.Generator | None = None) -> list[dict]:
    """Granger causality for every ordered channel pair (source -> target) in already-cropped epochs.

    order=None picks the order by BIC (up to max_order). With n_perm > 0, p_perm comes from
    shuffling which trial the source signal comes from: that keeps each signal's own structure
    but breaks any trial-by-trial relation between them.
    gc_reversed is the same GC on time-reversed data. trgc (net flow forward minus net flow
    reversed) > 0 suggests real directed influence source -> target rather than signal mixing.
    """
    data = prepare_for_gc(epochs.get_data())
    data_rev = data[:, :, ::-1].copy()                        # time-reversed copy
    if order is None:
        order = select_order(data, max_order)
    rng = rng or np.random.default_rng(0)

    names = epochs.ch_names
    pairs = list(permutations(range(len(names)), 2))          # (i, j): i = target, j = source
    fwd = {(i, j): granger_pair(data[:, i], data[:, j], order) for i, j in pairs}
    rev = {(i, j): granger_pair(data_rev[:, i], data_rev[:, j], order)["gc"] for i, j in pairs}

    rows = []
    for i, j in pairs:
        res = fwd[(i, j)]
        net = res["gc"] - fwd[(j, i)]["gc"]                   # flow j -> i minus flow i -> j
        net_rev = rev[(i, j)] - rev[(j, i)]
        row = {"source": names[j], "target": names[i], "order": order, **res,
               "gc_reversed": rev[(i, j)], "net_gc": net, "trgc": net - net_rev}
        if n_perm:
            null = [granger_pair(data[:, i], data[rng.permutation(len(data)), j], order)["gc"]
                    for _ in range(n_perm)]
            row["p_perm"] = (1 + np.sum(np.array(null) >= res["gc"])) / (1 + n_perm)
            row["gc_null_mean"] = float(np.mean(null))
        rows.append(row)
    return rows
