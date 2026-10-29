"""Data preparation and feature construction. Every statistic is fitted on training rows only."""
import numpy as np

from brfss_codebook import SENTINEL_CODES_BY_COLUMN

# Survey administration variables: record identifiers, sampling-design weights, interview
# dates and the telephone screening questions. None of them describe the respondent, and
# as standardized magnitudes they are noise. _STATE is here because it is a FIPS code, so
# its ordering is meaningless; it would need one-hot encoding to be usable.
METADATA_COLUMNS = (
    "SEQNO", "_PSU", "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_LLCPWT", "_DUALUSE",
    "IDATE", "IMONTH", "IDAY", "IYEAR", "FMONTH", "DISPCODE",
    "CTELENUM", "CTELNUM1", "CELLFON3", "CELLFON2", "PVTRESD1", "PVTRESD2",
    "COLGHOUS", "CCLGHOUS", "LADULT", "STATERES", "LANDLINE",
    "QSTVER", "QSTLANG", "_STATE",
)


def build_sentinel_plan(feature_names):
    """Return {column index: sentinel codes} from the BRFSS codebook table.

    The codebook marks "don't know" and "refused" with a code just above each
    variable's valid range, and the code differs per variable: 7/9 for GENHLTH,
    77/99 for the 0-30 day counts, 7777/9999 for WEIGHT2, 3 for _AGE65YR. Left in
    place they reach the model as magnitudes - a GENHLTH refusal standardizes to
    +5.8, where the fitted log-odds slope extrapolates to a predicted risk of 88%
    against an observed 14%.
    """
    return {c: np.array(SENTINEL_CODES_BY_COLUMN[name], dtype=np.float64)
            for c, name in enumerate(feature_names)
            if name in SENTINEL_CODES_BY_COLUMN}


def sentinel_mask(x, plan):
    """Boolean mask of the cells holding one of a column's sentinel codes."""
    mask = np.zeros(x.shape, dtype=bool)
    for c, codes in plan.items():
        mask[:, c] = np.isin(x[:, c], codes)
    return mask


def response_counts(x, mask, keep_idx):
    """Per row: how many kept answers were refused, and how many were never asked.

    Recoding a sentinel to missing deliberately erases it from its own column, so
    these two columns are what is left of the distinction between a respondent who
    declined to answer and a question that was never put to them.
    """
    return np.column_stack([mask[:, keep_idx].sum(axis=1),
                            np.isnan(x[:, keep_idx]).sum(axis=1)]).astype(np.float64)


def fit_preprocessor(x_train, feature_names=None, recode_sentinels=True,
                     drop_metadata=True, add_counts=True):
    """Return preprocessing state, including learned scaling and feature selection.

    Sentinel codes are recoded as missing first, since that changes which columns
    pass the missingness filter. A column is then dropped when it is more than half
    missing, when it carries no variance, or when feature_names identifies it as
    survey metadata. The mean and standard deviation are taken over the observed
    values only, so that transform_features can leave missing entries at zero.

    Passing feature_names is what enables the codebook recoding and the metadata
    filter; without the column names neither is possible. The three switches exist
    for the ablation study and are all on in the submitted model.
    """
    x = np.array(x_train, dtype=np.float64, copy=True)
    use_names = feature_names is not None
    plan = build_sentinel_plan(feature_names) if use_names and recode_sentinels else {}
    mask = sentinel_mask(x, plan)
    x[mask] = np.nan
    keep = np.isnan(x).mean(axis=0) <= 0.5
    if use_names and drop_metadata:
        keep &= ~np.isin(feature_names, METADATA_COLUMNS)
    keep_idx = np.flatnonzero(keep)
    mean, std = np.nanmean(x[:, keep_idx], axis=0), np.nanstd(x[:, keep_idx], axis=0)
    nonconst = std > 0
    keep_idx = keep_idx[nonconst]
    counts = response_counts(np.asarray(x_train, dtype=np.float64), mask, keep_idx)
    count_std = counts.std(axis=0)
    return {
        "keep_idx": keep_idx,
        "sentinels": plan,
        "mean": mean[nonconst],
        "std": std[nonconst],
        "count_mean": counts.mean(axis=0),
        "count_std": np.where(count_std > 0, count_std, 1.0),
        "add_counts": add_counts,
    }


def transform_features(x, state):
    """Apply the fitted preprocessing state and return the transformed features.

    Missing entries end up at exactly zero, i.e. at the training mean of their own
    column, so they contribute nothing to the linear score. Imputing a raw zero
    before standardizing would instead place them outside the valid range of most
    of these variables.
    """
    x = np.array(x, dtype=np.float64)
    mask = sentinel_mask(x, state["sentinels"])
    counts = response_counts(x, mask, state["keep_idx"])
    x[mask] = np.nan
    z = (x[:, state["keep_idx"]] - state["mean"]) / state["std"]
    z[np.isnan(z)] = 0.0
    if not state["add_counts"]:
        return add_intercept(z)
    counts = (counts - state["count_mean"]) / state["count_std"]
    return np.concatenate([add_intercept(z), counts], axis=1)


def add_intercept(x):
    """Return the design matrix with a column of ones."""
    ones = np.ones(shape=(x.shape[0], 1))
    return np.concatenate((ones, x), axis=1)


def encode_labels(y):
    """Convert dataset labels {-1, 1} to logistic labels {0, 1}."""
    return np.where(y == 1, 1, 0)
