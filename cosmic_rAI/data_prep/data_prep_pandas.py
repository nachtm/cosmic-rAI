import math
import itertools
from functools import reduce
import pandas as pd


def event_df_from_matrices(matrices, remove_nan=True):
    """Given list of matrices, creates single, multi-indexed DF
    where event-related attributes are columns, are events are rows."""
    event_df = pd.concat(map(_event_df_from_matrix, matrices), ignore_index=True)
    return _remove_nan_events(event_df, matrices) if remove_nan else event_df


def flatten_event_df(df):
    """Given event_df with two levels of columns, condenses to single row."""
    new_df = df.copy()
    lvl0 = df.columns.get_level_values(0).astype('str')
    lvl1 = df.columns.get_level_values(1).astype('str')
    cols = lvl0 + '_' + lvl1
    new_df.columns = cols
    return new_df


def sensor_df_from_matrices(matrices):
    """Given list of matrices, constructs and merges DFs form sensor attributes.
    Returns single DF with sensor ids as rows and attributes as cols."""

    # create sensor df for each matrix, merge them together.
    frames = list(map(_sensor_df_from_matrix, matrices))
    df = reduce(lambda left, right: pd.merge(left, right), frames)

    # make sure all frames have the same indices
    assert all(frames[i].index.equals(frames[i + 1].index)
               for i in range(len(frames) - 1))

    df.index = frames[0].index  # restore index
    return df


def _sensor_df_from_matrix(mat):
    vals = {
        'gain': mat['Gain'][0],
        'pos_x': mat['Position'][0],
        'pos_y': mat['Position'][1], }
    return pd.DataFrame(vals)


def _event_df_from_matrix(mat):
    """Creates multi-index DataFrame from select attributes in matrix."""
    frames = {
        'charges': pd.DataFrame(mat['Charges']),
        'energy': pd.DataFrame(mat['Energy']),
        'composition': pd.DataFrame(mat['Composition']),
        'core_MC': pd.DataFrame(_partition_list(mat['core_MC'], 'x', 'y')),
        'core_reco': pd.DataFrame(_partition_list(mat['core_reco'], 'x', 'y')),
        'dir_MC': pd.DataFrame(_partition_list(mat['dir_MC'], 'azimuth', 'zenith')),
        'dir_reco': pd.DataFrame(_partition_list(mat['dir_reco'], 'azimuth', 'zenith'))}
    return pd.concat(frames, axis=1).fillna(0)


def _remove_nan_events(df, matrices):
    """Returns df with all events containing "nan" in original matrices removed.
    List of matrices should have same length and event order as df."""

    # chain lists of "charges" dicts together into one list (same len as df)
    charges = list(itertools.chain.from_iterable((m['Charges'] for m in matrices)))

    # find evil indices (where nan in matrix)
    evil_indices = []
    for idx, event in enumerate(charges):
        if any(math.isnan(v) for k, v in event.items()):
            evil_indices.append(idx)

    # drop evil indices from DF
    return df.drop(evil_indices)


def _partition_list(lst, x_name='x', y_name='y'):
    """Given list that alternates between x,y values,
    partition into dictionary"""
    return {
        x_name: lst[::2],
        y_name: lst[1::2]}


def get_charges_by_gain(charges_df, sensor_df, gain='Low'):
    """Filter charges_df to particular gain"""
    return charges_df[
        sensor_df.query('gain == "{}"'.format(gain)).T.columns]


def get_log_charges(charges_df):
    """Apply log function to every cell in charges_df, excluding 0s"""
    return charges_df.applymap(
        lambda x: math.log(x) if x != 0 else 0)
