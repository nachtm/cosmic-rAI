import math
import itertools
import numpy as np
import pandas as pd


class DataHelper:
    """Wrapper class to make it easier to interact with other functions."""

    def __init__(self, matrices, remove_nan=True):
        self.matrices = matrices
        self.event_df = event_df_from_matrices(matrices, remove_nan=remove_nan)
        self.sensor_df = sensor_df_from_matrices(matrices)

    def __repr__(self):
        return "DataHelper :: event_df: {}, sensor_df: {}".format(
            self.event_df.shape, self.sensor_df.shape)

    @property
    def low_gain_charges(self):
        return get_charges_by_gain(self.event_df['charges'], self.sensor_df, gain='Low')

    @property
    def high_gain_charges(self):
        return get_charges_by_gain(self.event_df['charges'], self.sensor_df, gain='High')

    @property
    def log_charges(self):
        return get_log_charges(self.event_df['charges'])

    def charges_with_metadata(self, use_log_charges=False):
        return get_charges_with_metadata(
            self.event_df['charges'], self.sensor_df, use_log_charges=use_log_charges)


# ----------------------------------------------------
# Loading raw data
def load_data(path):
    """Create python dict from .npy data file"""
    return np.load(path).item()


# ----------------------------------------------------
# Working with event data
def event_df_from_matrices(matrices, remove_nan=True):
    """Given list of matrices, creates single, multi-indexed DF
    where event-related attributes are columns, are events are rows."""
    event_df = pd.concat((event_df_from_matrix(m) for m in matrices), ignore_index=True)
    return remove_nan_events(event_df, matrices) if remove_nan else event_df


def event_df_from_matrix(mat):
    """Creates multi-index DataFrame from event-related attributes in matrix."""
    frames = {
        'charges': pd.DataFrame(mat['Charges']),
        'energy': pd.DataFrame(mat['Energy']),
        'file_info': pd.DataFrame(mat['File_info']),
        'fit_status': pd.DataFrame(mat['Fit_status']),
        'composition': pd.DataFrame(mat['Composition']),
        'core_MC': pd.DataFrame(partition_list(mat['core_MC'], 'x', 'y')),
        'core_reco': pd.DataFrame(partition_list(mat['core_reco'], 'x', 'y')),
        'dir_MC': pd.DataFrame(partition_list(mat['dir_MC'], 'azimuth', 'zenith')),
        'dir_reco': pd.DataFrame(partition_list(mat['dir_reco'], 'azimuth', 'zenith'))}
    return pd.concat(frames, axis=1).fillna(0)


def partition_list(lst, x_name='x', y_name='y'):
    """Given list that alternates between x,y values, partition into dictionary"""
    return {
        x_name: lst[::2],
        y_name: lst[1::2]}


def remove_nan_events(df, matrices):
    """Returns event df with events where charge = NaN in original matrices removed.
    List of matrices should have same length and event order as df."""

    # chain list of "charges" dicts together into one list (same len as df)
    charges = list(itertools.chain.from_iterable((m['Charges'] for m in matrices)))

    # find evil indices (where charge is nan in matrix)
    evil_indices = []
    for idx, event in enumerate(charges):
        if any(math.isnan(v) for k, v in event.items()):
            evil_indices.append(idx)

    # drop evil indices from DF
    return df.drop(evil_indices)


def flatten_event_df(df):
    """Given event_df with two levels of columns, condenses to single row."""
    new_df = df.copy()
    lvl0 = df.columns.get_level_values(0).astype('str')
    lvl1 = df.columns.get_level_values(1).astype('str')
    cols = lvl0 + '_' + lvl1
    new_df.columns = cols
    return new_df


# ----------------------------------------------------
# Working with sensor data
def sensor_df_from_matrices(matrices):
    """Given list of matrices, constructs and merges DFs form sensor attributes.
    Returns single DF with sensor ids as rows and attributes as cols."""

    # Make sure all matrices describe sensors in the same way
    # (Assert will throw an error if not)
    for i in range(len(matrices) - 1):
        assert matrices[i]['Position'] == matrices[i + 1]['Position']
        assert matrices[i]['Gain'] == matrices[i + 1]['Gain']

    # Simply construct sensor df from first matrix in list
    return sensor_df_from_matrix(matrices[0])


def sensor_df_from_matrix(mat):
    attributes = {
        'gain': mat['Gain'][0],
        'pos_x': mat['Position'][0],
        'pos_y': mat['Position'][1], }
    return pd.DataFrame(attributes)


# ----------------------------------------------------
# Working with charge data
def get_charges_by_gain(charges_df, sensor_df, gain='Low'):
    """Filter charges_df to particular gain"""
    return charges_df[
        sensor_df.query('gain == "{}"'.format(gain)).T.columns]


def get_log_charges(charges_df):
    """Apply log function to every cell in charges_df, excluding 0s"""
    return np.log(charges_df.mask(charges_df <= 0)).fillna(0)


def get_charges_with_metadata(charges_df, sensor_df, use_log_charges=False):
    if use_log_charges:
        charges_df = get_log_charges(charges_df)
    return sensor_df.merge(
        charges_df.T, how='outer',
        left_index=True, right_index=True)
