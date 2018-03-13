from .data_prep import *
from .data_viz import display_event


class DataHelper:
    """Wrapper class to make it easier to interact with other functions."""

    def __init__(self, matrices, remove_nan=True):
        self.matrices = matrices
        self.event_df = event_df_from_matrices(matrices, remove_nan=remove_nan)
        self.sensor_df = sensor_df_from_matrices(matrices)

    def __repr__(self):
        return "DataHelper :: event_df: {}, sensor_df: {}".format(
            self.event_df.shape, self.sensor_df.shape)

    def charges_with_metadata(self, use_log_charges=False):
        return get_charges_with_metadata(
            self.event_df['charges'], self.sensor_df, use_log_charges=use_log_charges)

    def display_event(self, event, **kwargs):
        display_event(self.event_df, self.sensor_df, event, **kwargs)

    @property
    def low_gain_charges(self):
        return get_charges_by_gain(self.event_df['charges'], self.sensor_df, gain='Low')

    @property
    def high_gain_charges(self):
        return get_charges_by_gain(self.event_df['charges'], self.sensor_df, gain='High')

    @property
    def log_charges(self):
        return get_log_charges(self.event_df['charges'])
