import matplotlib.pyplot as plt
import seaborn as sns
from .data_prep import *


def display_event(event_df, sensor_df, event, set_size_by_charges=False, def_size=10, def_alpha=0.8):
    """

    :param event_df:
    :param sensor_df:
    :param event:
    :param set_size_by_charges:
    :param def_size:
    :param def_alpha:
    :return:
    """

    # DATA PREP
    # merge sensor data wth (transposed) charge data
    merge = get_charges_with_metadata(event_df['charges'], sensor_df)
    data = merge[['gain', 'pos_x', 'pos_y', event]].rename({event: 'charge'}, axis=1)

    hgain_off = data.query('gain == "High" & charge == 0')
    lgain_off = data.query('gain == "Low" & charge == 0')
    hgain_on = data.query('gain == "High" & charge != 0')
    lgain_on = data.query('gain == "Low" & charge != 0')

    hgain_sizes = lgain_sizes = def_size
    if set_size_by_charges:
        # take log of this data and "interpolate" to new range, to avoid negatives
        hgain_sizes = np.interp(np.log(hgain_on['charge']), (-4, 10), (10, 130))
        lgain_sizes = np.interp(np.log(lgain_on['charge']), (-4, 10), (10, 130))

    # construct plot
    fig, ax = plt.subplots(figsize=(10, 7.5))
    core_circle = plt.Circle(event_df.loc[event, 'core_MC'], radius=80, color='y', alpha=0.3)
    ax.scatter(x=hgain_off['pos_x'], y=hgain_off['pos_y'], s=def_size, c='0.7', label='hgain_off', alpha=def_alpha)
    ax.scatter(x=lgain_off['pos_x'], y=lgain_off['pos_y'], s=def_size, c='0.2', label='lgain_off', alpha=def_alpha)
    ax.scatter(x=hgain_on['pos_x'], y=hgain_on['pos_y'], s=hgain_sizes, c='r', label='hgain_on', alpha=def_alpha)
    ax.scatter(x=lgain_on['pos_x'], y=lgain_on['pos_y'], s=lgain_sizes, c='b', label='lgain_on', alpha=def_alpha)
    ax.add_artist(core_circle)
    ax.set_xlim(-800, 800)
    ax.set_ylim(-600, 600)
    ax.legend()
    ax.set_title('Event #{}'.format(event))
    plt.show()


def display_event_sns(charges_df, sensor_df, event):
    c = get_charges_with_metadata(charges_df, sensor_df)
    data = c[['gain', 'pos_x', 'pos_y', event]].rename({event: 'charge'}, axis=1)

    no_charge = data.query('charge==0')
    data.loc[no_charge.query('gain == "High"').index, 'gain'] = 'high_off'
    data.loc[no_charge.query('gain == "Low"').index, 'gain'] = 'low_off'

    sns.lmplot(
        x='pos_x', y='pos_y', data=data, size=10, aspect=1.25,
        palette=dict(high_off='0.7', low_off='0.2', High='r', Low='b'),
        scatter_kws={'alpha': 0.6, 's': 40},
        hue='gain', fit_reg=False)
    plt.show()
