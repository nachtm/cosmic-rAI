import numpy as np
import math


def cleanup(matrices, keys=('dir_MC', 'core_MC', 'dir_reco', 'core_reco')):
    """Some fields are twice as long as other fields.
    Want to combine every two sequential values into one."""
    for matrix in matrices:
        for key in keys:
            matrix[key] = _cleanup_helper(matrix[key])


def _cleanup_helper(li):
    new_list = []
    for i in range(len(li) // 2):
        new_list.append((li[2 * i], li[2 * i + 1]))
    return new_list


def gain_differentiate(mat):
    """Find all the sensors that are high/low gain"""
    hgain = list(filter(lambda x: mat['Gain'][0][x] == 'High', mat['Position'][0].keys()))
    lgain = list(filter(lambda x: mat['Gain'][0][x] == 'Low', mat['Position'][0].keys()))
    all_sensors = list(mat['Gain'][0].keys())
    return hgain, lgain, all_sensors


def get_index_dict(sensors, direction_data=False):
    """build a name->index dict"""
    name_index_dict = {}
    for i in range(len(sensors)):
        name_index_dict[sensors[i]] = i
    if direction_data:
        name_index_dict['zenith'] = i + 1
        name_index_dict['azimuth'] = i + 2
    return name_index_dict


def gen_matrix(proton_data, iron_data, feature_dict, direction=False):
    """Build the matrix by setting up an empty one and populating it
    with the values according to the indices created with get_index_dict."""

    event_matrix = np.zeros((len(proton_data['Composition']) + len(iron_data['Composition']), len(feature_dict) + 1))
    for i in range(len(proton_data['Charges'])):
        # insert sensor data
        event = proton_data['Charges'][i]
        for sensor in proton_data['Charges'][i].keys():
            try:
                event_matrix[i, feature_dict[sensor]] = 0 if math.isnan(event[sensor]) else event[sensor]
            except KeyError:
                continue
        # insert direction data
        if direction:
            event_matrix[i, feature_dict['zenith']] = proton_data['dir_MC'][i][0]
            event_matrix[i, feature_dict['azimuth']] = proton_data['dir_MC'][i][1]
        event_matrix[i, -1] = 1

    for i in range(len(iron_data['Charges'])):
        # insert sensor data
        event = iron_data['Charges'][i]
        for sensor in iron_data['Charges'][i].keys():
            try:
                event_matrix[len(proton_data['Charges']) + i, feature_dict[sensor]] = 0 if math.isnan(
                    event[sensor]) else event[sensor]
            except KeyError:
                continue

        # insert direction data
        if direction:
            # print(iron_data['dir_MC'][i])
            event_matrix[len(proton_data['Charges']) + i, feature_dict['zenith']] = iron_data['dir_MC'][i][0]
            event_matrix[len(proton_data['Charges']) + i, feature_dict['azimuth']] = iron_data['dir_MC'][i][1]
        event_matrix[len(proton_data['Charges']) + i, -1] = 0

    return event_matrix


def split_train(events_matrix, cutoff=0.9):
    """Split event matrix into testing and training sets"""
    np.random.shuffle(events_matrix)
    train_size = int(events_matrix.shape[0] * cutoff)
    trainset = events_matrix[:train_size]
    testset = events_matrix[train_size:]
    return trainset, testset


def example():

    # load data
    mat1 = np.load('../data/sim_12360_00.npy').item()  # protons
    mat2 = np.load('../data/sim_12362_00.npy').item()  # iron

    # cleanup mc and reco attributes
    cleanup((mat1, mat2))

    # Make into gain differentiated data set
    hgain, lgain, all_sensors = gain_differentiate(mat1)

    # Make event matrix
    hgain_indices = get_index_dict(hgain)   # Construct dictionary that goes name->column index
    hgain_events = gen_matrix(mat1, mat2, hgain_indices)

    # Split into testing and training sets
    trainset, testset = split_train(hgain_events)
    np.save('../data_processed/small_train_high_low.npy', trainset)
    np.save('../data_processed/small_test_high_low.npy', testset)

    # Add azimuth, zenith features
    all_indices_dir = get_index_dict(all_sensors, direction_data=True)
    all_events_direction = gen_matrix(mat1, mat2, all_indices_dir, direction=True)
    trainset, testset = split_train(all_events_direction)
    np.save('../data_processed/direction_train.npy', trainset)
    np.save('../data_processed/direction_test.npy', testset)
