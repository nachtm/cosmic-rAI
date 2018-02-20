import numpy as np
import math

# load data
mat1 = np.load('../data/sim_12360_00.npy').item()  # protons
mat2 = np.load('../data/sim_12362_00.npy').item()  # iron

# find high/low gain sensors
hgain = list(filter(lambda x: mat1['Gain'][0][x] == 'High', mat1['Position'][0].keys()))
lgain = list(filter(lambda x: mat1['Gain'][0][x] == 'Low', mat1['Position'][0].keys()))


#build a name->index dict
def get_index_dict(sensors):
    name_index_dict = {}
    for i in range(len(hgain)):
        name_index_dict[hgain[i]] = i
    return name_index_dict
hgain_indices = get_index_dict(hgain)
lgain_indices = get_index_dict(lgain)


# Build the matrix by setting up an empty one and populating it with the values according to the indices we created above
hgain_events = np.zeros((len(mat1['Composition']) + len(mat2['Composition']), len(hgain_indices) + 1))
for i in range(len(mat1['Charges'])):
    event = mat1['Charges'][i]
    for sensor in mat1['Charges'][i].keys():
        try:
            hgain_events[i, hgain_indices[sensor]] = 0 if math.isnan(event[sensor]) else event[sensor]
            # print('updated x: {} y: {} to be {}'.format(i,hgain_indices[sensor],event[sensor]))
        except KeyError:
            continue
    hgain_events[i, -1] = 1
for i in range(len(mat2['Charges'])):
    event = mat2['Charges'][i]
    for sensor in mat2['Charges'][i].keys():
        try:
            hgain_events[len(mat1['Charges']) + i, hgain_indices[sensor]] = 0 if math.isnan(event[sensor]) else event[sensor]
        except KeyError:
            continue
    hgain_events[len(mat1['Charges']) + i, -1] = 0
