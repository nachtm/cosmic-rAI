def cleanup(matrices):
    keys = ('dir_MC', 'core_MC', 'dir_reco', 'core_reco')
    for matrix in matrices:
        for key in keys:
            matrix[key] = _cleanup_helper(matrix[key])
            
            
def _cleanup_helper(li):
    new_list = []
    for i in range(len(li)//2):
        new_list.append((li[2*i], li[2*i+1]))
    return new_list
            
    
def gain_differentiate(matrices):
    """Find all the sensors that are high/low gain"""
    hgain = list(filter(
        lambda x: mat1['Gain'][0][x] == 'High', mat1['Position'][0].keys()))
    lgain = list(filter(
        lambda x: mat1['Gain'][0][x] == 'Low', mat1['Position'][0].keys()))
    all_sensors = list(mat1['Gain'][0].keys())
    return (hgain, lgain, all_sensors)


def get_index_dict(sensors, direction_data=False):
    """build a name->index dict"""
    name_index_dict = {}
    for i in range(len(sensors)):
        name_index_dict[sensors[i]] = i
    if direction_data:
        name_index_dict['zenith'] = i+1
        name_index_dict['azimuth'] = i+2
    return name_index_dict