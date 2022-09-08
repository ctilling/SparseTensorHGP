import numpy as np


def load_data(data_path: str):
    ind = []
    y = []
    with open(data_path, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            y.append(float(items[-1]))
            ind.append([int(idx)-1 for idx in items[0:-1]])
        ind = np.array(ind)
        y = np.array(y)
    return ind, y