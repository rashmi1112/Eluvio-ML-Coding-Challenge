import glob
import os
import pathlib
import sys

import pandas as pd


def unpickle_data(path_name):
    read_data = []
    for table in pathlib.Path(path_name).glob("*.pkl"):
        read_data.append(pd.read_pickle(table))
    return read_data


if __name__ == '__main__':
    print('Screen Segmentation')
    data_dir = sys.argv[1]
    filenames = glob.glob(os.path.join(data_dir, "tt*.pkl"))
    # path = r"data\data_dir"
    read_data = unpickle_data(filenames)
    print('Successfully unpickled {} files!'.format(len(read_data)))
