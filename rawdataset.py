import os

import numpy as np
from skimage.io import imread
from data_split_type import DataSplitType, get_datasplit_tuples

def load_tiff(path):
    """
    Returns a 4d numpy array: num_imgs*h*w*num_channels
    """
    data = imread(path, plugin='tifffile')
    return data


def get_train_val_data(fpath, datasplit_type, val_fraction, test_fraction):
    # actin-60x-noise2-highsnr.tif  mito-60x-noise2-highsnr.tif
    print('Loading data from', fpath)
    data = load_tiff(fpath)[..., np.newaxis ]
    if datasplit_type == DataSplitType.All:
        return data.astype(np.float32)

    train_idx, val_idx, test_idx = get_datasplit_tuples(val_fraction, test_fraction, len(data), starting_test=True)
    breakpoint()

    if datasplit_type == DataSplitType.Train:
        return data[train_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Val:
        return data[val_idx].astype(np.float32)
    elif datasplit_type == DataSplitType.Test:
        return data[test_idx].astype(np.float32)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = get_train_val_data('/group/jug/ashesh/data/ventura_gigascience/actin-60x-noise2-lowsnr.tif', DataSplitType.Train,
                              0.1, 0.1)

    _, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(data[0, ..., 0])
