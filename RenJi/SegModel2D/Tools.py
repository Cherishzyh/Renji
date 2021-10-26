import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def KeepLargest(mask):
    def _Keep(mask):
        new_mask = np.zeros(mask.shape)
        label_im, nb_labels = ndimage.label(mask)
        max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
        index = np.argmax(max_volume)
        new_mask[label_im == index + 1] = 1
        return new_mask

    if np.ndim(mask) == 3:
        mask_list = [_Keep(mask[slice]) for slice in range(mask.shape[0])]
        new_mask = np.array(mask_list)
    elif np.ndim(mask) == 2:
        new_mask = _Keep(mask)
    else:
        new_mask = None
        print('check roi!')
    return new_mask