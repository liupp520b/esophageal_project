import matplotlib.pyplot as plt
import numpy as np
import torch

def get_data_voc_labels():
    return np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128]
    ])

def decode_segmap(label_mask, dataset, plot=False):
    if dataset == 'data_voc':
        n_class = 5
        label_colours = get_data_voc_labels()
    elif dataset == 'cityspace':
        n_class = 5
        label_colours = get_data_voc_labels()
    else:
        raise NotImplemented
    print(label_mask.shape)
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_class):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def decode_seg_map_sequence(label_masks, dataset='data_voc'):
     rgb_masks = []
     for label_mask in label_masks:
         rgb_mask = decode_segmap(label_mask, dataset)
         rgb_masks.append(rgb_mask)
     rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3 , 1, 2]))
     return rgb_masks