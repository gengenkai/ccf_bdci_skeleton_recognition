import os

import mmcv
import numpy as np

ski_bone = ((0, 1), (1, 1), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6),
            (8, 1), (9, 8), (10, 9), (11, 10), (12, 8), (13, 12), (14, 13),
            (15, 0), (16, 0), (17, 15), (18, 16), (19, 14), (20, 19), (21, 14),
            (22, 11), (23, 22), (24, 11))

sets = ['train', 'val', 'test']

for set in sets:

    results = []
    path = '{}/{}.pkl'.format('/mnt/lustre/data/ski', set)
    data = mmcv.load(path)
    print('len(data)', len(data))

    prog_bar = mmcv.ProgressBar(len(data))
    for i, item in enumerate(data):

        keypoint = item['keypoint']
        M, T, V, C = keypoint.shape
        bone = np.zeros((M, T, V, C), dtype=np.float32)
        for v1, v2 in ski_bone:
            bone[:, :, v1, :] = keypoint[:, :, v1, :] - keypoint[:, :, v2, :]
        item['keypoint'] = bone

        results.append(item)
        prog_bar.update()

    out_path = '/mnt/lustre/data/ski/bone'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path = '{}/{}.pkl'.format(out_path, set)
    mmcv.dump(results, out_path)
    print(f'{out_path} finish!!!!~')
