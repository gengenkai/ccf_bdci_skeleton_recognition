import os

import mmcv
import numpy as np

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
        motion = np.zeros((M, T, V, 5), dtype=np.float32)
        for t in range(T - 2):
            motion[:,
                   t, :, :2] = keypoint[:, t + 1, :, :2] - keypoint[:,
                                                                    t, :, :2]
            motion[:, t, :,
                   2:4] = keypoint[:, t + 2, :, :2] - keypoint[:, t, :, :2]
            motion[:, t, :,
                   -1] = (keypoint[:, t + 1, :, 2] + keypoint[:, t, :, 2]) / 2
        item['keypoint'] = motion

        results.append(item)
        prog_bar.update()

    out_path = '/mnt/lustre/data/ski/motion_xy_5chan'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path = '{}/{}.pkl'.format(out_path, set)
    mmcv.dump(results, out_path)
    print(f'{out_path} finish!!!!~\n')
