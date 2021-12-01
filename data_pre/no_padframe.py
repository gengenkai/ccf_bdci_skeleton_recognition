import os

import mmcv

sets = ['train', 'val', 'test']

for set in sets:
    path = '{}/{}.pkl'.format('/mnt/lustre/data/ski/2500_422', set)
    data = mmcv.load(path)
    print('len(data)--', len(data))

    results = []

    prog_bar = mmcv.ProgressBar(len(data))
    for i, anno in enumerate(data):

        keypoint = anno['keypoint']  # M T V C
        M, T, V, C = keypoint.shape
        total_frames = anno['total_frames']

        for i_p, person in enumerate(keypoint):  # keypoint M T V C
            if person[0].sum() == 0:  # person T V C
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp

            for i_f, frame in enumerate(person):  # frame  V C
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        print(f'start from {i_f} is zero')
                        total_frames = i_f + 1
                        break

        # sub the center joint spine joint (1)
        for v in range(V):
            keypoint[:, :,
                     v, :2] = keypoint[:, :,
                                       v, :2] - keypoint[:, :, 1, :
                                                         2]  # xy coordinate

        anno['keypoint'] = keypoint[:, :total_frames, :, :]
        anno['total_frames'] = total_frames
        results.append(anno)
        prog_bar.update()

    out_dir = '/mnt/lustre/data/ski/2500_422'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = '{}/{}.pkl'.format(out_dir, set)
    mmcv.dump(results, out_path)
    print(f'{set} finish no padding and save real total_frames!!!!~')
