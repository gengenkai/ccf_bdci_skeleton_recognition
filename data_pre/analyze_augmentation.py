import random

import mmcv
import numpy as np

angle_candidate = [-10., -5., 0., 5., 10.]
scale_candidate = [0.9, 1.0, 1.1]
transform_candidate = [-0.2, -0.1, 0.0, 0.1, 0.2]
move_time_candidate = [1]

sets = ['train']

aug_classes = [3, 6, 7, 10, 12, 14, 15, 17, 20, 24, 25, 28]
# extreme_classes = [3, 12, 14, 17, 20, 24, 25]

# do the data augmentation on the less-sample class
for set in sets:
    path = '{}/{}.pkl'.format('/mnt/lustre/data/ski/motion_xy', set)
    data = mmcv.load(path)

    results = []

    prog_bar = mmcv.ProgressBar(len(data))
    for i, anno in enumerate(data):
        keypoint = anno['keypoint']
        M, T, V, C = keypoint.shape

        label = anno['label']
        results.append(anno)

        if label in aug_classes:
            aug_anno = dict()
            aug_anno['total_frames'] = anno['total_frames']
            aug_anno['img_shape'] = anno['img_shape']
            aug_anno['original_shape'] = anno['original_shape']
            aug_anno['label'] = anno['label']
            aug_keypoint = keypoint.transpose(3, 1, 2,
                                              0).copy()  # MTVC -> CTVM

            move_time = random.choice(move_time_candidate)
            node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
            node = np.append(node, T)
            num_node = len(node)

            A = np.random.choice(angle_candidate, num_node)
            S = np.random.choice(scale_candidate, num_node)
            T_x = np.random.choice(transform_candidate, num_node)
            T_y = np.random.choice(transform_candidate, num_node)

            a = np.zeros(T)
            s = np.zeros(T)
            t_x = np.zeros(T)
            t_y = np.zeros(T)

            # linspace
            for i in range(num_node - 1):
                a[node[i]:node[i + 1]] = np.linspace(
                    A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
                s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                                     node[i + 1] - node[i])
                t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                                       node[i + 1] - node[i])
                t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                                       node[i + 1] - node[i])

            theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                              [np.sin(a) * s, np.cos(a) * s]])

            # perform transformation
            for i_frame in range(T):
                xy = aug_keypoint[0:2, i_frame, :, :]
                new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
                new_xy[0] += t_x[i_frame]
                new_xy[1] += t_y[i_frame]
                aug_keypoint[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

            aug_anno['keypoint'] = aug_keypoint.transpose(
                3, 1, 2, 0)  # C T V M -> M T V C
            results.append(aug_anno)

        prog_bar.update()

print('Finish augmentation--len(results)=',
      len(results))  # 2954(train)  492(val)

out_path = '/mnt/lustre/data/ski/motion_xy/train_aug2954.pkl'
mmcv.dump(results, out_path)
print(f'{out_path} Finish ～～～～～')
