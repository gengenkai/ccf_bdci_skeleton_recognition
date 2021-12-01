import random

import numpy as np
from mmcv import dump

train_x = np.load('train_data.npy')[..., 0].transpose(
    (0, 2, 3, 1)).astype(np.float16)
train_y = np.load('train_label.npy')
test_x = np.load('test_A_data.npy')[..., 0].transpose(
    (0, 2, 3, 1)).astype(np.float16)


# x is 3D array
def nozero(x):
    score = x[..., -1]
    score_sum = np.sum(score, axis=-1)
    j = 2499
    while score_sum[j] < 1e-2:
        j -= 1
    return x[:j + 1]


train_x = [nozero(x) for x in train_x]
test_x = [nozero(x) for x in test_x]

data = []
i = 0
for xx, yy in zip(train_x, train_y):
    i += 1
    item = {}
    item['keypoint'] = ((xx[:, :, :2] + 1) * 50)[None]
    item['keypoint_score'] = xx[:, :, 2][None]
    item['frame_dir'] = str(i)
    item['img_shape'] = (100, 100)
    item['original_shape'] = (100, 100)
    item['label'] = int(yy)
    item['total_frames'] = xx.shape[0]
    data.append(item)

random.seed(0)
random.shuffle(data)

dump(data, 'skitrain_all.pkl')
dump(data[:2500], 'skitrain.pkl')
dump(data[2500:], 'skival.pkl')

data = []
i = 0
for xx in test_x:
    i += 1
    item = {}
    item['keypoint'] = ((xx[:, :, :2] + 1) * 50)[None]
    item['keypoint_score'] = xx[:, :, 2][None]
    item['frame_dir'] = str(i)
    item['img_shape'] = (100, 100)
    item['original_shape'] = (100, 100)
    item['label'] = 0
    item['total_frames'] = xx.shape[0]
    data.append(item)

dump(data, 'skitest.pkl')
