import random

import mmcv
import numpy as np

random.seed(0)

data = '/mnt/lustre/data/ski/train_data.npy'
label = '/mnt/lustre/data/ski/train_label.npy'

label = np.load(label)

data = np.load(data)

output_train_pkl = '/mnt/lustre/data/ski/train.pkl'
output_val_pkl = '/mnt/lustre/data/ski/val.pkl'

n_samples = len(label)
results = []
prog_bar = mmcv.ProgressBar(n_samples)

for i, keypoint in enumerate(data):
    anno = dict()
    anno['total_frames'] = 2500
    anno['keypoint'] = keypoint.transpose(3, 1, 2, 0)  # C T V M -> M T V C
    anno['img_shape'] = (1080, 720)
    anno['original_shape'] = (1080, 720)
    anno['label'] = int(label[i])
    results.append(anno)
    prog_bar.update()

random.shuffle(results)

# total = 2922 split into train(2500) val(422)
train_list = results[:2500]
val_list = results[2500:]
print(f'len(train)={len(train_list)}, len(val)={len(val_list)}')

mmcv.dump(train_list, output_train_pkl)
mmcv.dump(val_list, output_val_pkl)
print('Finish!')
