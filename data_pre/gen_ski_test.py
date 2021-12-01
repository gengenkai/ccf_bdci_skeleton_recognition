import mmcv
import numpy as np

output_pkl = '/mnt/lustre/data/ski/test.pkl'
test_data = '/mnt/lustre/data/ski/test_A_data.npy'

data = np.load(test_data)
n_samples = len(data)
print(n_samples)

results = []
prog_bar = mmcv.ProgressBar(n_samples)

for i, keypoint in enumerate(data):
    anno = dict()
    anno['total_frames'] = 2500
    anno['keypoint'] = keypoint.transpose(3, 1, 2, 0)  # C T V M -> M T V C
    anno['img_shape'] = (1080, 720)
    anno['original_shape'] = (1080, 720)
    anno['label'] = 0
    results.append(anno)
    prog_bar.update()

mmcv.dump(results, output_pkl)
print(f'{output_pkl}---Finish!')
