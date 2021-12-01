import mmcv

train_pkl = '/mnt/lustre/data/ski/motion_xy/train.pkl'
val_pkl = '/mnt/lustre/data/ski/motion_xy/val.pkl'
output_pkl = '/mnt/lustre/data/ski/motion_xy/train_val.pkl'

train_data = mmcv.load(train_pkl)
val_data = mmcv.load(val_pkl)

results = []
prog_bar = mmcv.ProgressBar(len(train_data))
for i, anno in enumerate(train_data):
    results.append(anno)
    prog_bar.update()

prog_bar = mmcv.ProgressBar(len(val_data))
for i, anno in enumerate(val_data):
    results.append(anno)
    prog_bar.update()

mmcv.dump(results, output_pkl)
print(f'{output_pkl} finish !!!')
