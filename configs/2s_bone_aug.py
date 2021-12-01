custom_imports = dict(
    imports=['nets.stgcn2', 'nets.stgcn3'],
    allow_failed_imports=False,
)
model = dict(type='SkeletonGCN',
             backbone=dict(type='STGCN2',
                           in_channels=3,
                           edge_importance_weighting=True,
                           adj_len=25,
                           graph_cfg=dict(layout='ski', strategy='spatial')),
             cls_head=dict(type='STGCNHead',
                           num_classes=30,
                           in_channels=256,
                           loss_cls=dict(type='CrossEntropyLoss'),
                           num_person=1),
             train_cfg=None,
             test_cfg=None)

dataset_type = 'PoseDataset'
# ann_file_train = '/mnt/lustre/data/ski/bone/train_val_aug3446.pkl'
ann_file_train = '/mnt/lustre/data/ski/bone/train_aug2954.pkl'
ann_file_val = '/mnt/lustre/data/ski/bone/val.pkl'
# ann_file_val = '/mnt/lustre/data/ski/bone/test.pkl'
train_pipeline = [
    dict(type='PaddingWithLoop', clip_len=2500),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PaddingWithLoop', clip_len=2500),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PaddingWithLoop', clip_len=2500),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', input_format='NCTVM', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(videos_per_gpu=4,
            workers_per_gpu=2,
            test_dataloader=dict(videos_per_gpu=1),
            train=dict(type=dataset_type,
                       ann_file=ann_file_train,
                       data_prefix='',
                       pipeline=train_pipeline),
            val=dict(type=dataset_type,
                     ann_file=ann_file_val,
                     data_prefix='',
                     pipeline=val_pipeline),
            test=dict(type=dataset_type,
                      ann_file=ann_file_val,
                      data_prefix='',
                      pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD',
                 lr=0.1,
                 momentum=0.9,
                 weight_decay=0.0001,
                 nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30, 40])
total_epochs = 80
checkpoint_config = dict(interval=3)
evaluation = dict(interval=3, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/2s_bone_aug/'
load_from = None
resume_from = None
workflow = [('train', 1)]
