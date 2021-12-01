model = dict(type='Recognizer3D',
             backbone=dict(type='ResNet3dSlowOnly',
                           depth=50,
                           pretrained=None,
                           in_channels=24,
                           base_channels=32,
                           num_stages=3,
                           out_indices=(2, ),
                           stage_blocks=(4, 6, 3),
                           conv1_stride_s=1,
                           pool1_stride_s=1,
                           inflate=(0, 1, 1),
                           spatial_strides=(2, 2, 2),
                           temporal_strides=(1, 1, 2),
                           dilations=(1, 1, 1)),
             cls_head=dict(type='I3DHead',
                           in_channels=512,
                           num_classes=30,
                           spatial_type='avg',
                           dropout_ratio=0.5),
             train_cfg=dict(),
             test_cfg=dict(average_clips='prob'))

dataset_type = 'PoseDataset'
# ann_file_train = 'skitrain_all.pkl'
ann_file_train = 'skitrain.pkl'
ann_file_val = 'skitest.pkl'
left_kp = [2, 3, 4, 9, 10, 11, 22, 23, 24, 15, 17]
right_kp = [5, 6, 7, 12, 13, 14, 19, 20, 21, 16, 18]
skeletons = [[0, 15], [0, 16], [15, 17], [16, 18], [0, 1], [1, 2], [2, 3],
             [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [10, 11],
             [11, 24], [11, 22], [22, 23], [8, 12], [12, 13], [13, 14],
             [14, 21], [14, 19], [19, 20]]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget',
         sigma=0.6,
         use_score=True,
         with_kp=False,
         with_limb=True,
         skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(type='GeneratePoseTarget',
         sigma=0.6,
         use_score=True,
         with_kp=False,
         with_limb=True,
         skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=10,
         test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(type='GeneratePoseTarget',
         sigma=0.6,
         use_score=True,
         with_kp=False,
         with_limb=True,
         double=True,
         left_kp=left_kp,
         right_kp=right_kp,
         skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(videos_per_gpu=16,
            workers_per_gpu=2,
            test_dataloader=dict(videos_per_gpu=1),
            train=dict(type='RepeatDataset',
                       times=30,
                       dataset=dict(type=dataset_type,
                                    ann_file=ann_file_train,
                                    data_prefix='',
                                    pipeline=train_pipeline)),
            val=dict(type=dataset_type,
                     ann_file=ann_file_val,
                     data_prefix='',
                     pipeline=val_pipeline),
            test=dict(type=dataset_type,
                      ann_file=ann_file_val,
                      data_prefix='',
                      pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.2, momentum=0.9,
                 weight_decay=0.0003)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 8
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(interval=1,
                  metrics=['top_k_accuracy', 'mean_class_accuracy'],
                  topk=(1, 5))
log_config = dict(interval=20, hooks=[
    dict(type='TextLoggerHook'),
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ski/ski_limb'
load_from = None
resume_from = None
