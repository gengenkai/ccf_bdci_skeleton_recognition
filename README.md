# ccf bdci 2021

这里分享一个 ccf bdci 2021 花样滑冰选手骨骼点动作识别比赛的基于 MMAction2 的解决方案。
我们在 a 榜的分数是 **73.56**。

Here we provide a MMAction2-based solution for the competition [2021 ccf bdci](https://www.datafountain.cn/competitions/519).

<div align="center">
  <img src="https://user-images.githubusercontent.com/30782254/144176051-9bbbe5bb-c83f-4d8c-99d5-656f96eb5075.png" width="1300"/>
</div>

## Installation

- MMCV >= 1.3.14

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```

Please replace ``{cu_version}`` and ``{torch_version}`` in the url to your desired one. For example, to install the latest ``mmcv-full`` with ``CUDA 11`` and ``PyTorch 1.7.0``, use the following command:

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
```

See [here](https://github.com/open-mmlab/mmcv#installation) for different versions of MMCV compatible to different PyTorch and CUDA versions.

- MMAction2 >= 0.20.0

```shell
pip install mmaction2
```

## Data preprocess

You can use the following command to preprocess the data:

1. generate the joint data (比赛提供的skeleton数据， 包含三维信息 (x,y,confidence))

```shell
python data_pre/gen_ski_all.py
python data_pre/gen_ski_test.py
python data_pre/no_padframe.py
```

2. generate the bone data (每个 bone 由两个skeleton连接而成， 离中心点远的 skeleton 和离中心点近的 skeleton 在 (x ,y, confidence) 维度做差)

```shell
python data_pre/gen_ski_all_bone.py
```

3. generate the bone_xy data (跟 Bone 类似，但是仅在 x 和 y 维度上做差， confidence 维度上取两个 joint 的confidence均值)

```shell
python data_pre/gen_ski_all_bone_xy.py
```

4. generate the motion data (时间间隔 T=1 相邻两帧的对应 skeleton 在 x, y 维度上做差， confidence 维度上取两个skeleton的confidence均值)

```shell
python data_pre/gen_ski_all_motion.py
```

5. generate the  motion2 data  (时间间隔 T=1 相邻两帧的对应 skeleton 在 x,y 维度上做差， 时间间隔 T=2 相邻两帧的对应 skeleton 在 x,y 维度上做差， confidence 维度上取时间间隔 T=1 相邻两帧的对应 skeleton 的confidence均值)

```shell
python data_pre/gen_ski_all_motion_5chan.py
```

6. generate the data for PoseC3D model

```shell
python data_pre/gen_posec3d_all.py
```

7. augmentation (对少样本类别的 x,y 坐标进行一定的随机旋转和位移)

```shell
python data_pre/analyze_augmentation.py
```

## Model training scripts

```shell
GPUS=${GPUS} tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} --work-dir ${WORK_DIR} --validate
```

我们比赛的解决方案中共使用 9 个模型， 对应的 config 文件都置于 **configs/** 目录下:

其中 2s_bone_aug.py, 2s_bone.py, 2s_joint.py, posec3d_keypoint.py, posec3d_limb.py 使用 8 个 GPU 训练,
ms_bone_xy_500.py, ms_bone_xy_600.py, ms_motion_xy_500_5chan.py 使用 1 个 GPU 训练,
ms_motion_xy_750.py 使用 2 个 GPU 训练。

## Model testing scripts and generate outputs

```shell
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_DIR} --eval top_k_accuracy
    --out joint.pkl
```

要生成 test 数据的预测结果时， 把 ann_file_val 的路径改成对应的 test.pkl 的路径。
同时， 由于上面把比赛的带标注的数据集分成了 train/val 来训练和验证， 我们需要将所有这些
train/val 都用来训练，训练的 epoch 数取决于之前 train 的时候 best_pth 所在的epoch，
最后用 retrain 的模型来生成对比赛 test 数据的预测结果。

## Model ensemble

```shell
python data_pre/gen_test_output_ensemble.py
```

在融合 softmax score 时，GCN-based 模型的系数选择了 1/7，PoseC3D 模型的系数选择了 1/10。


## License

This project is released under the [Apache 2.0 license](LICENSE).
