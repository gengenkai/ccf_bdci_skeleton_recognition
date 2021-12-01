import csv

import mmcv
import numpy as np
'''
joint_2s + (bone_2s_aug + bone_2s) + (bone_xy_ms_500+600）
+ (motionxy_ms_5chan + 750) +softmax  73.25 + poseC3d 73.56
'''


def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(y)
    return f_x


test_data = '/mnt/lustre/data/ski/test_A_data.npy'
data = np.load(test_data)
n_samples = len(data)  # 628

test_output_joint = 'joint_2s.pkl'  # 2500
test_output_bone = 'bone_2s.pkl'  # 2500
test_output_bone_aug = 'bone_2s_aug.pkl'
test_output_bone_xy = 'bone_xy_ms_500.pkl'
test_output_bone_xy600 = 'bonexy_ms_600.pkl'
test_output_joint_mo = 'motionxy_ms_750.pkl'
test_output_joint_mo5chan = 'motionxy_ms_500_5chan.pkl'
test_output_posec3d_keypoint = 'posec3d_keypoint.pkl'
test_output_posec3d_limb = 'posec3d_limb.pkl'

preds_joint = mmcv.load(test_output_joint)
preds_bone = mmcv.load(test_output_bone)
preds_bone_aug = mmcv.load(test_output_bone_aug)
preds_joint_mo = mmcv.load(test_output_joint_mo)
preds_joint_mo5chan = mmcv.load(test_output_joint_mo5chan)
preds_bone_xy = mmcv.load(test_output_bone_xy)
preds_bone_xy600 = mmcv.load(test_output_bone_xy600)
preds_posec3d_keypoint = mmcv.load(test_output_posec3d_keypoint)
preds_posec3d_limb = mmcv.load(test_output_posec3d_limb)

output_file = 'submission.csv'

values = []

for i in range(len(preds_joint)):
    # posec3d
    c3d_pred = softmax(preds_posec3d_keypoint[i]) + softmax(
        preds_posec3d_limb[i])

    pred = softmax(preds_joint[i])
    pred += softmax(preds_bone[i]) + softmax(preds_bone_aug[i])
    pred += softmax(preds_joint_mo5chan[i]) + softmax(preds_joint_mo[i])
    pred += softmax(preds_bone_xy[i]) + softmax(preds_bone_xy600[i])

    pred = pred / 7.0 + c3d_pred / 10.0

    cate = np.argmax(pred)
    values.append((i, cate))

header = ['sample_index', 'predict_category']
with open(output_file, 'w') as fp:
    writer = csv.writer(fp)
    writer.writerow(header)
    writer.writerows(values)

print('Finish~')
