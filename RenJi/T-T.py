import os
import shutil

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Preprocess.Nii2NPY import Resampler

# data_folder = r'Z:\RenJi\ExternalTest\CH3'
# for case in os.listdir(data_folder):
#     image = sitk.ReadImage(os.path.join(data_folder, case))
#     output = Resampler(image, is_roi=False, expected_resolution=[1.0416666269302368, 1.0416666269302368, 1],
#                        store_path=os.path.join(r'Z:\RenJi\ExternalTest\ExternalNii', '{}/resize_3ch.nii.gz'.format(case.split('.nii')[0])))
#     print(case)


def ECEStatistics():
    case_list = os.listdir(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\AdcSlice')
    case_list = [case.split('_-_')[0] for case in case_list]
    train_list = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\alltrain-name.csv').values.tolist()[0]
    train_list = [case for case in train_list if case in case_list]
    internal_list = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\test-name.csv').values.tolist()[0]
    internal_list = [case for case in internal_list if case in case_list]

    jsph_df = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\ECE-ROI.csv', index_col='case', encoding='gbk')
    suh_df = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\SUH_Dwi1500\SUH_ECE_clinical-report.csv', index_col='case', encoding='gbk')
    external_list = suh_df.index.tolist()

    train_age = np.array(jsph_df.loc[train_list, 'age'].values.tolist(), dtype=np.float32)
    internal_age = np.array(jsph_df.loc[internal_list, 'age'].values.tolist(), dtype=np.float32)
    external_age = np.array(suh_df.loc[external_list, 'age'].values.tolist(), dtype=np.float32)

    print('age')
    print(np.mean(train_age), np.std(train_age), np.quantile(train_age, 0.25), np.quantile(train_age, 0.75))
    print(np.mean(internal_age), np.std(internal_age), np.quantile(internal_age, 0.25), np.quantile(internal_age, 0.75))
    print(np.mean(external_age), np.std(external_age), np.quantile(external_age, 0.25), np.quantile(external_age, 0.75))

    train_psa = np.array(jsph_df.loc[train_list, 'psa'].values.tolist(), dtype=np.float32)
    internal_psa = np.array(jsph_df.loc[internal_list, 'psa'].values.tolist(), dtype=np.float32)
    external_psa = np.array(suh_df.loc[external_list, 'PSA'].values.tolist(), dtype=np.float32)
    print('PSA')
    print(np.mean(train_psa), np.std(train_psa), np.quantile(train_psa, 0.25), np.quantile(train_psa, 0.75))
    print(np.mean(internal_psa), np.std(internal_psa), np.quantile(internal_psa, 0.25), np.quantile(internal_psa, 0.75))
    print(np.mean(external_psa), np.std(external_psa), np.quantile(external_psa, 0.25), np.quantile(external_psa, 0.75))

    train_pGs = np.array(jsph_df.loc[train_list, 'bGs'].values.tolist(), dtype=np.float32)
    internal_pGs = np.array(jsph_df.loc[internal_list, 'bGs'].values.tolist(), dtype=np.float32)
    external_pGs = np.array(suh_df.loc[external_list, '手术GS grade'].values.tolist(), dtype=np.float32)
    print('pGs')
    print('{}/({:.1f}), {}/({:.1f}), {}/({:.1f}), {}/({:.1f})'.format(sum(train_pGs == 1), sum(train_pGs == 1)/596*100,
                                                                      sum(train_pGs == 2), sum(train_pGs == 2)/596*100,
                                                                      sum(train_pGs == 3), sum(train_pGs == 3)/596*100,
                                                                      sum(train_pGs > 3), sum(train_pGs > 3)/596*100))
    print('{}/({:.1f}), {}/({:.1f}), {}/({:.1f}), {}/({:.1f})'.format(sum(internal_pGs == 1), sum(internal_pGs == 1)/150*100,
                                                                      sum(internal_pGs == 2), sum(internal_pGs == 2)/150*100,
                                                                      sum(internal_pGs == 3), sum(internal_pGs == 3)/150*100,
                                                                      sum(internal_pGs > 3), sum(internal_pGs > 3)/150*100))
    print('{}/({:.1f}), {}/({:.1f}), {}/({:.1f}), {}/({:.1f})'.format(sum(external_pGs == 1), sum(external_pGs == 1)/146*100,
                                                                      sum(external_pGs == 2), sum(external_pGs == 2)/146*100,
                                                                      sum(external_pGs == 3), sum(external_pGs == 3)/146*100,
                                                                      sum(external_pGs > 3), sum(external_pGs > 3)/146*100))

    train_pece = np.array(jsph_df.loc[train_list, 'pECE'].values.tolist(), dtype=np.float32)
    internal_pece = np.array(jsph_df.loc[internal_list, 'pECE'].values.tolist(), dtype=np.float32)
    external_pece = np.array(suh_df.loc[external_list, '包膜突破'].values.tolist(), dtype=np.float32)
    print('ece')
    print('{}/({:.1f}), {}/({:.1f})'.format(sum(train_pece == 1), sum(train_pece == 1) / 596 * 100,
                                            sum(train_pece == 0), sum(train_pece == 0) / 596 * 100))
    print('{}/({:.1f}), {}/({:.1f}),'.format(sum(internal_pece == 1), sum(internal_pece == 1) / 150 * 100,
                                             sum(internal_pece == 0), sum(internal_pece == 0) / 150 * 100))
    print('{}/({:.1f}), {}/({:.1f})'.format(sum(external_pece == 1), sum(external_pece == 1) / 146 * 100,
                                            sum(external_pece == 0), sum(external_pece == 0) / 146 * 100))


def CheckNPYValue():
    npy_folder = r'/home/zhangyihong/Documents/RenJi/Data/SegData/NPY'
    for case in os.listdir(npy_folder):
        data = np.load(os.path.join(npy_folder, case))
        # print(np.std(data), np.mean(data), np.min(data), np.max(data))
        print(data.shape)
# CheckNPYValue()


def MergeData():
    case_list = os.listdir(r'/home/zhangyihong/Documents/RenJi/Data/3CHROINPY')
    [shutil.move(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/2CHNPY', case),
                 os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/SegData/NPY', '2ch_{}'.format(case))) for case in case_list]
    [shutil.move(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/3CHNPY', case),
                 os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/SegData/NPY', '3ch_{}'.format(case))) for case in case_list]

    [shutil.move(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/2CHROINPY', case),
                 os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/SegData/ROINPY', '2ch_{}'.format(case))) for case in case_list]
    [shutil.move(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/3CHROINPY', case),
                 os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/SegData/ROINPY', '3ch_{}'.format(case))) for case in case_list]
# MergeData()


def SavebySlice():
    npy_folder = r'/home/zhangyihong/Documents/RenJi/Data/SegData/NPY'
    for case in os.listdir(npy_folder):
        data = np.load(os.path.join(npy_folder, case))
        roi = np.load(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/SegData/ROINPY', case))
        new_data = data[0]
        new_roi = roi[0]
        np.save(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/SegData/NPYOneSlice', case), new_data)
        np.save(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/SegData/ROIOneSlice', case), new_roi)
# SavebySlice()


def CheckbySlice():
    npy_folder = r'/home/zhangyihong/Documents/RenJi/Data/SegData/NPYOneSlice'
    roi_folder = r'/home/zhangyihong/Documents/RenJi/Data/SegData/ROIOneSlice'
    for case in os.listdir(npy_folder):
        data = np.load(os.path.join(npy_folder, case))
        roi = np.load(os.path.join(roi_folder, case))
        data = data[np.newaxis, ...]
        roi = roi[np.newaxis, ...]
        np.save(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/SegData/NPYOneSlice', case), data)
        np.save(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/SegData/ROIOneSlice', case), roi)
        # plt.imshow(data, cmap='gray')
        # plt.contour(roi, colors='r')
        # plt.savefig(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/SegData/Image', case.split('.npy')[0]))
        # plt.close()
CheckbySlice()
