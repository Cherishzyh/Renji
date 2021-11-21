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
# CheckbySlice()


# data_folder = r'Z:\RenJi\2CH 20210910\2CH_MASK Data\20140923 suyongming'
# image = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_folder, 'resize_2ch_1117.nii.gz')))
# print(image.shape)

def Image2NPY():
# ['3CH', 'Image2CHExternal', '2CH', 'Image2CH', '2CHExternal', 'Image3CH', '3CHExternal', 'Image3CHExternal']
    raw_folder = r'/home/zhangyihong/Documents/RenJi/Data/PredData/3CHExternal'
    save_folder_5slice = r'/home/zhangyihong/Documents/RenJi/Data/ClassData/3CHExternalROINPYPred_5slice'
    if not os.path.exists(save_folder_5slice):
        os.mkdir(save_folder_5slice)
    save_folder = r'/home/zhangyihong/Documents/RenJi/Data/ClassData/3CHExternalROINPYPred'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    for case in os.listdir(raw_folder):
        case_path = os.path.join(raw_folder, '{}/Mask.nii.gz'.format(case))
        data = sitk.GetArrayFromImage(sitk.ReadImage(case_path))
        new_data = np.concatenate([data[9: 12], data[-2:]], axis=0)
        np.save(os.path.join(save_folder_5slice, '{}.npy'.format(case)), new_data)
        np.save(os.path.join(save_folder, '{}.npy'.format(case)), data)


def CheckData():
    from MeDIT.Visualization import FlattenImages
    a = ['3CH', '2CH']
    for key in a:
        data_folder = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}Pred'.format(key)
        for case in os.listdir(data_folder):
            data = np.load(os.path.join(data_folder, case))
            if data.shape != (30, 180, 180):
                print(case)
        # data_flatten = FlattenImages(data, is_show=False)
        # pred_flatten = FlattenImages(roi, is_show=False)
        # plt.figure(figsize=(8, 8), dpi=100)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.imshow(data_flatten, cmap='gray', vmin=0.)
        # plt.contour(pred_flatten, colors='y')
        # plt.savefig(os.path.join(save_folder, '{}.jpg'.format(case.split('.npy')[0])), pad_inches=0)
        # plt.close()
CheckData()


def GetCenter(mask):
    assert (np.ndim(mask) == 2)
    roi_row = np.sum(mask, axis=1)
    roi_column = np.sum(mask, axis=0)

    row = np.nonzero(roi_row)[0]
    column = np.nonzero(roi_column)[0]

    center = [int(np.mean(row)), int(np.mean(column))]
    return center


def CropData():
    # crop---5slice/5slice_dilate/allslice/allslice_dilate
    from scipy.ndimage import binary_dilation
    from MeDIT.ArrayProcess import ExtractBlock
    a = ['3CH', '2CH', '2CHExternal', '3CHExternal']
    for key in a:
        raw_folder = r'/home/zhangyihong/Documents/RenJi/Data/PredData/{}'.format(key)

        data_folder_slice = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}Pred_5slice'.format(key)
        data_folder = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}Pred'.format(key)
        roi_folder_slice_dilated = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}ROIPred_5slice'.format(key)
        roi_folder_dilated = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}ROIPred'.format(key)
        image_folder = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}Image'.format(key)
        if not os.path.exists(data_folder_slice): os.mkdir(data_folder_slice)
        if not os.path.exists(data_folder): os.mkdir(data_folder)
        if not os.path.exists(roi_folder_slice_dilated): os.mkdir(roi_folder_slice_dilated)
        if not os.path.exists(roi_folder_dilated): os.mkdir(roi_folder_dilated)
        if not os.path.exists(image_folder): os.mkdir(image_folder)

        for case in os.listdir(raw_folder):
            data_path = os.path.join(raw_folder, '{}/Image.nii.gz'.format(case))
            roi_path = os.path.join(raw_folder, '{}/Mask.nii.gz'.format(case))

            data = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
            roi = sitk.GetArrayFromImage(sitk.ReadImage(roi_path))
            data_slice = np.concatenate([data[9: 12], data[-2:]], axis=0)
            roi_slice = np.concatenate([roi[9: 12], roi[-2:]], axis=0)
            try:
                center = GetCenter(roi[0])
            except Exception:
                center = [-1, -1]

            data_crop, _ = ExtractBlock(data, (data.shape[0], 180, 180), [-1, center[0], center[1]])
            roi_crop, _ = ExtractBlock(roi, (data.shape[0], 180, 180), [-1, center[0], center[1]])
            data_crop_slice, _ = ExtractBlock(data_slice, (data_slice.shape[0], 180, 180), [-1, center[0], center[1]])
            roi_crop_slice, _ = ExtractBlock(roi_slice, (data_slice.shape[0], 180, 180), [-1, center[0], center[1]])

            roi_dilate = binary_dilation(roi_crop, structure=np.ones((1, 11, 11)))
            roi_dilate_slice = binary_dilation(roi_crop_slice, structure=np.ones((1, 11, 11)))

            np.save(os.path.join(data_folder_slice, '{}.npy'.format(case)), data_crop_slice)
            np.save(os.path.join(roi_folder_slice_dilated, '{}.npy'.format(case)), roi_dilate_slice)
            np.save(os.path.join(data_folder, '{}.npy'.format(case)), data_crop)
            np.save(os.path.join(roi_folder_dilated, '{}.npy'.format(case)), roi_dilate)

            plt.figure(figsize=(8, 8), dpi=100)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.imshow(data_crop[0], cmap='gray', vmin=0.)
            plt.contour(roi_dilate[0], colors='y')
            plt.savefig(os.path.join(image_folder, '{}.jpg'.format(case.split('.npy')[0])), pad_inches=0)
            plt.close()
# CropData()