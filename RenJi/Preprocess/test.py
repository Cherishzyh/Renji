import os
import shutil

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from copy import deepcopy
from pathlib import Path

from MeDIT.Visualization import FlattenImages
from MeDIT.Normalize import Normalize01


def LoadImage(path):
    list = os.listdir(path)
    nii = [a for a in list if '2CH' in a]
    if len(nii) > 0:
        image = sitk.ReadImage(os.path.join(path, nii[0]))
        print(image.GetSpacing())
        data = sitk.GetArrayFromImage(image)
        return np.squeeze(data)
    else:
        image = sitk.ReadImage(os.path.join(path, list[-1]))
        data = sitk.GetArrayFromImage(image)
        return np.squeeze(data)
        # print(path, 'no 2ch')
        # return 'a'


def DataClip(data):
    max = np.percentile(data, 99.9)
    min = np.percentile(data, 0.1)
    data = np.clip(data, a_min=min, a_max=max)
    return data


def demo():
    path_1 = r'D:\Data\renji\Nii\Nii0722\20200721 gurenhua'
    path_2 = r'D:\Data\renji\Nii\Nii0721\20200721 gurenhua 1'

    for case in os.listdir(path_1):
        print(case)
        # remove_path = os.path.join(path_1, case)
        # repeat_path = os.path.join(path_2, case)
        remove_path = path_1
        repeat_path = path_2
        if os.path.exists(repeat_path):
            data1 = LoadImage(remove_path)
            data2 = LoadImage(repeat_path)
            if isinstance(data1, str) or isinstance(data1, str):
                print(case, 'no 2ch data')
                continue
            print(data1.shape, data2.shape)
            if np.ndim(data1) == 4:
                data1 = data1[:, 1, ...]
            if np.ndim(data2) == 4:
                data2 = data2[:, 1, ...]
            # data1 = DataClip(data1)
            # data2 = DataClip(data2)
            # data2 = np.flip(np.flip(data2, axis=1), axis=2)
            try:
                print((Normalize01(data1) == Normalize01(data2)).all())

                if (Normalize01(data1) == Normalize01(data2)).all() == False:
                    data1_flatten = FlattenImages(data1, is_show=False)
                    data2_flatten = FlattenImages(data2, is_show=False)
                    plt.figure(figsize=(16, 8), dpi=100)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.subplot(121)
                    plt.imshow(data1_flatten, cmap='gray')
                    plt.subplot(122)
                    plt.imshow(data2_flatten, cmap='gray')
                    plt.show()

                    FlattenImages(Normalize01(data1 - data2), is_show=True)
            except Exception:
                continue
        break
# demo()


def RemoveMS():
    root = r'D:\Data\renji\AllNii'
    for case in os.listdir(root):
        case_folder = os.path.join(root, case)
        case_list = os.listdir(case_folder)
        ch2_list = [nii for nii in case_list if '2CH' in nii]
        if len(ch2_list) == 0:
            print(case)
            print(ch2_list)
            print( )
            shutil.move(case_folder, os.path.join(r'C:\Users\82375\Desktop\NO2CH', case))
# RemoveMS()


def CompareTwoFolder():
    path1 = r'C:\Users\82375\Desktop\repeat'
    path2 = r'C:\Users\82375\Desktop\move'
    list1 = os.listdir(path1)

    for case in list1:
        case_folder1 = os.path.join(path1, case)
        nii_list1 = [case for case in os.listdir(case_folder1) if '2CH' in case]
        if len(nii_list1) == 1:
            image1 = sitk.ReadImage(os.path.join(case_folder1, nii_list1[0]))
            data1 = sitk.GetArrayFromImage(image1)
        else:
            print('have no {} in path1'.format(case))
            continue
        case_folder2 = os.path.join(path2, case)
        nii_list2 = [case for case in os.listdir(case_folder2) if '2CH' in case]
        if len(nii_list2) == 1:
            image2 = sitk.ReadImage(os.path.join(case_folder2, nii_list2[0]))
            data2 = sitk.GetArrayFromImage(image2)
        else:
            print('have no {} in path2'.format(case))
            continue
        if (data1 == data2).all():
            shutil.move(case_folder1, os.path.join(r'D:\Data\renji\AllNii', case))
            shutil.rmtree(case_folder2)
        else:
            print('{}: data in path1 is different from data in path2'.format(case))
# CompareTwoFolder()


def CompareCase():
    path = r'D:\Data\renji\AllNii'
    list = os.listdir(path)
    list = ['{} {}'.format(case.split(' ')[0], case.split(' ')[1]) for case in list]
    from collections import Counter  # 引入Counter
    b = dict(Counter(list))
    repeat_case = [key for key, value in b.items() if value > 1]  # 只展示重复元素

    # print(repeat_case)
    # print(len(repeat_case))
    return repeat_case
# CompareCase()


def CheckSameCase():
    repeat_case = CompareCase()
    path = r'D:\Data\renji\AllNii'
    list = os.listdir(path)
    repeat_list = [case for case in list if '{} {}'.format(case.split(' ')[0], case.split(' ')[1]) in repeat_case]
    # print(repeat_list)
    [shutil.move(os.path.join(path, file), os.path.join(r'C:\Users\82375\Desktop\repeat', file)) for file in repeat_list]
# CheckSameCase()


# data_folder = r'C:\Users\82375\Desktop\NO2CH'
# for case in os.listdir(data_folder):
#     if len(os.listdir(os.path.join(data_folder, case))) < 4:
#         print(case)
# for case in os.listdir(r'C:\Users\82375\Desktop\RepeatCase'):
#     case_folder = os.path.join(r'C:\Users\82375\Desktop\RepeatCase', case)
#     nii = [a for a in os.listdir(case_folder) if '2CH' in a]
#     if len(nii) == 0:
#         nii_path = os.path.join(case_folder, os.listdir(case_folder)[-1])
#         image = sitk.ReadImage(nii_path)
#         data = np.squeeze(sitk.GetArrayFromImage(image))
#         try:
#             data_flatten = FlattenImages(data, is_show=False)
#             plt.figure(figsize=(16, 17), dpi=100)
#             plt.title(case)
#             plt.gca().xaxis.set_major_locator(plt.NullLocator())
#             plt.gca().yaxis.set_major_locator(plt.NullLocator())
#             plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#             plt.margins(0, 0)
#             plt.imshow(data_flatten, cmap='gray')
#             plt.show()
#         except Exception:
#             print(case)

def CopyNii(src_root):
    # des_root = r'D:\Data\renji\AllNii'
    des_root = r'D:\Data\renji\Nii0723'
    if not os.path.exists(des_root):
        os.mkdir(des_root)
    for root, dirs, files in os.walk(src_root):
        if len(files) > 0 and len(dirs) == 0:
            case_name = Path(root).name
            case_root = Path(root).parent
            while True:
                if len(str(case_name)) > 9:
                    break
                else:
                    case_name = Path(case_root).name
                    case_root = Path(case_root).parent
            des_folder = os.path.join(des_root, case_name)
            if not os.path.exists(des_folder):
                os.mkdir(des_folder)
            # print(case_name)
            # [shutil.copyfile(os.path.join(root, file), os.path.join(des_folder, file)) for file in files if file.endswith('.nii')]
        elif len(files) == 0 and len(dirs) == 0:
            case_name = Path(root).name
            case_root = Path(root).parent
            print(case_root, case_name)
# CopyNii(r'D:\Data\renji\renji0723\RawData')


def Check():
    data_folder = r'D:\Data\renji\Npy'
    for case in os.listdir(data_folder):
        data_path = os.path.join(data_folder, case)
        data = np.load(data_path)
        if data.shape == (1, 30, 240, 240):
            continue
        else:
            print(case, '\t', data.shape)
# Check()


def DealWithDifference():
    from MeDIT.Normalize import Normalize01
    from MeDIT.Visualization import Imshow3DArray
    # def _KeepLargest(mask):
    #     new_mask = np.zeros(mask.shape)
    #     label_im, nb_labels = ndimage.label(mask)
    #     Imshow3DArray(Normalize01(label_im.transpose((1, 2, 0))))
    #     index_list = [index for index in range(1, nb_labels + 1) if (label_im == index).sum() > 100]
    #     for index in index_list:
    #         new_mask[label_im == index + 1] = 1
    #     return label_im, nb_labels, new_mask

    data_folder = r'Z:\RenJi\Npy2D'
    for case in os.listdir(data_folder):
        case_path = os.path.join(data_folder, case)
        data = np.load(case_path)
        diff_data = data - data[0]
        diff_data = Normalize01(diff_data)
        # diff_data = np.clip(data, a_min=np.percentile(diff_data, 0.01), a_max=np.percentile(diff_data, 99.9))
        # plt.hist(diff_data.flatten(), bins=20)
        # plt.show()
        threshold = np.percentile(diff_data, 90)
        thres_diff = deepcopy(diff_data)
        thres_diff[thres_diff < threshold] = 0
        thres_diff[thres_diff >= threshold] = 1
        new_mask = ndimage.median_filter(thres_diff, size=7)
        Imshow3DArray(Normalize01(new_mask.transpose((1, 2, 0))))
        # new_mask = thres_diff
        # _, _, new_mask = _KeepLargest(thres_diff)
        # Imshow3DArray(np.concatenate([Normalize01(data.transpose((1, 2, 0))), Normalize01(diff_data.transpose((1, 2, 0)))], axis=1),
        #               roi=np.concatenate([Normalize01(new_mask.transpose((1, 2, 0))), Normalize01(new_mask.transpose((1, 2, 0)))], axis=1))
# DealWithDifference()


def RemoveNormal():
    label_path = r'Z:\RenJi\label_norm.csv'
    df = pd.read_csv(label_path, index_col='CaseName')
    case_list = []
    label_list = []
    for case in df.index:
        if df.loc[case].item() <= 1:
            case_list.append(case)
            label_list.append(int(0))
        elif df.loc[case].item() >= 2:
            case_list.append(case)
            label_list.append(int(1))
    new_df = pd.DataFrame({'CaseName': case_list, 'Label': label_list})
    new_df.to_csv(r'Z:\RenJi\label_2cl.csv', index=False)
# RemoveNormal()


def Compare3CHCase():
    data_folder = r'Z:\RenJi\3CHHeathy20211012\3CH'
    csv_path = r'Z:\RenJi\normal_case.csv'
    csv_list = pd.read_csv(csv_path, index_col='CaseName').index.tolist()
    case_list = os.listdir(data_folder)
    case_list = ['{} {}'.format(case.split(' ')[0], case.split(' ')[1]) for case in case_list]

    case_in_csv = [case for case in csv_list if case not in case_list]
    case_in_folder = [case for case in case_list if case not in csv_list]

    print()
# Compare3CHCase()

# print(sitk.ReadImage(r'Z:\RenJi\RawDataUseful\Combine\20141010 liujinxin\501_c3CH.nii').GetSpacing())
# print(sitk.ReadImage(r'Z:\RenJi\RawDataUseful\Combine\20141010 liujinxin\501_c3CH.nii').GetSize())
#
# print(sitk.ReadImage(r'Z:\RenJi\RawDataUseful\Combine\20140923 suyongming\501_B-TFE_BH_3CH.nii').GetSpacing())
# print(sitk.ReadImage(r'Z:\RenJi\RawDataUseful\Combine\20140923 suyongming\501_B-TFE_BH_3CH.nii').GetSize())
# print(sitk.ReadImage(r'Z:\RenJi\3CHHeathy20211012\3CH\20161020 chenzhonghua\3CH_MASK.nii.gz').GetSpacing())


def Combine():
    folder = r'/home/zhangyihong/Documents/RenJi/3CHROINPY'
    case_list = os.listdir(folder)
    # case_list = [case.split('.npy')[0] for case in case_list]
    for case in case_list:
        shutil.copyfile(os.path.join(r'/home/zhangyihong/Documents/RenJi/3CHROINPY', case),
                        os.path.join(r'/home/zhangyihong/Documents/RenJi/CaseWithROI/RoiNPY', '3ch_{}'.format(case)))
# Combine()

# for case in os.listdir(r'/home/zhangyihong/Documents/RenJi/CaseWithROI/NPY'):
#     data = np.squeeze(np.load(os.path.join(r'/home/zhangyihong/Documents/RenJi/CaseWithROI/NPY', case)))
#     roi = np.squeeze(np.load(os.path.join(r'/home/zhangyihong/Documents/RenJi/CaseWithROI/RoiNPY', case)))
#
#     plt.imshow(data[0], cmap='gray')
#     plt.contour(roi[0], color='r')
#     plt.savefig(os.path.join(r'/home/zhangyihong/Documents/RenJi/CaseWithROI/Image', '{}.jpg'.format(case.split('.npy')[0])))
#     plt.close()


# data_folder = r'/home/zhangyihong/Documents/RenJi/3CHROINPY'
# for case in os.listdir(data_folder):
#     if os.path.exists(os.path.join(r'/home/zhangyihong/Documents/RenJi/2CHNPY', case)):
#         os.remove(os.path.join(r'/home/zhangyihong/Documents/RenJi/2CHNPY', case))
#     if os.path.exists(os.path.join(r'/home/zhangyihong/Documents/RenJi/3CHNPY', case)):
#         os.remove(os.path.join(r'/home/zhangyihong/Documents/RenJi/3CHNPY', case))

# case_list = [case for case in os.listdir(r'/home/zhangyihong/Documents/RenJi/2CHNPY')
#              if case not in os.listdir(r'/home/zhangyihong/Documents/RenJi/3CHNPY')]
# print(case_list)
# label = pd.read_csv(r'/home/zhangyihong/Documents/RenJi/label_2cl.csv', index_col='CaseName')
# for case in case_list:
#     case = case.split('.npy')[0]
#     if case in label.index.tolist():
#         continue
#     else:
#         print(case, 'not in label')

from MeDIT.Normalize import NormalizeZ
for case in os.listdir(r'/home/zhangyihong/Documents/RenJi/CaseWithROI/RoiNPY'):
    if '2ch_' in case:
        data = np.load(os.path.join(r'/home/zhangyihong/Documents/RenJi/CaseWithROI/RoiNPY', case))
        print(data.mean(), data.std())
        # norm_data = NormalizeZ(data)
        # np.save(r'/home/zhangyihong/Documents/RenJi/CaseWithROI/NPY/{}'.format(case), norm_data)



# external_test = pd.read_csv(r'/home/zhangyihong/Documents/RenJi/ExternalTest/external_test.csv', index_col='CaseName')
# case_list = []
# label_list = []
# for case in external_test.index:
#     case_list.append(case)
#     if external_test.loc[case, 'Label'] > 0:
#         label = 1
#         label_list.append(label)
#     elif external_test.loc[case, 'Label'] == 0:
#         label = 0
#         label_list.append(label)
#     else:
#         print('error: {} label={}'.format(case, external_test.loc[case, 'Label']))
#         raise Exception

# new_df = pd.DataFrame({'CaseName': case_list, 'Label': label_list})
# new_df.to_csv(r'/home/zhangyihong/Documents/RenJi/ExternalTest/external_test_2cl.csv', index=False)

# new_df = pd.DataFrame({'CaseName': case_list, 'Label': label_list})
# new_df.to_csv(r'/home/zhangyihong/Documents/RenJi/ExternalTest/external_test_2cl.csv', index=False)