import os
import shutil

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from BasicTool.MeDIT.Visualization import FlattenImages
from BasicTool.MeDIT.Normalize import Normalize01


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
