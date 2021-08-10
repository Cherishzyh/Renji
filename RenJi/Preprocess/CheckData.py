import os
import shutil
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from BasicTool.MeDIT.SaveAndLoad import LoadImage
from BasicTool.MeDIT.Normalize import Normalize01, NormalizeZ
from BasicTool.MeDIT.Visualization import Imshow3DArray, FlattenImages
from BasicTool.MeDIT.Log import CustomerCheck, Eclog


def ImageSave(data, save_path):
    data_flatten = FlattenImages(data, is_show=False)
    plt.figure(figsize=(16, 16), dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(data_flatten, cmap='gray')
    plt.savefig(save_path, pad_inches=0)
    plt.close()


def SaveNii(folder, save_path=r'', failed_folder=r''):
    log = CustomerCheck(os.path.join(failed_folder, 'failed_log.csv'), patient=1,
                             data={'State': [], 'Info': []})
    eclog = Eclog(os.path.join(failed_folder, 'failed_log_details.log')).GetLogger()
    for root, dirs, files in os.walk(folder):
        if len(dirs) == 0 and len(files) > 0:
            image_list = [x for x in files if x.endswith('.nii') and '2CH' in x]
            if len(image_list) == 0:
                log.AddOne(Path(root).name, {'State': 'Save Image failed.', 'Info': 'Do not have 2CH data'})
                eclog.error('{} do not have 2CH data'.format(Path(root).name))
            for image_name in image_list:
                try:
                    image = sitk.ReadImage(os.path.join(root, image_name))
                    data = sitk.GetArrayFromImage(image)
                    data = NormalizeZ(np.squeeze(data))
                    print(Path(root).name, image_name, data.shape)
                    data_flatten = FlattenImages(data, is_show=False)
                    plt.figure(figsize=(16, 16), dpi=100)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.imshow(data_flatten, cmap='gray')
                    plt.savefig(os.path.join(save_path, '{}.jpg'.format(Path(root).name)), pad_inches=0)
                    plt.close()
                except Exception as e:
                    log.AddOne(Path(root).name, {'State': 'Save Image failed.', 'Info': e.__str__()})
                    eclog.error(e)
                    print(e)


def RunSaveNii():
    folder = r'D:\Data\renji0722\ProcessData\Correction&Supplement_1'
    save_path = r'D:\Data\renji0722\ProcessData\Image'
    failed_folder = r'D:\Data\renji0722\Failed'
    SaveNii(folder, save_path, failed_folder)
# RunSaveNii()


def ShowErrorNii():
    case_list = ['20190129 dazifeng', '20190502 sudaming', '20190131 dongweigen']
    data_folder = r'D:\Data\renji0722\ProcessData'
    data_folder_list = ['0', '1', '2', '3']
    for case in case_list:
        for folder in data_folder_list:
            case_path = os.path.join(data_folder, '{}/{}'.format(folder, case))
            if os.path.exists(case_path):
                image_list = [x for x in os.listdir(case_path) if x.endswith('.nii') and '2CH' in x]
                assert len(image_list) == 1
                image = sitk.ReadImage(os.path.join(case_path, image_list[0]))
                data = sitk.GetArrayFromImage(image)
                data = Normalize01(np.squeeze(data))
                Imshow3DArray(data)


def Nii2Npy(nii_folder, npy_folder):
    for root, dirs, files in os.walk(nii_folder):
        if len(dirs) == 0 and len(files) > 0:
            image_list = [x for x in files if x.endswith('.nii') and '2CH' in x]
            if len(image_list) == 0:
                image_name = files[-1]
                try:
                    image = sitk.ReadImage(os.path.join(root, image_name))
                    data = sitk.GetArrayFromImage(image)
                    data = np.squeeze(data).transpose((1, 2, 0))
                    data_filp = np.flip(data, axis=0)
                    if np.count_nonzero(data_filp[:, 0:(data_filp.shape[0] // 2)]) > \
                            np.count_nonzero(data_filp[:, (data_filp.shape[0] // 2):]):
                        data_filp = np.flip(data_filp, axis=1)
                    np.save(os.path.join(npy_folder, '{}.npy'.format(Path(root).name)),
                            data_filp.transpose((2, 0, 1)))
                    ImageSave(NormalizeZ(data_filp.transpose((2, 0, 1))),
                              os.path.join(npy_image_folder, '{}.jpg'.format(Path(root).name)))
                except Exception as e:
                    print(Path(root).name, e)
            else:
                image_name = image_list[0]
                try:
                    image = sitk.ReadImage(os.path.join(root, image_name))
                    data = sitk.GetArrayFromImage(image)
                    data = np.squeeze(data).transpose((1, 2, 0))
                    data_filp = np.flip(data, axis=0)
                    #  ['20200618 yeshengjun', '20210111 chenlin'] change > to <
                    if np.count_nonzero(data_filp[:, 0:(data_filp.shape[0] // 2)]) >\
                            np.count_nonzero(data_filp[:, (data_filp.shape[0] // 2):]):
                        data_filp = np.flip(data_filp, axis=1)
                    # Imshow3DArray(np.concatenate([Normalize01(data), Normalize01(data_filp)],axis=1))
                    np.save(os.path.join(npy_folder, '{}.npy'.format(Path(root).name)), data_filp.transpose((2, 0, 1)))
                    ImageSave(NormalizeZ(data_filp.transpose((2, 0, 1))), os.path.join(npy_image_folder, '{}.jpg'.format(Path(root).name)))
                except Exception as e:
                    print(Path(root).name, e)


def ConcatNii():
    case_path = r'C:\Users\82375\Desktop\20151121 normal 2'
    nii_list = [data for data in os.listdir(case_path) if 'M2D' in data]
    shortest_index = np.argmin(np.array([len(s) for s in nii_list]))
    nii_name = nii_list[shortest_index]
    assert len(nii_list) == 30
    data_list = []
    for nii in nii_list:
        image = sitk.ReadImage(os.path.join(case_path, nii))
        slice = sitk.GetArrayFromImage(image)
        data_list.append(NormalizeZ(slice))
    data = np.squeeze(np.array(data_list))
    # if np.ndim(np.squeeze(data)) == 3:
    #     Imshow3DArray(Normalize01(np.squeeze(data).transpose((1, 2, 0))))
    image_concat = sitk.GetImageFromArray(data)
    image_concat.SetOrigin(image.GetOrigin())
    image_concat.SetSpacing(image.GetSpacing())
    image_concat.SetDirection(image.GetDirection())
    sitk.WriteImage(image_concat, os.path.join(r'D:\Data\renji\ProcessedData\20151121 normal 2', nii_name))
# ConcatNii()


def CheckNoNii():
    dcm_list = ['20210315 gushanhua', '20210409 zhangmeirong', '20210713 liufang']
    store_folder = r'D:\Data\renji0721\RawData'
    folder_list = [folder for folder in os.listdir(store_folder) if not folder.endswith('.rar')]
    for folder in folder_list:
        # print(folder)
        for root, dirs, files in os.walk(os.path.join(store_folder, folder)):
            if len(dirs) > 4 and len(files) == 0:
                # print(len(dirs))
                dcm_list.extend(dirs)

    print(len(dcm_list))
    nii_list = os.listdir(r'D:\Data\renji0721\ProcessedData\Nii')
    print(len(nii_list))
    print([case for case in dcm_list if case not in nii_list])


def CheckNii():
    from BasicTool.MeDIT.ArrayProcess import ExtractBlock
    root = r'D:/Data/renji0723/ProcessedData/NPY'
    npy_image_folder = os.path.join(r'D:/Data/renji0721/ProcessedData', 'NPYImageNorm')
    npy_folder = os.path.join(r'D:/Data/renji0721/ProcessedData', 'NPYNorm')

    if not os.path.exists(npy_image_folder):
        os.mkdir(npy_image_folder)
    if not os.path.exists(npy_folder):
        os.mkdir(npy_folder)
    for case in os.listdir(root):
        print(case)
        case_path = os.path.join(root, case)
        data = np.load(case_path)
        patch_size = (data.shape[0], 192, 192)
        min_5 = np.percentile(data, 1)
        max_5 = np.percentile(data, 99)
        data, _ = ExtractBlock(data, patch_size)
        data = np.clip(data, a_min=min_5, a_max=max_5)
        np.save(os.path.join(npy_folder, case), NormalizeZ(data))
        ImageSave(NormalizeZ(data), os.path.join(npy_image_folder, '{}.jpg'.format(case[: case.index('.npy')])))


def SameData():
    root1 = r'D:\Data\renji\Nii0721'
    root2 = r'D:\Data\renji\Nii0722'
    root3 = r'D:\Data\renji\Nii0723'
    des_root = r'C:\Users\82375\Desktop\RepeatCase'
    des_root = r''
    root_list = [root1, root2, root3]
    all_case_list = []
    all_case_list.extend(os.listdir(root1))
    all_case_list.extend(os.listdir(root2))
    all_case_list.extend(os.listdir(root3))
    # all_case_list = os.listdir(r'D:\Data\renji\ProcessedData')
    all_case_name = ['{} {}'.format(case.split(' ')[0], case.split(' ')[1]) for case in all_case_list]
    from collections import Counter  # 引入Counter
    b = dict(Counter(all_case_name))
    repeat_case = [key for key, value in b.items() if value > 1]  # 只展示重复元素
    repeat_name = sorted([case for case in all_case_list if '{} {}'.format(case.split(' ')[0], case.split(' ')[1]) in repeat_case])
    print(len(repeat_name))
    for case in repeat_name:
        count = 0
        for root in root_list:
            if not os.path.exists(os.path.join(root, case)):
                continue
            else:
                count += 1
                case_folder = os.path.join(root, case)
                des_folder = os.path.join(des_root, case)
                if os.path.exists(des_folder):
                    des_folder = '{}_{}'.format(des_folder, count)
                shutil.move(case_folder, des_folder)
                # nii_name = [nii for nii in os.listdir(case_folder) if '2CH' in nii]
                # if len(nii_name) == 0:
                #     print('have no 2ch data', case_folder)
                #     continue
                # nii_path = os.path.join(case_folder, nii_name[0])
                # image = sitk.ReadImage(nii_path)
                # data = np.squeeze(sitk.GetArrayFromImage(image))
                # if np.ndim(data) == 3:
                #     ImageSave(data, os.path.join(r'D:\Data\SameCase', '{}_{}.jpg'.format(case, nii_name[0])))
                # else:
                #     print('not 3d data', case_folder)



if __name__ == '__main__':
    # ConcatNii()
    # npy_image_folder=r'D:\Data\renji0721\ProcessedData\\NPYImage'

    # Nii2Npy(r'D:\Data\renji0723\ProcessedData\Nii', r'D:\Data\renji0723\ProcessedData\NPY\ThreeD')
    # Nii2Npy(r'D:\Data\renji0722\ProcessData\Nii', r'C:\Users\82375\Desktop\test')
    # Nii2Npy(r'D:\Data\renji0721\ProcessedData\Nii', r'D:\Data\renji0721\ProcessedData\NPY')

    # CheckNii()
    # ConcatNii()
    # root = r'D:\Data\renji0721\ProcessedData\Nii\20210518 liuxiangyu 0'
    # image_list = [x for x in os.listdir(root) if x.endswith('.nii') and '2CH' in x]
    # if len(image_list) == 0:
    #     print('{} have no data'.format(Path(root).name))
    # for image_name in image_list:
    #     image = sitk.ReadImage(os.path.join(root, image_name))
    #     data = sitk.GetArrayFromImage(image)
    #     data = np.squeeze(data).transpose((1, 2, 0))
    #     data_filp = np.flip(data, axis=0)
    #     if np.count_nonzero(data_filp[:, 0:(data_filp.shape[0] // 2)]) > \
    #             np.count_nonzero(data_filp[:, (data_filp.shape[0] // 2):]):
    #         data_filp = np.flip(data_filp, axis=1)
    #     # Imshow3DArray(np.concatenate([Normalize01(data), Normalize01(data_filp)],axis=1))
    #     np.save(os.path.join(r'D:\Data\renji0721\ProcessedData\NPY', '{}.npy'.format(Path(root).name)), data_filp.transpose((2, 0, 1)))
    #     ImageSave(NormalizeZ(data_filp.transpose((2, 0, 1))),
    #               os.path.join(npy_image_folder, '{}.jpg'.format(Path(root).name)))
    # CheckNoNii()
    # list1 = os.listdir(r'D:\Data\renji0723\ProcessedData\Nii')
    # # list1 = [case[: case.index('.nii')] for case in list1]
    # list2 = os.listdir(r'D:\Data\renji0723\ProcessedData\NPY\ThreeD')
    # list2 = [case[: case.index('.npy')] for case in list2]
    # print([case for case in list1 if case not in list2])
    SameData()


