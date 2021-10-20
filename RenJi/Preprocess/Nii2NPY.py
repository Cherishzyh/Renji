import os
import numpy as np
import shutil
import SimpleITK as sitk
import matplotlib.pyplot as plt
from copy import deepcopy

import pandas as pd
from MeDIT.Visualization import FlattenImages, Imshow3DArray
from MeDIT.Normalize import Normalize01, NormalizeZ
from MeDIT.Augment import *

from scipy import stats

############################################## Tools  ##################################################################


def ShowFlatten(data, save_path=r''):
    data1_flatten = FlattenImages(data, is_show=False)
    plt.figure(figsize=(8, 8), dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(data1_flatten, cmap='gray')
    if save_path:
        plt.savefig(save_path, pad_inches=0)
    # else:
    #     plt.show()
    plt.close()
    return data1_flatten


def SaveImage():
    image_folder = r'D:\Data\renji\ResizeNormImage'
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    for case in os.listdir(r'D:\Data\renji\Resize'):
        print(case)
        case_folder = os.path.join(r'D:\Data\renji\Resize', case)
        nii_path = os.path.join(case_folder, 'resize_2CH.nii.gz')
        image = sitk.ReadImage(nii_path)
        data = NormalizeZ(np.squeeze(sitk.GetArrayFromImage(image)))
        ShowFlatten(data, os.path.join(image_folder, '{}.jpg'.format(case)))


def DataClip(data, value=99.9):
    max = np.percentile(data, value)
    data = np.clip(data, a_min=np.min(data), a_max=max)
    return data


def GetCenter(roi):
    roi = np.squeeze(roi)
    if np.ndim(roi) == 3:
        roi = roi[0]
    non_zero = np.nonzero(roi)
    center_x = int(np.median(np.unique(non_zero[1])))
    center_y = int(np.median(np.unique(non_zero[0])))
    return (center_x, center_y)


############################################## Resize ##################################################################


def ResizeSipmleITKImage(image, expected_resolution=None, method=sitk.sitkBSpline, dtype=np.float32, store_path=''):
    if (expected_resolution is None):
        print('Give at least one parameters. ')
        return image

    shape = (image.GetSize()[0], image.GetSize()[1])
    resolution = (image.GetSpacing()[0], image.GetSpacing()[1])

    expected_resolution = list(expected_resolution)
    dim_0, dim_1 = False, False
    if expected_resolution[0] < 1e-6:
        expected_resolution[0] = resolution[0]
        dim_0 = True
    if expected_resolution[1] < 1e-6:
        expected_resolution[1] = resolution[1]
        dim_1 = True

    expected_shape = [int(raw_resolution * raw_size / dest_resolution) for
                      dest_resolution, raw_size, raw_resolution in
                      zip(expected_resolution, shape, resolution)]
    if dim_0: expected_shape[0] = shape[0]
    if dim_1: expected_shape[1] = shape[1]

    data = np.squeeze(sitk.GetArrayFromImage(image))
    if np.ndim(data) == 4:
        data = data[:, 2, ...]
    output_list = []
    for time in range(data.shape[0]):
        data_time = data[time, ...]
        image_time = sitk.GetImageFromArray(data_time)
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing(expected_resolution)
        resample_filter.SetInterpolator(method)
        resample_filter.SetSize(expected_shape)
        image_resize = resample_filter.Execute(image_time)
        data_resize = sitk.GetArrayFromImage(image_resize)
        output_list.append(data_resize)

    output = np.array(output_list, dtype=dtype)
    output = output[:, np.newaxis, ...]
    output = sitk.GetImageFromArray(output)
    output.SetSpacing((expected_resolution[0], expected_resolution[1], image.GetSpacing()[2]))

    if store_path and store_path.endswith(('.nii', '.nii.gz')):
        sitk.WriteImage(output, store_path)
    return output


def TestResize():
    # nii_folder = r'D:\Data\renji\ProcessedData'
    store_folder = r'Z:\RenJi\3CHHeathy20211012\3CH'
    nii_folder = r'Z:\RenJi\3CHHeathy20211012\3CH'

    for index, case in enumerate(os.listdir(nii_folder)):
        # if case in ['20151122 normal test', '20200630 guoshikuan', '20210518 liuxiangyu 0', '20210720 jiangguoxiang']:
        #     continue
        print('**********************{} / {}***********************'.format(index, len(os.listdir(nii_folder))))
        case_folder = os.path.join(nii_folder, case)
        des_folder = os.path.join(store_folder, case)
        if not os.path.exists(des_folder):
            os.mkdir(des_folder)
        # for data in os.listdir(case_folder):
        #     if '2CH' in data:
        #         try:
        #             image = sitk.ReadImage(os.path.join(case_folder, data))
        #             ResizeSipmleITKImage(image, expected_resolution=[1, 1, -1, -1],
        #                                  store_path=os.path.join(des_folder, 'resize_2CH.nii.gz'))
        #         except Exception as e:
        #             print('failed to resize {}'.format(case))
        nii_path = os.path.join(case_folder, list(sorted(os.listdir(case_folder)))[-1])
        try:
            image = sitk.ReadImage(nii_path)
            ResizeSipmleITKImage(image, expected_resolution=[1, 1, -1, -1],
                                 store_path=os.path.join(des_folder, 'resize_2CH.nii.gz'))
        except Exception as e:
            print('failed to resize {}'.format(case))
            print(e)


############################################### Nii to NPY #############################################################


def Nii2Npy():
    from MeDIT.ArrayProcess import ExtractBlock
    shape = [30, 200, 200]
    nii_folder = r'Z:\RenJi\3CHHeathy20211012\3CH'
    npy_folder = r'Z:\RenJi\3CHHeathy20211012\3CHNPY'
    npy_roi_folder = r'Z:\RenJi\3CHHeathy20211012\3CHROINPY'
    image_folder = r'Z:\RenJi\3CHHeathy20211012\3CHImage'
    if not os.path.exists(npy_folder):
        os.mkdir(npy_folder)
    flip_list = os.listdir(r'C:\Users\ZhangYihong\Desktop\RL')
    flip_list = [case[:case.index('.jpg')] for case in flip_list]
    for index, case in enumerate(os.listdir(nii_folder)):
        print('########################## {} / {} ###############################'.
              format(index + 1, len(os.listdir(nii_folder))))
        try:

            if case == '20180109 renjianxin':
                continue
            case_path = os.path.join(nii_folder, case)
            case_name = '{} {}'.format(case.split(' ')[0], case.split(' ')[1])
            image_path = os.path.join(case_path, 'resize_3ch.nii.gz')
            image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))

            if os.path.exists(os.path.join(case_path, 'mask_3ch.nii.gz')):
                mask_path = os.path.join(case_path, 'mask_3ch.nii.gz')
                mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
                if case in flip_list:
                    mask = np.rot90(mask, k=3, axes=(1, 2))
                else:
                    mask = np.flip(mask, axis=1)
                np.save(os.path.join(npy_roi_folder, '{}.npy'.format(case_name)), mask[np.newaxis, ...])
            #     mask = np.flip(mask, axis=1)
            #     center = GetCenter(mask)
            #     center = [-1, center[1], center[0]]
            #     image, _ = ExtractBlock(image, shape, center_point=center)
            #     mask, _ = ExtractBlock(mask, shape, center_point=center)

            ###########################################  2CH  ##########################################################
            # if case in flip_list:
            #     if np.count_nonzero(image[:, :, 0:(image.shape[2] // 4)]) <= np.count_nonzero(image[:, :, (image.shape[2] * 3 // 4):]):
            #         image = np.flip(image, axis=2)
            #         mask = np.flip(mask, axis=2)
            # else:
            #     if np.count_nonzero(image[:, :, 0:(image.shape[2] // 4)]) > np.count_nonzero(image[:, :, (image.shape[2] * 3 // 4):]):
            #         image = np.flip(image, axis=2)
            #         mask = np.flip(mask, axis=2)

                # np.save(os.path.join(npy_roi_folder, '{}.npy'.format(case)), mask[np.newaxis, ...])
            ###########################################  3CH  ##########################################################

            ############################################################################################################
            raw_data = ShowFlatten(image)
            plt.figure(figsize=(16, 17), dpi=100)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.subplot(223)
            plt.title(np.sum(Normalize01(image)))
            plt.axis('off')
            plt.hist(Normalize01(image).flatten(), bins=100)

            ###################################################################################
            if case in flip_list:
                image = np.rot90(image, k=3, axes=(1, 2))
            else:
                image = np.flip(image, axis=1)
            # image_list = []
            # for index in range(30):
            #     image_list.append(Rotate(image[index], {'theta': -45}))
            # image = np.array(image_list)

            value = 99.995
            if np.sum(Normalize01(image)) < 250000:
                while True:
                    image = DataClip(image, value=value)
                    if np.sum(Normalize01(image)) >= 250000:
                        print('clip {} by {}'.format(case, value))
                        break
                    else:
                        value -= 0.005

            ###################################################################################
            flip_data = ShowFlatten(image)
            plt.subplot(221)
            plt.axis('off')
            plt.imshow(raw_data, cmap='gray')
            plt.subplot(222)
            plt.axis('off')
            plt.imshow(flip_data, cmap='gray')
            plt.subplot(224)
            plt.axis('off')
            plt.hist(Normalize01(image).flatten(), bins=100)
            plt.title(np.sum(Normalize01(image)))
            plt.savefig(os.path.join(image_folder, '{}.jpg'.format(case)), pad_inches=0)
            # plt.show()
            plt.close()
            ###################################################################################
            np.save(os.path.join(npy_folder, '{}.npy'.format(case_name)), NormalizeZ(image[np.newaxis, ...]))
        except Exception as e:
            print(e)



def AddNewAxis():
    npy_folder = r'D:\Data\renji\Npy'
    for case in os.listdir(npy_folder):
        npy_path = os.path.join(npy_folder, case)
        data = np.load(npy_path)
        data = data[np.newaxis, ...]
        np.save(npy_path, data)


def DropNewAxis():
    npy_folder = r'D:\Data\renji\Npy'
    npy_2d_folder = r'D:\Data\renji\Npy2D'
    if not os.path.exists(npy_2d_folder):
        os.mkdir(npy_2d_folder)
    for case in os.listdir(npy_folder):
        npy_path = os.path.join(npy_folder, case)
        data = np.load(npy_path)
        data = np.squeeze(data)
        np.save(os.path.join(npy_2d_folder, case), data)

###############################################    Check   #############################################################


def CheckNo2CHData():
    folder = r'C:\Users\82375\Desktop\NO2CH'
    image_folder = r'C:\Users\82375\Desktop\image'
    for index, case in enumerate(os.listdir(folder)):
        case_folder = os.path.join(folder, case)
        file_list = list(sorted(os.listdir(case_folder)))
        if len([file for file in file_list if '2CH' in file]) == 0:
            try:
                nii_path = os.path.join(case_folder, file_list[-1])
                image = sitk.ReadImage(nii_path)
                data = np.squeeze(sitk.GetArrayFromImage(image))
                flatten_data = FlattenImages(data)
                plt.figure(figsize=(8, 8))
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.imshow(flatten_data, cmap='gray')
                plt.savefig(os.path.join(image_folder, '{}.jpg'.format(case)), pad_inches=0)
            except Exception as e:
                print(case)
                print(e)

        else:
            print(case)


def CheckNPY():
    npy_folder = r'D:\Data\renji\NPY'
    image_folder = r'D:\Data\renji\VaryImage'
    for index, case in enumerate(os.listdir(npy_folder)):
        print('########################## {} / {} ###############################'.format(index, len(os.listdir(npy_folder))))
        npy_path = os.path.join(npy_folder, case)
        data = np.load(npy_path)
        data = data - data[0]
        data_flatten = FlattenImages(data)
        plt.figure(figsize=(8, 8))
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(Normalize01(data_flatten), cmap='gray')
        plt.savefig(os.path.join(image_folder, '{}.jpg'.format(case[:case.index('.npy')])), pad_inches=0)
        plt.close()

        # plt.hist(Normalize01(data).flatten(), bins=100)
        # plt.title(np.sum(Normalize01(data)))
        # plt.savefig(os.path.join(image_folder, '{}_hist.jpg'.format(case[:case.index('.npy')])), pad_inches=0)
        # plt.close()


def SelectSlice():
    npy_2d = r'/home/zhangyihong/Documents/RenJi/Npy2D'
    save_folder = r'/home/zhangyihong/Documents/RenJi/Npy6Slice'
    for case in os.listdir(npy_2d):
        case_path = os.path.join(npy_2d, case)
        array = np.load(case_path)
        new_array = np.concatenate([array[14:17], array[-3:, ...]], axis=0)
        np.save(os.path.join(save_folder, case), new_array)


###############################################    New Preprocess   #############################################################
def CheckROI():
    folder = r'D:\BaiduNetdiskDownload\LVA finished 2CH_MASK 20210910\LVA finished 2CH_MASK'
    # list = ['20160125 wuguobing', '20160920 wangzhirong',  '20170214 caihanliang',  '20170412 houwenxiang',
    #         '20170825 qianliyan 2',  '20180612 chenzhenhua',  '20180804 GaoLingChen',  '20181030 jianggenju',
    #         '20190104 daizifeng',  '20190314 shifuliang',   '20190514 gujingen',
    #         '20191226 chencaichang',  '20200123 chencaichang',  '20200430 liyaolu', '20200818 mengqiaoyun']
    # list = ['20190502 sudaming']
    list = os.listdir(folder)
    for index, case in enumerate(sorted(list)):
        print('{}\t /\t {}, {}'.format(index, len(list), case))
        try:
            case_folder = os.path.join(folder, case)
            case_list = os.listdir(case_folder)
            mask_name = [name for name in case_list if 'MASK' in name][0]
            ch2_name = [name for name in case_list if name not in mask_name][0]
            mask_path = os.path.join(case_folder, mask_name)
            ch2_path = os.path.join(case_folder, ch2_name)

            ch2_image = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(ch2_path)))
            mask_image = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))

            ch_mask = Normalize01(ch2_image) * mask_image

            plt.imshow(ch_mask[0], cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(r'D:\BaiduNetdiskDownload\LVA finished 2CH_MASK 20210910\Image', '{}.jpg'.format(case)))
            plt.close()
        except Exception as e:
            print(e)
# CheckROI()


def CheckErrorCase():
    folder = r'D:\BaiduNetdiskDownload\LVA finished 2CH_MASK 20210910\LVA finished 2CH_MASK'
    # '20160125 wuguobing','20190502 sudaming','20170214 caihanliang', '20160920 wangzhirong',
    list = [ '20170412 houwenxiang',
            '20170825 qianliyan 2',  '20180612 chenzhenhua',  '20180804 GaoLingChen',  '20181030 jianggenju',
            '20190104 daizifeng',  '20190314 shifuliang',   '20190514 gujingen',
            '20191226 chencaichang',  '20200123 chencaichang',  '20200430 liyaolu', '20200818 mengqiaoyun']
    for index, case in enumerate(list):
        case_folder = os.path.join(folder, case)
        case_list = os.listdir(case_folder)
        # ch2_name = [name for name in case_list if '2CH' not in name][0]
        os.rename(os.path.join(case_folder, '2CH_MASK.nii'), os.path.join(case_folder, '2CH_MASK.nii.gz'))
        print(case_list)
# CheckErrorCase()


def Resampler(image, is_roi=False, expected_resolution=None, expected_shape=None, store_path=''):
    if (expected_resolution is None) and (expected_shape is None):
        print('Give at least one parameters. ')
        return image

    if is_roi:
        shape = (image.GetSize()[0], image.GetSize()[1], image.GetSize()[2])
        resolution = (image.GetSpacing()[0], image.GetSpacing()[1], image.GetSpacing()[2])
    else:
        shape = (image.GetSize()[0], image.GetSize()[1], image.GetSize()[-1])
        resolution = (image.GetSpacing()[0], image.GetSpacing()[1], image.GetSpacing()[-1])

    expected_resolution = list(expected_resolution)
    dim_0, dim_1, dim_2 = False, False, True
    if expected_resolution[0] < 1e-6:
        expected_resolution[0] = resolution[0]
        dim_0 = True
    if expected_resolution[1] < 1e-6:
        expected_resolution[1] = resolution[1]
        dim_1 = True
    if expected_resolution[2] < 1e-6:
        expected_resolution[2] = resolution[-1]
    expected_shape = [int(raw_resolution * raw_size / dest_resolution) for
                      dest_resolution, raw_size, raw_resolution in
                      zip(expected_resolution, shape, resolution)]
    if dim_0: expected_shape[0] = shape[0]
    if dim_1: expected_shape[1] = shape[1]
    if dim_2: expected_shape[2] = shape[2]
    expected_shape = tuple(expected_shape)

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetOutputSpacing(expected_resolution)
    resample_filter.SetSize(expected_shape)
    resample_filter.SetOutputPixelType(sitk.sitkFloat32)
    resample_filter.SetTransform(sitk.AffineTransform(len(shape)))
    if is_roi:
        resample_filter.SetInterpolator(sitk.sitkLinear)
    else:
        resample_filter.SetInterpolator(sitk.sitkBSpline)

    data = np.squeeze(sitk.GetArrayFromImage(image))
    if np.ndim(data) == 4:
        data = data[:, 2, ...]
    assert np.ndim(data) == 3
    image_time = sitk.GetImageFromArray(data)
    image_resize = resample_filter.Execute(image_time)
    data_resize = sitk.GetArrayFromImage(image_resize)

    if is_roi:
        new_data = np.zeros(data_resize.shape, dtype=np.uint8)
        pixels = np.unique(data_resize)
        for i in range(len(pixels)):
            if i == (len(pixels) - 1):
                max = pixels[i]
                min = (pixels[i - 1] + pixels[i]) / 2
            elif i == 0:
                max = (pixels[i] + pixels[i + 1]) / 2
                min = pixels[i]
            else:
                max = (pixels[i] + pixels[i + 1]) / 2
                min = (pixels[i - 1] + pixels[i]) / 2
            new_data[np.bitwise_and(data_resize > min, data_resize <= max)] = pixels[i]
        output = sitk.GetImageFromArray(new_data)
        output.CopyInformation(image_resize)
    else:
        output = image_resize

    if store_path and store_path.endswith(('.nii', '.nii.gz')):
        sitk.WriteImage(output, store_path)
    return output


def TestResampler():
    folder = r'Z:\RenJi\RawDataUseful\Combine'
    for index, case in enumerate(sorted(os.listdir(folder))):
        try:
            case_folder = os.path.join(folder, case)
            name_list = [name for name in os.listdir(case_folder) if '3CH' in name]
            name_list = [name for name in name_list if name != '3CH_MASK.nii.gz']
            image_path = os.path.join(case_folder, name_list[0])
            image = sitk.ReadImage(image_path)
            output_img = Resampler(image, is_roi=False, expected_resolution=[1.0416666269302368, 1.0416666269302368, 1],
                                   store_path=os.path.join(case_folder, 'resize_3ch.nii.gz'))
            mask_path = os.path.join(case_folder, '3CH_MASK.nii.gz')
            if os.path.exists(mask_path):
                mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
                if np.ndim(np.squeeze(mask)) == 3:
                    mask = np.sum(mask, axis=0)
                    mask = mask[np.newaxis, ...]
                mask = np.expand_dims(mask[0], 0).repeat(30, axis=0)
                mask = sitk.GetImageFromArray(mask)
                mask.SetSpacing((image.GetSpacing()[0], image.GetSpacing()[1], 1.))
                if mask.GetSpacing() == (1.0416666269302368, 1.0416666269302368, 1):
                    mask.CopyInformation(output_img)
                    output_mask = mask
                    sitk.WriteImage(mask, os.path.join(case_folder, 'mask_3ch.nii.gz'))
                else:
                    output_mask = Resampler(mask, is_roi=True, expected_resolution=[1.0416666269302368, 1.0416666269302368, 1],
                                            store_path=os.path.join(case_folder, 'mask_3ch.nii.gz'))
            print('--------{} / {}--{} --------'.format(index+1, len(os.listdir(folder)), case))
        except Exception as e:
            print('fail to resample case: {}'.format(case))
            print('error: {}'.format(e))


def Concat():
    case_folder = r'Z:\RenJi\3CHHeathy20211012\CaseDropped\20151121 normal 2'
    image_list = []
    resolution = (1.09375, 1.09375, 1)
    for case in sorted(os.listdir(case_folder)):
        if case == '2CH_MASK.nii.gz':
            continue
        else:
            image_path = os.path.join(case_folder, case)
            image = sitk.ReadImage(image_path)
            image_list.append(NormalizeZ(sitk.GetArrayFromImage(image)))
    image_array = np.squeeze(np.asarray(image_list))
    new_image = sitk.GetImageFromArray(image_array)
    new_image.SetSpacing(resolution)
    sitk.WriteImage(new_image, r'Z:\RenJi\3CHHeathy20211012\CaseDropped\20151121 normal 2\3Ch.nii.gz')


if __name__ == '__main__':
    # SaveImage()
    # TestResampler()
    Nii2Npy()
    # CheckNo2CHData()
    # CheckNPY()
    # AddNewAxis()
    # DropNewAxis()
    # SelectSlice()
    # Concat()


    # case_list = os.listdir(r'Z:\RenJi\LVA finished 2CH_MASK 20210910\NPY')
    # case_list = [case[: case.index('.npy')] for case in case_list]
    # case_list = ['{} {}'.format(case.split(' ')[0], case.split(' ')[1]) for case in case_list]
    #
    # df1 = pd.read_csv(r'Z:\RenJi\test_name.csv', index_col='CaseName').index.tolist()
    # df2 = pd.read_csv(r'Z:\RenJi\train_name.csv', index_col='CaseName').index.tolist()
    # df3 = pd.read_csv(r'Z:\RenJi\val_name.csv', index_col='CaseName').index.tolist()
    # df1.extend(df2)
    # df1.extend(df3)
    # print([case for case in case_list if case not in df1])
    # print([case for case in df1 if case not in case_list])

    # label_path = r'Z:\RenJi\label_2cl.csv'
    # label_df = pd.read_csv(label_path, index_col='CaseName')
    # label = []
    # for cv_index in range(1, 6):
    #     cv_path = r'Z:\RenJi\non_train-cv{}.csv'.format(cv_index)
    #     cv_list = pd.read_csv(cv_path, index_col='CaseName').index.to_list()
    #
    #     label_0 = []
    #     label_1 = []
    #     label_2 = []
    #     label_3 = []
    #     for case in cv_list:
    #         if label_df.loc[case].item() == 0:
    #             label_0.append(case)
    #         elif label_df.loc[case].item() == 1:
    #             label_1.append(case)
    #         elif label_df.loc[case].item() == 2:
    #             label_2.append(case)
    #         elif label_df.loc[case].item() == 3:
    #             label_3.append(case)
    #     print('cv_{}:\n label 0: {}\tlabel 1: {}\tlabel 2: {}\tlabel 3: {}\t'.format(cv_index, len(label_0), len(label_1),
    #                                                                                  len(label_2), len(label_3)))
    #     label.append([len(label_0), len(label_1), len(label_2), len(label_3)])
    # label = np.array(label)
    # print(stats.chisquare(f_obs=label[0],  f_exp=np.sum(label[1:], axis=0)))
    # print(stats.chisquare(f_obs=label[1],  f_exp=np.sum(label, axis=0)-label[1]))
    # print(stats.chisquare(f_obs=label[2],  f_exp=np.sum(label, axis=0)-label[2]))
    # print(stats.chisquare(f_obs=label[3],  f_exp=np.sum(label, axis=0)-label[3]))
    # print(stats.chisquare(f_obs=label[4],  f_exp=np.sum(label[:4], axis=0)))
    #
    #
    #
    # test_path = r'Z:\RenJi\non_test_name.csv'
    # cv_df = pd.read_csv(test_path, index_col='CaseName')
    # label_0, label_1, label_2, label_3 = 0, 0, 0, 0
    # for case in cv_df.index:
    #     if label_df.loc[case].item() == 0:
    #         label_0 += 1
    #     elif label_df.loc[case].item() == 1:
    #         label_1 += 1
    #     elif label_df.loc[case].item() == 2:
    #         label_2 += 1
    #     elif label_df.loc[case].item() == 3:
    #         label_3 += 1
    # print('label 0: {}\tlabel 1: {}\tlabel 2: {}\tlabel 3: {}\t'.format(label_0, label_1, label_2, label_3))

    # case = '20140923 suyongming'
    # case_path = os.path.join(r'Z:\RenJi\RawDataUseful\3CH', case)
    # image_path = os.path.join(case_path, 'resize_3ch.nii.gz')
    # image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    # # image = np.rot90(image, k=2, axes=(1, 2))
    # image = np.flip(image, axis=1)
    # # image = np.flip(image, axis=2)
    #
    # raw_data = ShowFlatten(image)
    # plt.figure(figsize=(16, 17), dpi=100)
    # plt.title('rotate90-flip1-flip2')
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.imshow(raw_data, cmap='gray')
    # plt.show()





