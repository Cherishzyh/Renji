import os
import numpy as np
import shutil
import SimpleITK as sitk
import matplotlib.pyplot as plt
from copy import deepcopy

import pandas as pd
from MeDIT.Visualization import FlattenImages
from MeDIT.Normalize import Normalize01, NormalizeZ


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
    store_folder = r'D:\Data\renji\ResizeNo2CH'
    nii_folder = r'C:\Users\82375\Desktop\NO2CH'

    for index, case in enumerate(os.listdir(nii_folder)):
        if case in ['20151122 normal test', '20200630 guoshikuan', '20210518 liuxiangyu 0', '20210720 jiangguoxiang']:
            continue
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
    from BasicTool.MeDIT.ArrayProcess import Crop3DArray, ExtractBlock
    shape = [30, 240, 240]
    nii_folder = r'D:\Data\renji\Resize'
    npy_folder = r'D:\Data\renji\Npy'
    image_folder = r'D:\Data\renji\NPYImage'
    # image_folder = r'D:\Data\renji\ResizeNo2CHImage'
    if not os.path.exists(npy_folder):
        os.mkdir(npy_folder)
    clip_case = []
    # TODO: change > to <=
    # case_list = ['20191010 lihuaqing', '20191107 xuweiguo', '20200618 yeshengjun', '20200811 yuanshenghe',
    #              '20201130 hufang 3', '20210111 chenlin', '20210322 gaojuwen 3', '20141010 liujinxin',
    #              '20160512 lingzhengshu 0', '20190808 shiweisheng', '20201207 wangjianmin 3', '20210416 jiguifang 1',
    #              '20210423 chenbaolin']
    # TODO: change 99.9 in cilp to 99.5
    # case_list = ['20210524 chengchen 3', '20180804 cuixinyuan', '20180804 WanJiaQi']
    # TODO: move center to [-1, data.shape[1]//2, data.shape[2]//2+25]
    # case_list = ['20191107 xuweiguo', '20160301 zhengsaifeng 2 or 1', '20210315 gushanhua', '20210720 zhugenliang 2']
    # TODO: move center to [-1, data.shape[1]//2, data.shape[2]//2+50]
    # case_list = ['20200922 zhangzhenyu 1', '20210706 shah 1', '20151117 lijian', '20180728 zhaofengyuan']
    # TODO: move center to [-1, data.shape[1]//2, data.shape[2]//2-25]
    # case_list = ['20210629 liuxiangyu 1', '20210702 tangchunsheng 0', '20200618 yeshengjun', '20210111 chenlin',
    #              '20151121 normal 2', '20190523 dongweigen', '20191105 linfeng', '20200924 zengweixiong c 2',
    #              '20191119 huchenlong 1', '20191210 jianggenju', '20200516 mafuyu', '20200519 yinliqun 2',
    #              '20200618 yeshengjun', '20200630 xujiansen', '20200716 zhaozhijian', '20200727 shuzhangyuan',
    #              '20200811 wangfanhua', '20200911 xushuangshuang 3', '20200921 jianglifeng 0', '20201006 chazhixiang',
    #              '20201008 wuguohua', '20201009 chenhuailing 3', '20201015 huanlianbao', '20201016 fengguoyu 3',
    #              '20201026 songtengfei', '20201103 fanhui', '20201105 chenjinping', '20201124 pankuilin',
    #              '20201124 xueyuling 1','20201126 huangyongqiang', '20201207 wangjianmin 3', '20201207 wangxiaoyan 3',
    #              '20210112 xuweiguang', '20210125 zhangchunming', '20210125 zhengxianggao', '20210216 zhonghao',
    #              '20210222 caiguoxing', '20210302 wujun', '20210309 zhangpeisheng 1', '20210319 shah 1',
    #              '20210322 gaojuwen 3', '20210330 zhoujiefang 1', '20210412 lixiaolu 3', '20210423 shengyun',
    #              '20210523 jiangguoxiang', '20210611 gurenhua 1', '20210618 xuzhijie 0', '20210629 zhouguocai 0',
    #              '20210702 tangchunsheng 0', '20210713 wujun', '20210713 xiezhongming', '20210720 zhouabao 0']

    for index, case in enumerate(os.listdir(nii_folder)):
    # for index, case in enumerate(case_list):
        print('########################## {} / {} ###############################'.format(index+1, len(os.listdir(nii_folder))))
        case_path = os.path.join(nii_folder, case)
        nii_path = os.path.join(case_path, 'resize_2CH.nii.gz')
        image = sitk.ReadImage(nii_path)
        data = sitk.GetArrayFromImage(image)
        data = np.squeeze(data)
        data = np.flip(data, axis=1)
        # center = [-1, -1, -1]
        center = [-1, data.shape[1]//2, data.shape[2]//2-50]
        data, _ = ExtractBlock(data, shape, center_point=center)
        raw_data = ShowFlatten(NormalizeZ(data))
        if np.count_nonzero(data[:, :, 0:(data.shape[2] // 4)]) > np.count_nonzero(data[:, :, (data.shape[2]*3 // 4):]):
            data = np.flip(data, axis=2)

        # plt.hist(Normalize01(data).flatten(), bins=100)
        # plt.title(np.sum(Normalize01(data)))
        # plt.savefig(os.path.join(image_folder, '{}_hist.jpg'.format(case)), pad_inches=0)
        # plt.close()
        if (np.sum(Normalize01(data)) - 1.6e5) < 0:
            print('clip {}'.format(case))
            clip_case.append(case)
            data = DataClip(data, value=99.9)

        data = NormalizeZ(data)
        flip_data = ShowFlatten(data)

        plt.figure(figsize=(16, 8), dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.subplot(121)
        plt.imshow(raw_data, cmap='gray')
        plt.subplot(122)
        plt.imshow(flip_data, cmap='gray')
        plt.savefig(os.path.join(image_folder, '{}.jpg'.format(case)), pad_inches=0)
        plt.close()
        case = '{} {}'.format(case.split(' ')[0], case.split(' ')[1])
        np.save(os.path.join(npy_folder, '{}.npy'.format(case)), data[np.newaxis,...])

    if len(clip_case) > 0:
        df = pd.DataFrame({'CaseName': clip_case})
        df.to_csv(r'D:\Data\renji\ClipData_240.csv', index=False, mode='a+', header=False)


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



if __name__ == '__main__':
    # SaveImage()
    # TestResize()
    # Nii2Npy()
    # CheckNo2CHData()
    # CheckNPY()
    # AddNewAxis()
    DropNewAxis()










