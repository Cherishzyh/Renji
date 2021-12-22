import os
import shutil

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from MeDIT.ArrayProcess import ExtractBlock

# data_folder = r'Z:\RenJi\ExternalTest\CH3'
# for case in os.listdir(data_folder):
#     image = sitk.ReadImage(os.path.join(data_folder, case))
#     output = Resampler(image, is_roi=False, expected_resolution=[1.0416666269302368, 1.0416666269302368, 1],
#                        store_path=os.path.join(r'Z:\RenJi\ExternalTest\ExternalNii', '{}/resize_3ch.nii.gz'.format(case.split('.nii')[0])))
#     print(case)


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
    # a = ['3CHExternal', '2CHExternal']
    data_folder = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}ROIPred'.format('2CHExternal')
    i = 0
    for case in os.listdir(data_folder):
        data_3ch = np.load(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}ROIPred'.format('3CHExternal'), case))
        data_2ch = np.load(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}ROIPred'.format('2CHExternal'), case))
        if data_2ch.shape != (30, 180, 180) or data_3ch.shape != (30, 180, 180):
            print(case, data_2ch.shape, data_3ch.shape)
            i += 1
    print(i)
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
# CheckData()


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


def CropDatabySlice():
    # crop---5slice/5slice_dilate/allslice/allslice_dilate
    from scipy.ndimage import binary_dilation
    from MeDIT.ArrayProcess import ExtractBlock
    # a = ['3CH', '2CH', '2CHExternal', '3CHExternal']
    a = ['2CHExternal', '3CHExternal']
    for key in a:
        print(key)
        raw_folder = r'/home/zhangyihong/Documents/RenJi/Data/{}'.format(key)

        data_folder_slice = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}Pred_SliceBySeg_Revision'.format(key)
        roi_folder_slice_dilated = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}ROIPred_SliceBySeg_Revision'.format(key)
        # image_folder = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}Image_5Slice_Revision'.format(key)
        if not os.path.exists(data_folder_slice): os.mkdir(data_folder_slice)
        if not os.path.exists(roi_folder_slice_dilated): os.mkdir(roi_folder_slice_dilated)
        # if not os.path.exists(image_folder): os.mkdir(image_folder)

        for index, case in enumerate(os.listdir(raw_folder)):
            print('    {}, {}/{}'.format(case, index+1, len(os.listdir(raw_folder))))
            data_path = os.path.join(raw_folder, '{}/Image.nii.gz'.format(case))
            roi_path = os.path.join(raw_folder, '{}/Mask.nii.gz'.format(case))
            # 9：12收缩，-2舒张
            data = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
            roi = sitk.GetArrayFromImage(sitk.ReadImage(roi_path))
            sum_array = np.sum(roi, axis=(1, 2)).argsort()
            slice = np.concatenate([np.sort(sum_array[:3]), np.sort(sum_array[-2:])], axis=0)
            # slice = [9, 10, 11, -2, -1]
            data_slice = data[slice]
            roi_slice = roi[slice]
            try:
                center = GetCenter(roi[0])
            except Exception:
                center = [-1, -1]

            data_crop_slice, _ = ExtractBlock(data_slice, (data_slice.shape[0], 180, 180), [-1, center[0], center[1]])
            roi_crop_slice, _ = ExtractBlock(roi_slice, (data_slice.shape[0], 180, 180), [-1, center[0], center[1]])

            roi_dilate_slice = binary_dilation(roi_crop_slice, structure=np.ones((1, 11, 11)))

            np.save(os.path.join(data_folder_slice, '{}.npy'.format(case)), data_crop_slice)
            np.save(os.path.join(roi_folder_slice_dilated, '{}.npy'.format(case)), roi_dilate_slice)

            # plt.figure(figsize=(8, 8), dpi=100)
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            # plt.margins(0, 0)
            # plt.imshow(data_crop_slice[0], cmap='gray', vmin=0.)
            # plt.contour(roi_dilate_slice[0], colors='y')
            # plt.savefig(os.path.join(image_folder, '{}.jpg'.format(case.split('.npy')[0])), pad_inches=0)
            # plt.close()
# CropDatabySlice()

# data_root = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/non_alltrain_name.csv'
# csv_path = os.path.join(data_root, 'non_alltrain_name.csv')


def NormalizeBatch(inputs, roi=None, dim=2):
    assert dim == 2 or dim == 3
    if dim == 2:
        assert len(inputs.shape) == 4
        if isinstance(roi, torch.Tensor):
            mean = torch.mean(inputs*roi, dim=(1, 2, 3))
            std = torch.std(inputs*roi, dim=(1, 2, 3))
        else:
            mean = torch.mean(inputs, dim=(1, 2, 3))
            std = torch.std(inputs, dim=(1, 2, 3))
        mean_expand = mean.expand((inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[0]))
        mean_expand = mean_expand.permute((-1, 0, 1, 2))
        std_expand = std.expand((inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[0]))
        std_expand = std_expand.permute((-1, 0, 1, 2))

        normal_data = inputs - mean_expand
        if (std < 1e-6).any():
            print('Check Normalization')
            return normal_data
        else:
            return normal_data / std_expand
    else:
        assert len(inputs.shape) == 5
        if isinstance(roi, torch.Tensor):
            mean = torch.mean(inputs*roi, dim=(1, 2, 3, 4))
            std = torch.std(inputs*roi, dim=(1, 2, 3, 4))
        else:
            mean = torch.mean(inputs, dim=(1, 2, 3, 4))
            std = torch.std(inputs, dim=(1, 2, 3, 4))
        mean_expand = mean.expand((inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4], inputs.shape[0]))
        mean_expand = mean_expand.permute((-1, 0, 1, 2, 3))
        std_expand = std.expand((inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4], inputs.shape[0]))
        std_expand = std_expand.permute((-1, 0, 1, 2, 3))

        normal_data = inputs - mean_expand
        if (std < 1e-6).any():
            print('Check Normalization')
            return normal_data
        else:
            return normal_data / std_expand


# df = pd.read_csv(r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/label_2cl.csv', index_col='CaseName')
# case_list = pd.read_csv(r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/non_test_name.csv', index_col='CaseName').index.tolist()
# label_list = []
# for case in sorted(case_list):
#     label_list.append(df.loc[case, 'Label'])
# new_df = pd.DataFrame({'CaseName': sorted(case_list), 'Label': label_list})
# new_df.to_csv(r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/test_name.csv', index=False)


# for case in os.listdir(r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/2CHPred_SliceBySeg_1207'):
#     if case not in os.listdir(r'/home/zhangyihong/Documents/RenJi/Data/CenterCrop_1/Ch2_Seg'):
#         print(case)
#         continue
#     data_before = np.load(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/2CHPred_SliceBySeg_1207', case))
#     data_before, _ = ExtractBlock(data_before, (5, 150, 150))
#     data_after = np.load(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/CenterCrop_1/Ch2_Seg', case))
#     plt.imshow((data_after-data_before)[0], cmap='gray')
#     plt.show()
#     plt.subplot(121)
#     plt.imshow(data_before[0], cmap='gray')
#     plt.subplot(122)
#     plt.imshow(data_after[0], cmap='gray')
#     plt.show()
#     print((data_after == data_before).all())

def ShowCM(cm, save_folder=r''):
    import seaborn as sns
    sns.set()
    f, ax = plt.subplots(figsize=(8, 8))

    sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', cbar=False)  # 画热力图

    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    if save_folder:
        plt.savefig(save_folder, dpi=500, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    plt.close()

def Result4NPY(label_list, pred_list, save_folder=r''):
    from sklearn import metrics
    from MeDIT.Statistics import BinaryClassification

    assert isinstance(label_list, list)
    assert isinstance(pred_list, list)

    bc = BinaryClassification(is_show=False)
    bc.Run(pred_list, label_list)
    fpr, tpr, threshold = metrics.roc_curve(label_list, pred_list)
    binary_pred = np.array(pred_list)
    index = np.argmax(1 - fpr + tpr)
    binary_pred[binary_pred >= threshold[index]] = 1
    binary_pred[binary_pred < threshold[index]] = 0
    cm = metrics.confusion_matrix(label_list, binary_pred.tolist())
    ShowCM(cm, save_folder=save_folder)

def DrawROC(csv_folder, save_folder=r''):
    from sklearn import metrics

    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.grid(False)

    train_df = pd.read_csv(os.path.join(csv_folder, '{}.csv'.format('alltrain')), index_col='CaseName')
    train_label = train_df.loc[:, 'Label'].tolist()
    train_pred = train_df.loc[:, 'Pred'].tolist()
    fpn, sen, the = metrics.roc_curve(train_label, train_pred)
    auc = metrics.roc_auc_score(train_label, train_pred)
    plt.plot(fpn, sen, label='Training: {:.3f}'.format(auc))

    test_df = pd.read_csv(os.path.join(csv_folder, '{}.csv'.format('test')), index_col='CaseName')
    test_label = test_df.loc[:, 'Label'].tolist()
    test_pred = test_df.loc[:, 'Pred'].tolist()
    fpn, sen, the = metrics.roc_curve(test_label, test_pred)
    auc = metrics.roc_auc_score(test_label, test_pred)
    plt.plot(fpn, sen, label='Testing:  {:.3f}'.format(auc))

    if os.path.exists(os.path.join(csv_folder, '{}.csv'.format('external'))):
        external_df = pd.read_csv(os.path.join(csv_folder, '{}.csv'.format('external')), index_col='CaseName')
        external_label = external_df.loc[:, 'Label'].tolist()
        external_pred = external_df.loc[:, 'Pred'].tolist()
        fpn, sen, the = metrics.roc_curve(external_label, external_pred)
        auc = metrics.roc_auc_score(external_label, external_pred)
        plt.plot(fpn, sen, label='External: {:.3f}'.format(auc))

    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    if save_folder:
        plt.savefig(os.path.join(save_folder, 'ROC.jpg'), dpi=500, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    plt.close()

# for model_name in os.listdir(r'C:\Users\ZhangYihong\Desktop\Result'):
#     from MeDIT.Statistics import BinaryClassification
#     if model_name == 'PredData':
#         continue
#     df_train = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\Result\{}\alltrain.csv'.format(model_name), index_col='CaseName')
#     df_test = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\Result\{}\test.csv'.format(model_name), index_col='CaseName')
#     df_external = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\Result\{}\external.csv'.format(model_name), index_col='CaseName')
#     # bc = BinaryClassification()
#     # bc.Run(df_train.loc[:, 'Pred'].tolist(), df_train.loc[:, 'Label'].tolist())
#     # bc.Run(df_test.loc[:, 'Pred'].tolist(), df_test.loc[:, 'Label'].tolist())
#     # bc.Run(df_external.loc[:, 'Pred'].tolist(), df_external.loc[:, 'Label'].tolist())
#     Result4NPY(df_train.loc[:, 'Label'].tolist(), df_train.loc[:, 'Pred'].tolist(), save_folder=r'C:\Users\ZhangYihong\Desktop\Result\{}\alltrain.jpg'.format(model_name))
#     Result4NPY(df_test.loc[:, 'Label'].tolist(), df_test.loc[:, 'Pred'].tolist(), save_folder=r'C:\Users\ZhangYihong\Desktop\Result\{}\test.jpg'.format(model_name))
#     Result4NPY(df_external.loc[:, 'Label'].tolist(), df_external.loc[:, 'Pred'].tolist(), save_folder=r'C:\Users\ZhangYihong\Desktop\Result\{}\external.jpg'.format(model_name))
#     DrawROC(r'C:\Users\ZhangYihong\Desktop\Result\{}'.format(model_name), save_folder=r'C:\Users\ZhangYihong\Desktop\Result\{}'.format(model_name))

# from MeDIT.Visualization import FlattenImages
# from MeDIT.SaveAndLoad import LoadImage
# for Ctype in ['2CHExternal', '3CHExternal']:
#     folder = os.path.join(r'/home/zhangyihong/Documents/RenJi/Data', Ctype)
#     for case in os.listdir(folder):
#         _, data, _ = LoadImage(os.path.join(folder, '{}/Image.nii.gz'.format(case)))
#         _, pred, _ = LoadImage(os.path.join(folder, '{}/Mask.nii.gz'.format(case)))
#         data_flatten = FlattenImages(data.transpose(2, 0, 1), is_show=False)
#         pred_flatten = FlattenImages(pred.transpose(2, 0, 1), is_show=False)
#         plt.figure(figsize=(8, 8), dpi=100)
#         plt.gca().xaxis.set_major_locator(plt.NullLocator())
#         plt.gca().yaxis.set_major_locator(plt.NullLocator())
#         plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#         plt.margins(0, 0)
#         plt.imshow(data_flatten, cmap='gray', vmin=0.)
#         plt.contour(pred_flatten, colors='y')
#         plt.savefig(os.path.join(r'/home/zhangyihong/Documents/RenJi/Data', '{}Image/{}.jpg'.format(Ctype, case)), pad_inches=0)
#         plt.close()
res_list = []
for case in os.listdir(r'Y:\RenJi\2CH 20210910\2CH_MASK Data'):
    case_folder = os.path.join(r'Y:\RenJi\2CH 20210910\2CH_MASK Data', case)
    data_list = os.listdir(case_folder)
    data = [data for data in data_list if '.nii.gz' not in data]
    if len(data) == 1:
        nii_path = os.path.join(case_folder, data[0])
        image = sitk.ReadImage(nii_path)
        res_list.append((image.GetSpacing()[0], image.GetSpacing()[1]))
print(set(res_list))