import os

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt

from scipy import ndimage
import scipy.signal as signal

from T4T.Utility.Data import *

from RenJi.SuccessfulModel.ConfigInterpretor import BaseImageOutModel


class InferenceByCase(BaseImageOutModel):
    def __init__(self):
        super(InferenceByCase).__init__()

    def __GetCenter(self, mask):
        assert (np.ndim(mask) == 2)
        roi_row = np.sum(mask, axis=1)
        roi_column = np.sum(mask, axis=0)

        row = np.nonzero(roi_row)[0]
        column = np.nonzero(roi_column)[0]

        center = [int(np.mean(row)), int(np.mean(column))]
        return center

    def LoadImage(self, dataset, dimension, slice=[]):
        assert dimension == 2 or dimension == 3
        from copy import deepcopy
        from scipy.ndimage import binary_dilation
        from MeDIT.ArrayProcess import ExtractBlock
        data_path = os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/PredData/{}'.format(dataset), '{}/Image.nii.gz'.format(case))
        roi_path = os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/PredData/{}'.format(dataset), '{}/Mask.nii.gz'.format(case))
        data = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
        roi = sitk.GetArrayFromImage(sitk.ReadImage(roi_path))

        if dimension == 2:
            if len(slice) < 1:
                sum_array = np.sum(roi, axis=(1, 2)).argsort()
                new_slice = np.concatenate([np.sort(sum_array[:3]), np.sort(sum_array[-2:])], axis=0)
                data = data[new_slice]
                roi = roi[new_slice]
            else:
                new_slice = deepcopy(slice)
                new_slice.extend([data.shape[0]-2, data.shape[0]-1])
                data = data[new_slice]
                roi = roi[new_slice]
        try:
            center = self.__GetCenter(roi[0])
        except Exception:
            center = [-1, -1]

        data_crop, _ = ExtractBlock(data, (data.shape[0], 150, 150), [-1, center[0], center[1]])
        roi_crop, _ = ExtractBlock(roi, (data.shape[0], 150, 150), [-1, center[0], center[1]])

        roi_dilate = binary_dilation(roi_crop, structure=np.ones((1, 11, 11)))
        return data_crop, roi_dilate

    def LoadNPY(self, case, dataset, dimension, slice='m'):
        assert dimension == 2 or dimension == 3
        assert slice == 'm' or slice == 'a'
        from MeDIT.ArrayProcess import ExtractBlock
        if dimension == 2:
            if slice == 'm':
                endwith = '_5slice_Revision'
            else:
                # endwith = '_SliceBySeg_1207'
                endwith = '_SliceBySeg_Revision'
            data_path = os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}Pred{}'.format(dataset, endwith), '{}.npy'.format(case))
            roi_path = os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}ROIPred{}'.format(dataset, endwith), '{}.npy'.format(case))
        else:
            data_path = os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}Pred'.format(dataset),
                                     '{}.npy'.format(case))
            roi_path = os.path.join(r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData/{}ROIPred'.format(dataset),
                                    '{}.npy'.format(case))

        data = np.load(data_path)
        roi = np.load(roi_path)

        data_crop, _ = ExtractBlock(data, (data.shape[0], 150, 150))
        roi_crop, _ = ExtractBlock(roi, (data.shape[0], 150, 150))

        return data_crop, roi_crop

    def NormailZ(self, data, roi=None):
        if not isinstance(roi, np.ndarray):
            mean, std = np.mean(data), np.std(data)
            normal_data = data - mean
            if std < 1e-6:
                print('Check Normalization')
                return normal_data
            else:
                return normal_data / std
        else:
            mean_value = np.mean(data[roi == 1])
            std = np.std(data[roi == 1])

            normal_data = data - mean_value
            if std < 1e-6:
                print('Check Normalization')
                return normal_data * roi
            else:
                return (normal_data / std) * roi
    #
    # def LoadModel(self, cv):
    #     self.model.load_state_dict(
    #         torch.load(os.path.join(self.fold_path, 'cv_{}.pt'.format(str(cv))), map_location=self.device))
    #     self.model.eval()
    #     return self.model

    def Run(self, data_2ch, data_3ch, is_mask=True, is_norm=True):
        if data_3ch == []:
            if is_mask:
                data_2ch = data_2ch[0]*data_2ch[1]
            else:
                data_2ch = data_2ch[0]
        else:
            if is_norm:
                if is_mask:
                    data_2ch, data_3ch = self.NormailZ(data_2ch[0], data_2ch[1]), self.NormailZ(data_3ch[0], data_3ch[1])
                else:
                    data_2ch, data_3ch = self.NormailZ(data_2ch[0]), self.NormailZ(data_3ch[0])
            else:
                if is_mask:
                    data_2ch, data_3ch = data_2ch[0]*data_2ch[1], data_3ch[0]*data_3ch[1]
                else:
                    data_2ch, data_3ch = data_2ch[0], data_3ch[0]

        pred_list = []
        with torch.no_grad():
            data_2ch = torch.from_numpy(data_2ch[np.newaxis]).float().to(self.device)
            if isinstance(data_3ch, list):
                pass
            else:
                data_3ch = torch.from_numpy(data_3ch[np.newaxis]).float().to(self.device)
            for cv in range(5):
                self.model.load_state_dict(
                    torch.load(os.path.join(self.fold_path, 'cv_{}.pt'.format(str(cv))), map_location=self.device))
                self.model.eval()
                if isinstance(data_3ch, list):
                    pred_list.append(torch.sigmoid(self.model(data_2ch)).cpu().detach().squeeze())
                else:
                    pred_list.append(torch.sigmoid(self.model(data_2ch, data_3ch)).cpu().detach().squeeze())

        return torch.mean(torch.tensor(pred_list))

    def ShowCM(self, cm, save_folder=r''):
        import seaborn as sns
        sns.set()
        f, ax = plt.subplots(figsize=(8, 8))

        sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', cbar=False)  # 画热力图

        ax.set_title('confusion matrix')  # 标题
        ax.set_xlabel('predict')  # x轴
        ax.set_ylabel('true')  # y轴
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        if save_folder:
            plt.savefig(save_folder, dpi=500)
        else:
            plt.show()
        plt.close()

    def Result4NPY(self, label_list, pred_list, save_folder=r''):
        from sklearn import metrics
        from MeDIT.Statistics import BinaryClassification

        assert isinstance(label_list, list)
        assert isinstance(pred_list, list)

        bc = BinaryClassification()
        bc.Run(pred_list, label_list)
        fpr, tpr, threshold = metrics.roc_curve(label_list, pred_list)
        binary_pred = np.array(pred_list)
        index = np.argmax(1 - fpr + tpr)
        binary_pred[binary_pred >= threshold[index]] = 1
        binary_pred[binary_pred < threshold[index]] = 0
        cm = metrics.confusion_matrix(label_list, binary_pred.tolist())
        self.ShowCM(cm, save_folder=save_folder)

    def DrawROC(self, csv_folder, save_folder=r''):
        from sklearn import metrics

        plt.figure(0, figsize=(6, 5))
        plt.plot([0, 1], [0, 1], 'k--')

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
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        if save_folder:
            plt.savefig(os.path.join(save_folder, 'ROC_revision.jpg'), dpi=500)
        else:
            plt.show()
        plt.close()


if __name__ == '__main__':
    from pathlib import Path

    model_root = r'/home/zhangyihong/Documents/RenJi/SuccessfulModel'
    data_root = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData'
    # 'ResNet_1123', 'ResNet_1123_mask', 'ResNet_1210_SliceBySeg', 'ResNet_1210_mask_SliceBySeg','ResNet_1210_2CH_mask', 'ResNet_1210_3CH_mask'
    for model_name in ['ResNet_1210_SliceBySeg']:
        # model_name = 'ResNet_1123'
        print(model_name)

        model_folder = os.path.join(model_root, model_name)
        # pred_folder = r'/home/zhangyihong/Documents/RenJi/Data/PredData'
        pred_folder = r'/home/zhangyihong/Documents/RenJi/Data'

        classification = InferenceByCase()
        classification.LoadConfigAndModel(model_folder)
        # for cv in range(5):
        #     classification.LoadModel(cv)
        for data_type in ['alltrain', 'test', 'external']:
            if data_type != 'external':
                continue
            df = pd.read_csv(os.path.join(data_root, '{}_name.csv'.format(data_type)), index_col='CaseName')
            sub_list = df.index.tolist()

            case_list, pred_list, label_list = [], [], []
            for case in sorted(sub_list):
                try:
                    if 'SliceBySeg' in model_name:
                        slice = 'a'
                    else:
                        slice = 'm'
                    # slice = 'm'
                    if not data_type == 'external':
                        ch2_data = classification.LoadNPY(case, '2CH', 2, slice=slice)
                        ch3_data = classification.LoadNPY(case, '3CH', 2, slice=slice)

                    else:
                        ch2_data = classification.LoadNPY(case, '2CHExternal', 2, slice=slice)
                        ch3_data = classification.LoadNPY(case, '3CHExternal', 2, slice=slice)
                except Exception as e:
                    continue
                if 'mask' in model_name:
                    is_mask = True
                else:
                    is_mask = False
                if '2CH' in model_name:
                    pred = classification.Run(ch2_data, [], is_mask=is_mask, is_norm=False)
                elif '3CH' in model_name:
                    pred = classification.Run(ch3_data, [], is_mask=is_mask, is_norm=False)
                else:
                    pred = classification.Run(ch2_data, ch3_data, is_mask=is_mask, is_norm=False)
                case_list.append(case)
                pred_list.append(float(pred))
                label_list.append(int(df.loc[case, 'Label']))
                # print('successful predict {}'.format(case))
            new_df = pd.DataFrame({'CaseName': case_list, 'Label': label_list, 'Pred': pred_list})
            new_df.to_csv(os.path.join(model_folder, '{}_revision.csv'.format(data_type)), index=False)
            classification.Result4NPY(label_list, pred_list, save_folder=os.path.join(model_folder, '{}_revision.jpg'.format(data_type)))

        # classification.DrawROC(model_folder, save_folder=model_folder)