import os

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import pandas as pd
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader

from BasicTool.MeDIT.Others import IterateCase
from CnnTools.T4T.Utility.Data import *
from RenJi.Network3D.ResNet3D import GenerateModel
# from MyDataLoader.Data import *

from RenJi.Metric.ConfusionMatrix import F1Score


def _GetLoader(sub_list, data_root, input_shape, batch_size):
    data = DataManager(sub_list=sub_list)

    data.AddOne(Image2D(data_root + '/Npy2D',  shape=input_shape))
    data.AddOne(Label(data_root + '/label.csv'), is_input=False)

    loader = DataLoader(data, batch_size=batch_size)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def _GetSubList(label_root, data_type):
    label_path = os.path.join(label_root, '{}_name.csv'.format(data_type))
    case_df = pd.read_csv(label_path, index_col='CaseName')
    case_list = case_df.index.tolist()
    return case_list


def _GetLabel(label_array):
    label = []
    for index in range(label_array.shape[1]):
        label.append(np.argmax(np.bincount(label_array[:, index])))
    return np.array(label)


def ModelEnhanced(device, model_name, weights_list=None, is_test=False):
    device = device
    input_shape = (200, 200)
    batch_size = 24

    model_folder = os.path.join(model_root, model_name)

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []

    for cv_index, cv_folder in enumerate(cv_folder_list):
        cv_index = 0
        cv_folder = cv_folder_list[cv_index]
        model = GenerateModel(50, n_input_channels=1, n_classes=4).to(device)
        if weights_list is None:
            one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if
                                     one.is_file()]
            if len(one_fold_weights_list) == 0:
                continue
            one_fold_weights_list = sorted(one_fold_weights_list, key=lambda x: os.path.getctime(str(x)))
            weights_path = one_fold_weights_list[-1]
        else:
            weights_path = weights_list[cv_index]

        print(weights_path.name)
        model.load_state_dict(torch.load(str(weights_path), map_location='cuda:0'))

        pred_list, label_list = [], []
        model.eval()
        sub_list = _GetSubList(data_root, is_test=is_test)
        data_loader, batches = _GetLoader(sub_list, data_root, input_shape, batch_size)
        with torch.no_grad():
            for inputs, outputs in data_loader:
                inputs = inputs[:, np.newaxis, ...]
                inputs = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(outputs, device)
                preds = model(inputs)

                pred_list.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().data.numpy().tolist())
                label_list.extend(outputs.cpu().data.numpy().astype(int).squeeze().tolist())

        cv_pred_list.append(pred_list)
        cv_label_list.append(label_list)

        del model, weights_path
        break

    cv_pred = np.array(cv_pred_list)
    cv_label = np.array(cv_label_list)
    mean_pred = _GetLabel(cv_pred)
    mean_label = np.mean(cv_label, axis=0)

    precision, recall, f1_score, cm = F1Score(mean_label, mean_pred)
    print(precision)
    print(recall)
    print(f1_score)
    print(cm)

    return mean_pred, mean_label


def ModelInference(device, model_name, data_type='test', n_classes=4, weights_list=None):
    device = device
    input_shape = (200, 200)
    batch_size = 24

    model_folder = os.path.join(model_root, model_name)

    model = GenerateModel(50, n_input_channels=1, n_classes=n_classes).to(device)
    if weights_list is None:
        weights_list = [one for one in IterateCase(model_folder, only_folder=False, verbose=0) if one.is_file()]
        if len(weights_list) == 0:
            raise Exception
        weights_list = sorted(weights_list, key=lambda x: os.path.getctime(str(x)))
        weights_path = weights_list[-1]
    else:
        weights_path = weights_list

    print(weights_path.name)
    model.load_state_dict(torch.load(str(weights_path), map_location='cuda:0'))

    pred_list, label_list = [], []
    model.eval()
    sub_list = pd.read_csv(os.path.join(data_root, '{}_name.csv'.format(data_type)), index_col='CaseName')
    # sub_list = [case for case in sub_list.index if sub_list.loc[case, 'Label'] < 3]
    sub_list = sub_list.index.tolist()

    data_loader, batches = _GetLoader(sub_list, data_root, input_shape, batch_size)
    with torch.no_grad():
        for inputs, outputs in data_loader:
            #######################################
            outputs[outputs < 3] = 1
            outputs[outputs == 3] = 0
            #######################################

            inputs = inputs[:, np.newaxis, ...]
            inputs = MoveTensorsToDevice(inputs, device)
            outputs = MoveTensorsToDevice(outputs, device)
            preds = model(inputs)

            pred_list.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().data.numpy().tolist())
            label_list.extend(outputs.cpu().data.numpy().astype(int).squeeze().tolist())

    del model, weights_path

    precision, recall, f1_score, cm = F1Score(label_list, pred_list)
    print(precision)
    print(recall)
    print(f1_score)
    print(cm)

    return cm


if __name__ == '__main__':
    from RenJi.Visualization.Show import ShowCM
    model_root = r'/home/zhangyihong/Documents/RenJi/Model'
    data_root = r'/home/zhangyihong/Documents/RenJi'
    # model_root = r'Z:\RenJi\Model2DAug'
    # data_root = r'Z:\RenJi'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # ModelEnhanced(device, 'ResNet3D_0812', weights_list=None, is_test=True)
    cm = ModelInference(device, 'ResNet3D_0812_0_12', data_type='test', n_classes=2, weights_list=None)
    ShowCM(cm)