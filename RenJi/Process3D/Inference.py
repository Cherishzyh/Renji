import os

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import pandas as pd
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader

from BasicTool.MeDIT.Others import IterateCase

from Network.ResNet3D import GenerateModel
from MyDataLoader.Data import *


def _GetLoader(sub_list, data_root, input_shape, batch_size):
    data = DataManager(sub_list=sub_list)

    data.AddOne(Image3D(data_root + '/Npy', aug_param=None, shape=input_shape))
    data.AddOne(Label(data_root + '/label.csv'), is_input=False)

    loader = DataLoader(data, batch_size=batch_size)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def _GetSubList(label_root, is_test=False):
    if is_test:
        all_label_df = os.path.join(label_root, 'test_name.csv')
    else:
        all_label_df = os.path.join(label_root, 'train_name.csv')
    all_case = pd.read_csv(all_label_df)
    all_case = all_case.values.tolist()[0]
    return all_case


def _GetLabel(label_array):
    label = []
    for index in range(label_array.shape[1]):
        label.append(np.argmax(np.bincount(label_array[:, index])))
    return np.array(label)


def ModelEnhanced(device, model_name, weights_list=None, is_test=False):
    device = device
    input_shape = (30, 200, 200)
    batch_size = 24

    model_folder = os.path.join(model_root, model_name)

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []

    for cv_index, cv_folder in enumerate(cv_folder_list):
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
        model.load_state_dict(torch.load(str(weights_path)))

        pred_list, label_list = [], []
        model.eval()
        sub_list = _GetSubList(data_root, is_test=is_test)
        data_loader, batches = _GetLoader(sub_list, data_root, input_shape, batch_size)
        with torch.no_grad():
            for inputs, outputs in data_loader:
                inputs = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(outputs, device)
                preds = model(inputs)

                pred_list.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().data.numpy().tolist())
                label_list.extend(outputs.cpu().data.numpy().astype(int).squeeze().tolist())

        cv_pred_list.append(pred_list)
        cv_label_list.append(label_list)

        del model, weights_path

    cv_pred = np.array(cv_pred_list)
    cv_label = np.array(cv_label_list)
    mean_pred = _GetLabel(cv_pred)
    mean_label = np.mean(cv_label, axis=0)

    cm = confusion_matrix(mean_pred, mean_label)
    print(cm)

    return mean_pred, mean_label, cm


if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/RenJi/Model'
    data_root = r'/home/zhangyihong/Documents/RenJi'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    ModelEnhanced(device, 'ResNet3D_0810', weights_list=None, is_test=False)