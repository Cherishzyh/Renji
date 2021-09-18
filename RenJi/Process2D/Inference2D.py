import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader

from MeDIT.Others import IterateCase
from MeDIT.Normalize import Normalize01
from T4T.Utility.Data import *
# from RenJi.Network2D.ResNet2D import resnet50
from RenJi.Network2D.ResNet2D_CBAM import resnet50
# from RenJi.Network2D.ResNet2D_Focus import resnet50
# from RenJi.Network2D.AttenBlock import resnet50
from RenJi.Network2D.ResNeXt2D import ResNeXt

from RenJi.Metric.ConfusionMatrix import F1Score
from RenJi.Metric.ROC import ROC


def _GetLoader(sub_list, data_root, input_shape, batch_size):
    data = DataManager(sub_list=sub_list)

    data.AddOne(Image2D(data_root + '/NPY', shape=input_shape))
    data.AddOne(Image2D(data_root + '/RoiNPY_Dilation', shape=input_shape, is_roi=True))
    data.AddOne(Label(data_root + '/label_norm.csv'), is_input=False)

    loader = DataLoader(data, batch_size=batch_size)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def _GetSubList(label_root, data_type):
    label_path = os.path.join(label_root, '{}_name.csv'.format(data_type))
    case_df = pd.read_csv(label_path, index_col='CaseName')
    case_list = case_df.index.tolist()
    return case_list


def Inference(device, model_name, data_type='test', n_classes=4, weights_list=None):
    device = device
    input_shape = (150, 150)
    batch_size = 24

    model_folder = os.path.join(model_root, model_name)

    model = ResNeXt(input_channels=5, num_classes=4, num_blocks=[3, 4, 6, 3]).to(device)
    if weights_list is None:
        weights_list = [one for one in IterateCase(model_folder, only_folder=False, verbose=0) if one.is_file()]
        weights_list = [weights for weights in weights_list if str(weights).endswith('.pt')]
        if len(weights_list) == 0:
            raise Exception
        weights_list = sorted(weights_list, key=lambda x: os.path.getctime(str(x)))
        weights_path = weights_list[-1]
    else:
        weights_path = weights_list

    print(weights_path.name)
    model.load_state_dict(torch.load(str(weights_path)))

    pred_list, label_list = [], []
    model.eval()
    sub_list = pd.read_csv(os.path.join(data_root, '{}_name.csv'.format(data_type)), index_col='CaseName')
    sub_list = sub_list.index.tolist()

    data_loader, batches = _GetLoader(sub_list, data_root, input_shape, batch_size)
    with torch.no_grad():
        for i, (inputs, outputs) in enumerate(data_loader):
            image = inputs[0] * inputs[1]
            image = torch.cat([image[:, 9: 12], image[:, -2:]], dim=1)
            image = MoveTensorsToDevice(image, device)
            preds = model(image)

            # inputs = inputs[:, :9]
            # inputs = MoveTensorsToDevice(inputs, device)
            # preds = model(inputs)
            # inputs_0 = inputs[:, :9]
            # inputs_0 = inputs_0 - torch.min(inputs_0)
            # inputs_0 = inputs_0 / torch.max(inputs_0)
            # inputs_1 = abs(inputs - inputs[:, 0:1])
            # inputs_1 = inputs_1[:, :9]
            # inputs_1 = inputs_1 - torch.min(inputs_1)
            # inputs_1 = inputs_1 / torch.max(inputs_1)

            # inputs_0 = MoveTensorsToDevice(inputs_0, device)
            # inputs_1 = MoveTensorsToDevice(inputs_1, device)
            # outputs = MoveTensorsToDevice(outputs, device)
            #
            # preds = model(inputs_0, inputs_1)
            # inputs = torch.cat([inputs[:, 9: 12], inputs[:, -2:]], dim=1)
            # inputs = MoveTensorsToDevice(inputs, device)
            # preds = model(inputs)

            pred_list.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().detach().numpy().tolist())
            label_list.extend(outputs.cpu().data.numpy().astype(int).squeeze().tolist())
            # pred_list.append(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().data.numpy().tolist())
            # label_list.append(outputs.cpu().data.numpy().astype(int).squeeze().tolist())

    del model, weights_path

    precision, recall, f1_score, cm = F1Score(label_list, pred_list)

    print([float('{:.3f}'.format(i)) for i in precision])
    print([float('{:.3f}'.format(i)) for i in recall])
    print([float('{:.3f}'.format(i)) for i in f1_score])
    print(cm)

    return cm


def EnsembleInference(model_root, data_root, model_name, data_type, weights_list=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (150, 150)
    batch_size = 24
    model_folder = os.path.join(model_root, model_name)

    sub_list = pd.read_csv(os.path.join(data_root, '{}_name.csv'.format(data_type)), index_col='CaseName').index.tolist()

    data = DataManager(sub_list=sub_list)

    data.AddOne(Image2D(data_root + '/NPY', shape=input_shape))
    data.AddOne(Image2D(data_root + '/RoiNPY_Dilation', shape=input_shape, is_roi=True))
    data.AddOne(Label(data_root + '/label_norm.csv'), is_input=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []
    for cv_index, cv_folder in enumerate(cv_folder_list):
        model = ResNeXt(input_channels=5, num_classes=2, num_blocks=[3, 4, 6, 3]).to(device)
        if weights_list is None:
            one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if one.is_file()]
            one_fold_weights_list = sorted(one_fold_weights_list,  key=lambda x: os.path.getctime(str(x)))
            weights_path = one_fold_weights_list[-1]
        else:
            weights_path = weights_list[cv_index]

        print(weights_path.name)
        model.load_state_dict(torch.load(str(weights_path)))

        pred_list, label_list = [], []
        model.eval()
        for inputs, outputs in data_loader:
            image = inputs[0] * inputs[1]
            image = torch.cat([image[:, 9: 12], image[:, -2:]], dim=1)
            image = MoveTensorsToDevice(image, device)
            outputs[outputs <= 1] = 0
            outputs[outputs >= 2] = 1

            preds = model(image)

            pred_list.extend(torch.softmax(preds, dim=1).cpu().detach().numpy().tolist())
            label_list.extend(outputs.cpu().data.numpy().astype(int).squeeze().tolist())

        cv_pred_list.append(pred_list)
        cv_label_list.append(label_list)

        _, _, f1_score, _ = F1Score(label_list, torch.argmax(torch.tensor(pred_list), dim=1))
        print([float('{:.3f}'.format(i)) for i in f1_score])

        del model, weights_path

    cv_pred = np.array(cv_pred_list)
    cv_label = np.array(cv_label_list)
    mean_pred = np.mean(cv_pred, axis=0)
    mean_label = np.mean(cv_label, axis=0)
    np.save(os.path.join(model_folder, '{}_preds.npy'.format(data_type)), mean_pred)
    np.save(os.path.join(model_folder, '{}_label.npy'.format(data_type)), mean_label)
    return mean_label, mean_pred


def Result4NPY(model_folder, data_type):
    pred = np.load(os.path.join(model_folder, '{}_preds.npy'.format(data_type)))
    label = np.load(os.path.join(model_folder, '{}_label.npy'.format(data_type)))

    fpn, sen, the = roc_curve(label.tolist(), pred[:, -1].tolist())
    auc = roc_auc_score(label.tolist(), pred[:, -1].tolist())
    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpn, sen, label='AUC: {:.3f}'.format(auc))
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.show()
    plt.close()

    precision, recall, f1_score, cm = F1Score(label.tolist(), np.argmax(pred, axis=-1).tolist())
    ShowCM(cm)
    print(data_type)
    print('precision', [float('{:.3f}'.format(i)) for i in precision])
    print('recall   ', [float('{:.3f}'.format(i)) for i in recall])
    print('f1_score ', [float('{:.3f}'.format(i)) for i in f1_score])


if __name__ == '__main__':
    from RenJi.Visualization.Show import ShowCM
    model_root = r'/home/zhangyihong/Documents/RenJi/Model2D'
    data_root = r'/home/zhangyihong/Documents/RenJi/CaseWithROI'
    model_name = 'ResNeXt_0915_5slice_cv_2cl'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # cm = Inference(device, 'ResNeXt_0914_5slice_focal', data_type='train', n_classes=4, weights_list=None)
    # ShowCM(cm)
    # cm = Inference(device, 'ResNeXt_0914_5slice_focal', data_type='val', n_classes=4, weights_list=None)
    # ShowCM(cm)
    # cm = Inference(device, 'ResNeXt_0914_5slice_focal', data_type='test', n_classes=4, weights_list=None)
    # ShowCM(cm)

    EnsembleInference(model_root, data_root, model_name, 'alltrain', weights_list=None)
    EnsembleInference(model_root, data_root, model_name, 'test', weights_list=None)

    Result4NPY(os.path.join(model_root, model_name), data_type='alltrain')
    Result4NPY(os.path.join(model_root, model_name), data_type='test')