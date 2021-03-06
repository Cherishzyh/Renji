import os

import pandas as pd
from copy import deepcopy
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import torch
from torch.utils.data import DataLoader

from MeDIT.Others import IterateCase
from T4T.Utility.Data import *
from MeDIT.Statistics import BinaryClassification
# from RenJi.Network2D.ResNet2D_CBAM import resnet50
# from RenJi.Network2D.ResNet2D_Focus import resnet50
# from RenJi.Network2D.AttenBlock import resnet50
# from RenJi.Network2D.ResNeXt2D import ResNeXt
from RenJi.Network2D.MergeResNet import resnet50


from RenJi.Metric.ConfusionMatrix import F1Score
from RenJi.Metric.ROC import ROC


def EnsembleInference(model_root, data_root, model_name, data_type, weights_list=None):
    from RenJi.Network2D.ResNet2D import resnet50
    input_shape = (150, 150)
    batch_size = 24
    model_folder = os.path.join(model_root, model_name)

    if data_type == 'external':
        data = DataManager()
        data.AddOne(Image2D(data_root + '/2CHNPY_5Slice', shape=input_shape))
        data.AddOne(Image2D(data_root + '/2CHPredROIDilated', shape=input_shape))
        #
        # data.AddOne(Image2D(data_root + '/3CHNPY_5Slice', shape=input_shape))
        # data.AddOne(Image2D(data_root + '/3CHPredROIDilated', shape=input_shape))
    else:
        sub_list = pd.read_csv(os.path.join(data_root, '{}_name.csv'.format(data_type)), index_col='CaseName').index.tolist()
        data = DataManager(sub_list=sub_list)
        data.AddOne(Image2D(data_root + '/2CHNPY', shape=input_shape))
        data.AddOne(Image2D(data_root + '/2CHPredDilated', shape=input_shape, is_roi=True))
        # data.AddOne(Image2D(data_root + '/3CHNPY', shape=input_shape))
        # data.AddOne(Image2D(data_root + '/3CHPredDilated', shape=input_shape, is_roi=True))
    data.AddOne(Label(data_root + '/label_2cl.csv'), is_input=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []
    for cv_index, cv_folder in enumerate(cv_folder_list):
        model = resnet50(input_channels=5, num_classes=1).to(device)
        if weights_list is None:
            one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if
                                     one.is_file()]
            one_fold_weights_list = sorted(one_fold_weights_list, key=lambda x: os.path.getctime(str(x)))
            weights_path = one_fold_weights_list[-1]
        else:
            weights_path = weights_list[cv_index]

        print(weights_path.name, end='\t')
        model.load_state_dict(torch.load(str(weights_path)))

        pred_list, label_list = [], []
        model.eval()
        for inputs, outputs in data_loader:
            image = MoveTensorsToDevice(inputs, device)
            outputs = MoveTensorsToDevice(outputs, device)

            image = image[0] * image[1]
            preds = model(image)

            pred_list.extend(torch.squeeze(torch.sigmoid(preds)).cpu().detach())
            label_list.extend(outputs.int().cpu().detach())

        auc = roc_auc_score(label_list, pred_list)
        print(auc)

        cv_pred_list.append(pred_list)
        cv_label_list.append(label_list)

        del model, weights_path

    cv_pred = np.array(cv_pred_list)
    cv_label = np.array(cv_label_list)
    mean_pred = np.mean(cv_pred, axis=0)
    mean_label = np.mean(cv_label, axis=0)
    np.save(os.path.join(model_folder, '{}_preds.npy'.format(data_type)), mean_pred)
    np.save(os.path.join(model_folder, '{}_label.npy'.format(data_type)), mean_label)
    return mean_label, mean_pred


def EnsembleInferenceBySeg(model_root, data_root, model_name, data_type, weights_list=None):
    from RenJi.Network2D.MergeResNet import resnet50
    input_shape = (150, 150)
    batch_size = 24
    model_folder = os.path.join(model_root, model_name)

    if data_type == 'external':
        data = DataManager()
        data.AddOne(Image2D(data_root + '/2CHExternalPred_5slice', shape=input_shape))
        data.AddOne(Image2D(data_root + '/3CHExternalPred_5slice', shape=input_shape))
        data.AddOne(Image2D(data_root + '/2CHExternalROIPred_5slice', shape=input_shape, is_roi=True))
        data.AddOne(Image2D(data_root + '/3CHExternalROIPred_5slice', shape=input_shape, is_roi=True))
        data.AddOne(Label(data_root + '/non_external_name.csv'), is_input=False)
    else:
        sub_list = pd.read_csv(os.path.join(data_root, '{}_name.csv'.format(data_type)), index_col='CaseName').index.tolist()
        data = DataManager(sub_list=sub_list)
        data.AddOne(Image2D(data_root + '/2CHPred_5slice', shape=input_shape))
        data.AddOne(Image2D(data_root + '/3CHPred_5slice', shape=input_shape))
        data.AddOne(Image2D(data_root + '/2CHROIPred_5slice', shape=input_shape, is_roi=True))
        data.AddOne(Image2D(data_root + '/3CHROIPred_5slice', shape=input_shape, is_roi=True))
        data.AddOne(Label(data_root + '/label_2cl.csv'), is_input=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []
    for cv_index, cv_folder in enumerate(cv_folder_list):
        model = resnet50(input_channels=5, num_classes=1).to(device)
        if weights_list is None:
            one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if one.is_file()]
            one_fold_weights_list = sorted(one_fold_weights_list,  key=lambda x: os.path.getctime(str(x)))
            weights_path = one_fold_weights_list[-1]
        else:
            weights_path = weights_list[cv_index]

        print(weights_path.name, end='\t')
        model.load_state_dict(torch.load(str(weights_path)))

        pred_list, label_list = [], []
        model.eval()
        with torch.no_grad():
            for inputs, outputs in data_loader:
                image = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(outputs, device)

                # image_2ch = image[0] * image[2]
                # image_3ch = image[1] * image[3]
                image_2ch = image[0]
                image_3ch = image[1]
                preds = model(image_2ch, image_3ch)

                pred_list.extend(torch.squeeze(torch.sigmoid(preds)).cpu().detach())
                label_list.extend(outputs.int().cpu().detach())

        auc = roc_auc_score(label_list, pred_list)
        print(auc)

        cv_pred_list.append(pred_list)
        cv_label_list.append(label_list)

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

    bc = BinaryClassification()
    bc.Run(pred.tolist(), np.asarray(label, dtype=np.int32).tolist())
    binary_pred = deepcopy(pred)
    fpr, tpr, threshold = metrics.roc_curve(np.asarray(label, dtype=np.int32).tolist(), pred.tolist())
    index = np.argmax(1 - fpr + tpr)
    binary_pred[binary_pred >= threshold[index]] = 1
    binary_pred[binary_pred < threshold[index]] = 0
    cm = confusion_matrix(label.tolist(), binary_pred.tolist())
    ShowCM(cm)


def DrawROC(model_folder):
    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')

    train_pred = np.load(os.path.join(model_folder, '{}_preds.npy'.format('non_alltrain')))
    train_label = np.load(os.path.join(model_folder, '{}_label.npy'.format('non_alltrain')))
    fpn, sen, the = roc_curve(train_label.tolist(), train_pred.tolist())
    auc = roc_auc_score(train_label.tolist(), train_pred.tolist())
    plt.plot(fpn, sen, label='Training: {:.3f}'.format(auc))

    test_pred = np.load(os.path.join(model_folder, '{}_preds.npy'.format('non_test')))
    test_label = np.load(os.path.join(model_folder, '{}_label.npy'.format('non_test')))
    fpn, sen, the = roc_curve(test_label.tolist(), test_pred.tolist())
    auc = roc_auc_score(test_label.tolist(), test_pred.tolist())
    plt.plot(fpn, sen, label='Testing:  {:.3f}'.format(auc))

    if os.path.exists(os.path.join(model_folder, '{}_preds.npy'.format('external'))):
        external_pred = np.load(os.path.join(model_folder, '{}_preds.npy'.format('external')))
        external_label = np.load(os.path.join(model_folder, '{}_label.npy'.format('external')))
        fpn, sen, the = roc_curve(external_label.tolist(), external_pred.tolist())
        auc = roc_auc_score(external_label.tolist(), external_pred.tolist())
        plt.plot(fpn, sen, label='External: {:.3f}'.format(auc))

    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.show()
    plt.close()


if __name__ == '__main__':
    from RenJi.Visualization.Show import ShowCM
    model_root = r'/home/zhangyihong/Documents/RenJi/Model2D'
    data_root = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData'
    model_name = 'ResNet_1129_SliceBySeg'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # cm = Inference(device, 'ResNeXt_0914_5slice_focal', data_type='train', n_classes=4, weights_list=None)
    # ShowCM(cm)
    # cm = Inference(device, 'ResNeXt_0914_5slice_focal', data_type='val', n_classes=4, weights_list=None)
    # ShowCM(cm)
    # cm = Inference(device, 'ResNeXt_0914_5slice_focal', data_type='test', n_classes=4, weights_list=None)
    # ShowCM(cm)

    # EnsembleInference(model_root, data_root, model_name, 'non_alltrain', weights_list=None)
    # EnsembleInference(model_root, data_root, model_name, 'non_test', weights_list=None)
    # EnsembleInference(model_root, r'/home/zhangyihong/Documents/RenJi/ExternalTest',
    #                   model_name, 'external', weights_list=None)
    #
    # Result4NPY(os.path.join(model_root, model_name), data_type='non_alltrain', n_class=2)
    # Result4NPY(os.path.join(model_root, model_name), data_type='non_test', n_class=2)
    # Result4NPY(os.path.join(model_root, model_name), data_type='external', n_class=2)
    # DrawROC(os.path.join(model_root, model_name))


    # EnsembleInferenceBySeg(model_root, data_root, model_name, 'non_alltrain', weights_list=None)
    # EnsembleInferenceBySeg(model_root, data_root, model_name, 'non_test', weights_list=None)
    # EnsembleInferenceBySeg(model_root, data_root, model_name, 'external', weights_list=None)
    # Result4NPY(os.path.join(model_root, model_name), data_type='non_alltrain', n_class=2)
    # Result4NPY(os.path.join(model_root, model_name), data_type='non_test', n_class=2)
    # Result4NPY(os.path.join(model_root, model_name), data_type='external', n_class=2)
    DrawROC(os.path.join(model_root, model_name))