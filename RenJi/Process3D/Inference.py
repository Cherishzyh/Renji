import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from sklearn.metrics import roc_auc_score, roc_curve
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader

from MeDIT.Others import IterateCase
from MeDIT.Normalize import Normalize01
from MeDIT.Statistics import BinaryClassification
from T4T.Utility.Data import *
# from RenJi.Network3D.ResNet3D import GenerateModel
from RenJi.Network3D.MergeResNet3D import GenerateModel
from RenJi.SegModel2D.UNet import UNet
# from RenJi.Network3D.ResNet3D_Focus import GenerateModel
from RenJi.Metric.ConfusionMatrix import F1Score


def _GetLoader(sub_list, data_root, input_shape, batch_size):
    data = DataManager(sub_list=sub_list)

    data.AddOne(Image2D(data_root + '/Npy2D',  shape=input_shape))
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
    input_shape = (200, 200)
    batch_size = 8

    model_folder = os.path.join(model_root, model_name)

    model = GenerateModel(50, n_input_channels=1, n_classes=n_classes).to(device)
    if weights_list is None:
        weights_list = [one for one in IterateCase(model_folder, only_folder=False, verbose=0) if one.is_file()]
        weights_list = [one for one in weights_list if str(one).endswith('.pt')]
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

            inputs_0 = inputs[:, np.newaxis, 15:, ...]
            # inputs_1 = abs(inputs - inputs[:, 0:1])
            # inputs_1 = torch.from_numpy(Normalize01(inputs_1[:, np.newaxis, ...]))
            inputs_0 = MoveTensorsToDevice(inputs_0, device)
            # inputs_1 = MoveTensorsToDevice(inputs_1, device)

            # preds = model(inputs_0, inputs_1)
            preds = model(inputs_0)

            pred_list.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().data.numpy().tolist())
            if not isinstance(outputs.squeeze().tolist(), list):
                label_list.append(outputs.squeeze().tolist())
            else:
                label_list.extend(outputs.squeeze().tolist())

    del model, weights_path

    precision, recall, f1_score, cm = F1Score(label_list, pred_list)
    print(precision)
    print(recall)
    print(f1_score)
    print(cm)

    return cm


def EnsembleInference(model_root, data_root, model_name, data_type, weights_list=None, n_class=2):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (150, 150)
    batch_size = 8
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
        model =GenerateModel(50, n_input_channels=1, n_classes=n_class).to(device).to(device)
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
            image = torch.unsqueeze(image, dim=1)
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


def EnsembleInferenceBySeg(model_root, data_root, model_name, data_type, weights_list=None):

    input_shape = (150, 150)
    model_folder = os.path.join(model_root, model_name)

    if data_type == 'external':
        batch_size = 1
        data = DataManager()
        data.AddOne(Image2D(data_root + '/2CHExternalPred', shape=input_shape))
        data.AddOne(Image2D(data_root + '/3CHExternalPred', shape=input_shape))
        data.AddOne(Image2D(data_root + '/2CHExternalROIPred', shape=input_shape, is_roi=True))
        data.AddOne(Image2D(data_root + '/3CHExternalROIPred', shape=input_shape, is_roi=True))
        data.AddOne(Label(data_root + '/external_label_2cl.csv'), is_input=False)
    else:
        batch_size = 8
        sub_list = pd.read_csv(os.path.join(data_root, '{}_name.csv'.format(data_type)), index_col='CaseName').index.tolist()
        try:
            sub_list.remove('20180502 jingchunhua')
        except Exception:
            pass
        try:
            sub_list.remove('20151224 liuhongfei')
        except Exception:
            pass
        data = DataManager(sub_list=sub_list)
        data.AddOne(Image2D(data_root + '/2CHPred', shape=input_shape))
        data.AddOne(Image2D(data_root + '/3CHPred', shape=input_shape))
        data.AddOne(Image2D(data_root + '/2CHROIPred', shape=input_shape, is_roi=True))
        data.AddOne(Image2D(data_root + '/3CHROIPred', shape=input_shape, is_roi=True))
        data.AddOne(Label(data_root + '/label_2cl.csv'), is_input=False)

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=36, pin_memory=True)

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []
    for cv_index, cv_folder in enumerate(cv_folder_list):
        model = GenerateModel(50, n_input_channels=1, n_classes=1).to(device)
        if weights_list is None:
            one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if one.is_file()]
            one_fold_weights_list = sorted(one_fold_weights_list,  key=lambda x: os.path.getctime(str(x)))
            weights_path = one_fold_weights_list[-1]
        else:
            weights_path = weights_list[cv_index]

        print(weights_path.name)
        model.load_state_dict(torch.load(str(weights_path), map_location=lambda storge, loc: storge.cuda(1)))

        pred_list, label_list = [], []
        model.eval()
        with torch.no_grad():
            for inputs, outputs in data_loader:
                image = MoveTensorsToDevice(inputs, device)
                image_2ch = image[0] * image[2]
                image_2ch = torch.unsqueeze(image_2ch, dim=1)
                image_3ch = image[1] * image[3]
                image_3ch = torch.unsqueeze(image_3ch, dim=1)

                # image_2ch = image[0]
                # image_2ch = torch.unsqueeze(image_2ch, dim=1)
                # image_3ch = image[1]
                # image_3ch = torch.unsqueeze(image_3ch, dim=1)

                preds = model(image_2ch, image_3ch)

                if data_type == 'external':
                    pred_list.append(float(torch.squeeze(torch.sigmoid(preds)).cpu().detach()))
                    label_list.append(int(outputs.cpu().detach()))
                else:
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


def Result4NPY(model_folder, data_type, n_class=2):
    pred = np.load(os.path.join(model_folder, '{}_preds.npy'.format(data_type)))
    label = np.load(os.path.join(model_folder, '{}_label.npy'.format(data_type)))

    if n_class == 2:
        bc = BinaryClassification()
        bc.Run(pred.tolist(), np.asarray(label, dtype=np.int32).tolist())
        binary_pred = deepcopy(pred)
        fpr, tpr, threshold = metrics.roc_curve(np.asarray(label, dtype=np.int32).tolist(), pred.tolist())
        index = np.argmax(1 - fpr + tpr)
        binary_pred[binary_pred >= threshold[index]] = 1
        binary_pred[binary_pred < threshold[index]] = 0
        cm = confusion_matrix(label.tolist(), binary_pred.tolist())
        ShowCM(cm)
    else:
        precision, recall, f1_score, cm = F1Score(label.tolist(), np.argmax(pred, axis=-1).tolist())
        ShowCM(cm)
        print(data_type)
        print('precision', [float('{:.3f}'.format(i)) for i in precision])
        print('recall   ', [float('{:.3f}'.format(i)) for i in recall])
        print('f1_score ', [float('{:.3f}'.format(i)) for i in f1_score])


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
    model_root = r'/home/zhangyihong/Documents/RenJi/Model3D'
    data_root = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData'
    model_name = 'ResNet3D_1123_mask'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # EnsembleInference(model_root, data_root, model_name, data_type='alltrain',  weights_list=None, n_class=2)
    # EnsembleInference(model_root, data_root, model_name, data_type='test',  weights_list=None, n_class=2)
    EnsembleInferenceBySeg(model_root, data_root, model_name, data_type='non_alltrain',  weights_list=None)
    EnsembleInferenceBySeg(model_root, data_root, model_name, data_type='non_test',  weights_list=None)
    EnsembleInferenceBySeg(model_root, data_root, model_name, data_type='external', weights_list=None)
    Result4NPY(os.path.join(model_root, model_name), data_type='non_alltrain', n_class=2)
    Result4NPY(os.path.join(model_root, model_name), data_type='non_test', n_class=2)
    Result4NPY(os.path.join(model_root, model_name), data_type='external', n_class=2)
    DrawROC(os.path.join(model_root, model_name))