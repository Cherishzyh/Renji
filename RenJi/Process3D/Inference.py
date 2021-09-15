import torch
from torch.utils.data import DataLoader

from MeDIT.Others import IterateCase
from MeDIT.Normalize import Normalize01
from T4T.Utility.Data import *
from RenJi.Network3D.ResNet3D import GenerateModel
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


def EnsembleInference(model_root, data_root, model_name, data_type, weights_list=None):
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
        model =GenerateModel(50, n_input_channels=1, n_classes=4).to(device).to(device)
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
    mean_pred = np.argmax(np.sum(cv_pred, axis=0), axis=1)
    mean_label = np.mean(cv_label, axis=0)

    precision, recall, f1_score, cm = F1Score(mean_label.tolist(), mean_pred.tolist())

    print([float('{:.3f}'.format(i)) for i in precision])
    print([float('{:.3f}'.format(i)) for i in recall])
    print([float('{:.3f}'.format(i)) for i in f1_score])
    print(cm)
    return cm

    # np.save(os.path.join(model_folder, '{}_preds.npy'.format(data_type)), mean_pred)
    # np.save(os.path.join(model_folder, '{}_label.npy'.format(data_type)), mean_label)


if __name__ == '__main__':
    from RenJi.Visualization.Show import ShowCM
    model_root = r'/home/zhangyihong/Documents/RenJi/Model'
    data_root = r'/home/zhangyihong/Documents/RenJi/CaseWithROI'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    cm = EnsembleInference(model_root, data_root, 'ResNet3D_0914_mask_cv', data_type='alltrain',  weights_list=None)
    ShowCM(cm)
    cm = EnsembleInference(model_root, data_root, 'ResNet3D_0914_mask_cv', data_type='test',  weights_list=None)
    ShowCM(cm)