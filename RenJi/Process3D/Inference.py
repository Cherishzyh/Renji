import torch
from torch.utils.data import DataLoader

from BasicTool.MeDIT.Others import IterateCase
from BasicTool.MeDIT.Normalize import Normalize01
from CnnTools.T4T.Utility.Data import *
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


if __name__ == '__main__':
    from RenJi.Visualization.Show import ShowCM
    model_root = r'/home/zhangyihong/Documents/RenJi/Model'
    data_root = r'/home/zhangyihong/Documents/RenJi'

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    cm = Inference(device, 'ResNet3D_0825_15slice_1', data_type='train', n_classes=4, weights_list=None)
    ShowCM(cm)
    cm = Inference(device, 'ResNet3D_0825_15slice_1', data_type='val', n_classes=4, weights_list=None)
    ShowCM(cm)
    cm = Inference(device, 'ResNet3D_0825_15slice_1', data_type='test', n_classes=4, weights_list=None)
    ShowCM(cm)