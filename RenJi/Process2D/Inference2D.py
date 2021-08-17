import torch
from torch.utils.data import DataLoader

from BasicTool.MeDIT.Others import IterateCase
from CnnTools.T4T.Utility.Data import *
# from RenJi.Network2D.ResNet2D import resnet50
# from RenJi.Network2D.ResNet2D_CBAM import resnet50
from RenJi.Network2D.ResNet2D_Focus import resnet50

from RenJi.Metric.ConfusionMatrix import F1Score
from RenJi.Metric.ROC import ROC


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



def Inference(device, model_name, data_type='test', n_classes=4, weights_list=None):
    device = device
    input_shape = (200, 200)
    batch_size = 24

    model_folder = os.path.join(model_root, model_name)

    model = resnet50(input_channels=9, num_classes=n_classes).to(device)
    if weights_list is None:
        weights_list = [one for one in IterateCase(model_folder, only_folder=False, verbose=0) if one.is_file()]
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
            inputs_0 = inputs[:, :9]
            inputs_1 = inputs
            inputs_0 = MoveTensorsToDevice(inputs_0, device)
            inputs_1 = MoveTensorsToDevice(inputs_1, device)
            outputs = MoveTensorsToDevice(outputs, device)

            preds = model(inputs_0, inputs_1)

            # inputs = inputs[:, :9]
            # inputs = MoveTensorsToDevice(inputs, device)
            # outputs = MoveTensorsToDevice(outputs, device)
            # preds = model(inputs)
            pred_list.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().data.numpy().tolist())
            label_list.extend(outputs.cpu().data.numpy().astype(int).squeeze().tolist())
            # pred_list.append(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().data.numpy().tolist())
            # label_list.append(outputs.cpu().data.numpy().astype(int).squeeze().tolist())

    del model, weights_path

    # ROC(label_list, pred_list)

    precision, recall, f1_score, cm = F1Score(label_list, pred_list)

    print(precision)
    print(recall)
    print(f1_score)
    print(cm)

    return cm


if __name__ == '__main__':
    from RenJi.Visualization.Show import ShowCM
    model_root = r'/home/zhangyihong/Documents/RenJi/Model2D'
    data_root = r'/home/zhangyihong/Documents/RenJi'
    # model_root = r'Z:\RenJi\Model2DAug'
    # data_root = r'Z:\RenJi'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    cm = Inference(device, 'ResNet_0817_atten', data_type='train', n_classes=4, weights_list=None)
    ShowCM(cm)
    cm = Inference(device, 'ResNet_0817_atten', data_type='val', n_classes=4, weights_list=None)
    ShowCM(cm)
    cm = Inference(device, 'ResNet_0817_atten', data_type='test', n_classes=4, weights_list=None)
    ShowCM(cm)