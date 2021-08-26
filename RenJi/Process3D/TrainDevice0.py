import os.path
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


from BasicTool.MeDIT.Augment import *
from BasicTool.MeDIT.Others import MakeFolder, CopyFile
from BasicTool.MeDIT.Normalize import Normalize01

from CnnTools.T4T.Utility.Data import *
from CnnTools.T4T.Utility.CallBacks import EarlyStopping

from RenJi.Network3D.ResNet3D import GenerateModel
# from RenJi.Network3D.ResNet3D_Focus import GenerateModel


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(data_root, sub_list, aug_param_config, input_shape, batch_size, shuffle, is_balance=True):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)

    data.AddOne(Image2D(data_root + '/Npy2D', shape=input_shape))
    data.AddOne(Label(data_root + '/label_norm.csv'), is_input=False)
    if is_balance:
        data.Balance(Label(data_root + '/label_norm.csv'))

    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=16)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches

def Train(device, model_root, model_name, data_root):
    torch.autograd.set_detect_anomaly(True)

    device = device
    input_shape = (200, 200)
    total_epoch = 10000
    batch_size = 12
    model_folder = MakeFolder(model_root + '/{}'.format(model_name))
    if os.path.exists(model_folder):
        ClearGraphPath(model_folder)

    param_config = {
        RotateTransform.name: {'theta': ['uniform', -10, 10]},
        ShiftTransform.name: {'horizontal_shift': ['uniform', -0.05, 0.05],
                              'vertical_shift': ['uniform', -0.05, 0.05]},
        ZoomTransform.name: {'horizontal_zoom': ['uniform', 0.95, 1.05],
                             'vertical_zoom': ['uniform', 0.95, 1.05]},
        FlipTransform.name: {'horizontal_flip': ['choice', True, False]},
        BiasTransform.name: {'center': ['uniform', -1., 1., 2],
                             'drop_ratio': ['uniform', 0., 1.]},
        NoiseTransform.name: {'noise_sigma': ['uniform', 0., 0.03]},
        ContrastTransform.name: {'factor': ['uniform', 0.8, 1.2]},
        GammaTransform.name: {'gamma': ['uniform', 0.8, 1.2]},
        ElasticTransform.name: ['elastic', 1, 0.1, 256]
    }

    train_case = pd.read_csv(os.path.join(data_root, 'train_name.csv'), index_col='CaseName')
    train_case = train_case.index.tolist()

    val_case = pd.read_csv(os.path.join(data_root, 'val_name.csv'), index_col='CaseName')
    val_case = val_case.index.tolist()

    train_loader, train_batches = _GetLoader(data_root, train_case, param_config, input_shape, batch_size, True, True)
    val_loader, val_batches = _GetLoader(data_root, val_case, None, input_shape, batch_size, True, False)

    # no softmax or sigmoid
    model = GenerateModel(50, n_input_channels=1, n_classes=4).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ce_loss = torch.nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(store_path=str(model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
    writer = SummaryWriter(log_dir=str(model_folder / 'log'), comment='Net')

    for epoch in range(total_epoch):
        train_loss, val_loss = 0., 0.
        train_pred, val_pred = [], []
        train_label, val_label = [], []

        model.train()
        for ind, (inputs, outputs) in enumerate(train_loader):

            optimizer.zero_grad()

            inputs_0 = inputs[:, np.newaxis, 15:, ...]
            # inputs_1 = abs(inputs - inputs[:, 0:1])
            # inputs_1 = torch.from_numpy(Normalize01(inputs_1[:, np.newaxis, ...]))
            inputs_0 = MoveTensorsToDevice(inputs_0, device)
            # inputs_1 = MoveTensorsToDevice(inputs_1, device)
            outputs = MoveTensorsToDevice(outputs, device)

            # preds = model(inputs_0, inputs_1)
            preds = model(inputs_0)

            loss = ce_loss(preds, outputs.long())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pred.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().data.numpy().tolist())
            train_label.extend(outputs.cpu().data.numpy().astype(int).squeeze().tolist())

        train_acc = sum([1 for index in range(len(train_label)) if train_label[index] == train_pred[index]]) / len(
            train_label)

        model.eval()
        with torch.no_grad():
            for ind, (inputs, outputs) in enumerate(val_loader):
                inputs_0 = inputs[:, np.newaxis, 15:, ...]
                # inputs_1 = abs(inputs - inputs[:, 0:1])
                # inputs_1 = torch.from_numpy(Normalize01(inputs_1[:, np.newaxis, ...]))
                inputs_0 = MoveTensorsToDevice(inputs_0, device)
                # inputs_1 = MoveTensorsToDevice(inputs_1, device)
                outputs = MoveTensorsToDevice(outputs, device)

                # preds = model(inputs_0, inputs_1)
                preds = model(inputs_0)

                loss = ce_loss(preds, outputs.long())

                val_loss += loss.item()

                val_pred.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().data.numpy().tolist())
                val_label.extend(outputs.cpu().data.numpy().astype(int).squeeze().tolist())

        val_acc = sum([1 for index in range(len(val_label)) if val_label[index] == val_pred[index]]) / len(val_label)

        # Save Tensor Board
        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Loss', {'train_loss': train_loss / train_batches, 'val_loss': val_loss / val_batches},
                           epoch + 1)
        writer.add_scalars('Acc', {'train_acc': train_acc, 'val_acc': val_acc}, epoch + 1)

        print('Epoch {}:\tloss: {:.3f}, val-loss: {:.3f}; acc: {:.3f}, val-acc: {:.3f}'.format(
            epoch + 1, train_loss / train_batches, val_loss / val_batches, train_acc, val_acc))
        print('         \ttrain: {}, {}, {}, {},\tval: {}, {}, {}, {}'.format(
            train_pred.count(0), train_pred.count(1), train_pred.count(2), train_pred.count(3),
            val_pred.count(0), val_pred.count(1), val_pred.count(2), val_pred.count(3)))

        scheduler.step(val_loss)
        early_stopping(val_loss, model, (epoch + 1, val_loss))

        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.flush()
        writer.close()


def CheckInput():
    from BasicTool.MeDIT.Visualization import Imshow3DArray
    from BasicTool.MeDIT.Normalize import Normalize01
    torch.autograd.set_detect_anomaly(True)
    input_shape = (200, 200)
    batch_size = 4
    data_root = r'Z:\RenJi'

    param_config = {
        RotateTransform.name: {'theta': ['uniform', -10, 10]},
        ShiftTransform.name: {'horizontal_shift': ['uniform', -0.05, 0.05],
                              'vertical_shift': ['uniform', -0.05, 0.05]},
        ZoomTransform.name: {'horizontal_zoom': ['uniform', 0.95, 1.05],
                             'vertical_zoom': ['uniform', 0.95, 1.05]},
        FlipTransform.name: {'horizontal_flip': ['choice', True, False]},
        BiasTransform.name: {'center': ['uniform', -1., 1., 2],
                             'drop_ratio': ['uniform', 0., 1.]},
        NoiseTransform.name: {'noise_sigma': ['uniform', 0., 0.03]},
        ContrastTransform.name: {'factor': ['uniform', 0.8, 1.2]},
        GammaTransform.name: {'gamma': ['uniform', 0.8, 1.2]},
        ElasticTransform.name: ['elastic', 1, 0.1, 256]
    }

    sub_train = pd.read_csv(os.path.join(data_root, 'train_name.csv'), index_col='CaseName')
    sub_train = sub_train.index.tolist()
    train_loader, train_batches = _GetLoader(data_root, sub_train, param_config, input_shape, batch_size, True)

    for ind, (inputs, outputs) in enumerate(train_loader):
        for index in range(inputs.shape[0]):
            Imshow3DArray(Normalize01(np.squeeze(inputs[index])).transpose(1, 2, 0))


if __name__ == '__main__':
    # in own computer
    # CheckInput()

    model_root = r'/home/zhangyihong/Documents/RenJi/Model'
    data_root = r'/home/zhangyihong/Documents/RenJi'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    Train(device, model_root, 'ResNet3D_0825_15slice_1', data_root)
