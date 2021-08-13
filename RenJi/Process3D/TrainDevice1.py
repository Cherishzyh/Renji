import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


from BasicTool.MeDIT.Augment import *
from BasicTool.MeDIT.Others import MakeFolder, CopyFile


from CnnTools.T4T.Utility.Data import *
from CnnTools.T4T.Utility.CallBacks import EarlyStopping

from RenJi.Network3D.ResNet3D import GenerateModel


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(sub_list, aug_param_config, input_shape, batch_size, shuffle):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)

    data.AddOne(Image2D(data_root + '/Npy2D', shape=input_shape))
    data.AddOne(Label(data_root + '/label.csv'), is_input=False)
    data.Balance(Label(data_root + '/label.csv'))

    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
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
    train_case = [case for case in train_case.index if train_case.loc[case, 'Label'] < 3]
    # train_case = train_case.index.tolist()

    val_case = pd.read_csv(os.path.join(data_root, 'val_name.csv'), index_col='CaseName')
    val_case = [case for case in val_case.index if val_case.loc[case, 'Label'] < 3]
    # val_case = val_case.index.tolist()

    train_loader, train_batches = _GetLoader(train_case, param_config, input_shape, batch_size, True)
    val_loader, val_batches = _GetLoader(val_case, None, input_shape, batch_size, True)

    # no softmax or sigmoid
    model = GenerateModel(50, n_input_channels=1, n_classes=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ce_loss = torch.nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5,
                                                           verbose=True)
    early_stopping = EarlyStopping(store_path=str(model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
    writer = SummaryWriter(log_dir=str(model_folder / 'log'), comment='Net')

    for epoch in range(total_epoch):
        train_loss, val_loss = 0., 0.

        model.train()
        for ind, (inputs, outputs) in enumerate(train_loader):
            optimizer.zero_grad()

            inputs = inputs[:, np.newaxis, ...]
            outputs[outputs > 0] = 1
            # outputs[outputs == 3] = 0
            inputs = MoveTensorsToDevice(inputs, device)
            outputs = MoveTensorsToDevice(outputs, device)

            preds = model(inputs)

            loss = ce_loss(preds, outputs.long())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for ind, (inputs, outputs) in enumerate(val_loader):
                inputs = inputs[:, np.newaxis, ...]
                outputs[outputs > 0] = 1
                # outputs[outputs == 3] = 0
                inputs = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(outputs, device)

                preds = model(inputs)

                loss = ce_loss(preds, outputs.long())

                val_loss += loss.item()

        # Save Tensor Board
        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Loss',
                           {'train_loss': train_loss / train_batches,
                            'val_loss': val_loss / val_batches}, epoch + 1)

        print('Epoch {}: loss: {:.3f}, val-loss: {:.3f}'.format(
            epoch + 1, train_loss / train_batches, val_loss / val_batches))

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
    input_shape = (200, 200)
    batch_size = 12

    train_case = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/RenJi', 'train_name.csv'), index_col='CaseName')
    train_case = train_case.index.tolist()

    train_loader, train_batches = _GetLoader(train_case, None, input_shape, batch_size, True)
    label = []

    for ind, (inputs, outputs) in enumerate(train_loader):
        label.extend(sorted(outputs.tolist()))
    print(label.count(0.0), label.count(1.0), label.count(2.0), label.count(3.0))


if __name__ == '__main__':
    data_root = r'/home/zhangyihong/Documents/RenJi'

    model_root = r'/home/zhangyihong/Documents/RenJi/Model'

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    Train(device, model_root, 'ResNet3D_0812_0_12', data_root)
    # train_case = pd.read_csv(os.path.join(data_root, 'train_name.csv'), index_col='CaseName')
    # train_case = train_case.index.tolist()
