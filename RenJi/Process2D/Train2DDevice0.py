import shutil
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


from BasicTool.MeDIT.Augment import *
from BasicTool.MeDIT.Others import MakeFolder, CopyFile
# from MeDIT.Augment import *
# from MeDIT.Others import MakeFolder, CopyFile

from CnnTools.T4T.Utility.Data import *
from CnnTools.T4T.Utility.CallBacks import EarlyStopping
from CnnTools.T4T.Utility.Initial import HeWeightInit

from Network3D.ResNet3D import GenerateModel

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
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def _GetCV(label_root, cv):
    all_label_df = os.path.join(label_root, 'train_name.csv')
    all_case = pd.read_csv(all_label_df)
    all_case = all_case.values.tolist()[0]

    cv_label = os.path.join(label_root, 'train-cv{}.csv'.format(cv))
    cv_val = pd.read_csv(cv_label)
    cv_val = cv_val.loc[:, 'CaseName'].tolist()

    cv_train = [case for case in all_case if case not in cv_val]

    return cv_train, cv_val


def EnsembleTrain(device, model_root, data_root):
    torch.autograd.set_detect_anomaly(True)

    device = device
    input_shape = (200, 200)
    total_epoch = 10000
    batch_size = 12
    cv_folder = 5
    if not os.path.exists(model_root):
        os.mkdir(model_root)
    model_folder = MakeFolder(model_root + '/ResNet3D_0810_2Daug')
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

    for cv_index in range(cv_folder):
        sub_model_folder = MakeFolder(model_folder / 'CV_{}'.format(cv_index))

        sub_train, sub_val = _GetCV(data_root, cv_index + 1)
        train_loader, train_batches = _GetLoader(sub_train, param_config, input_shape, batch_size, True)
        val_loader, val_batches = _GetLoader(sub_val, None, input_shape, batch_size, True)

        # no softmax or sigmoid
        model = GenerateModel(50, n_input_channels=1, n_classes=4).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ce_loss = torch.nn.CrossEntropyLoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5,
                                                               verbose=True)
        early_stopping = EarlyStopping(store_path=str(sub_model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
        writer = SummaryWriter(log_dir=str(sub_model_folder / 'log'), comment='Net')

        for epoch in range(total_epoch):
            train_loss, val_loss = 0., 0.

            model.train()
            for ind, (inputs, outputs) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs = inputs[:, np.newaxis, ...]
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

        del writer, optimizer, scheduler, early_stopping, model


def CheckInput():
    from BasicTool.MeDIT.Visualization import Imshow3DArray
    from BasicTool.MeDIT.Normalize import Normalize01
    torch.autograd.set_detect_anomaly(True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (30, 200, 200)
    batch_size = 4

    random_3d_augment = {
        'stretch_x': [0.8, 1.2],
        'stretch_y': [0.8, 1.2],
        'stretch_z': 1.0,
        'shear': 0.,
        'rotate_x_angle': 0,
        'rotate_z_angle': [-10, 10],
        'shift_x': [-0.1, 0.1],
        'shift_y': [-0.1, 0.1],
        'shift_z': 0,
        'horizontal_flip': False,
        'vertical_flip': False,
        'slice_flip': False,

        'bias_center': [[0.25, 0.75], [0.25, 0.75]],  # 0代表在center在中央，1代表全图像随机取，0.5代表在0.25-0.75范围内随机取
        'bias_drop_ratio': [0, 0.5],  # 随机生成0.5及以下的drop ratio
        'noise_sigma': [0., 0.03],
        'factor': [0.8, 1.2],
        'gamma': [0.8, 1.2],

        'elastic': False
    }

    sub_train = pd.read_csv(r'D:\Data\renji\Model\ResNet3D_0805\train-cv1.csv')
    sub_train = sub_train.loc[:, 'CaseName'].tolist()
    train_loader, train_batches = _GetLoader(sub_train, random_3d_augment, input_shape, batch_size, True)

    for ind, (inputs, outputs) in enumerate(train_loader):
        # inputs = MoveTensorsToDevice(inputs, device)
        # outputs = MoveTensorsToDevice(outputs, device)
        for index in range(inputs.shape[0]):
            print(inputs[0].shape)
            # Imshow3DArray(Normalize01(np.squeeze(inputs[index])).transpose(1, 2, 0))


if __name__ == '__main__':
    # CheckInput()
    # _GetCV(r'D:\Data\renji', 1)
    model_root = r'/home/zhangyihong/Documents/RenJi/Model2D'
    data_root = r'/home/zhangyihong/Documents/RenJi'
    # model_root = r'D:\Data\renji\Model'
    # data_root = r'D:\Data\renji'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    EnsembleTrain(device, model_root, data_root)