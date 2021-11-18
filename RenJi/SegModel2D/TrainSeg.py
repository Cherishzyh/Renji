import os.path
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from MeDIT.Augment import *
from MeDIT.Others import MakeFolder

from T4T.Utility.Data import *
from T4T.Utility.CallBacks import EarlyStopping
from T4T.Utility.Initial import HeWeightInit

from RenJi.SegModel2D.UNet import UNet
from RenJi.Metric.Loss import DiceLoss
from RenJi.Metric.ROC import Dice


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(data_root, sub_list, aug_param_config, input_shape, batch_size, shuffle):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)

    data.AddOne(Image2D(data_root + '/NPYOneSlice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/ROIOneSlice', shape=input_shape, is_roi=True), is_input=False)

    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=24, pin_memory=True)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def Train(device, model_root, model_name, data_root):
    torch.autograd.set_detect_anomaly(True)

    input_shape = (200, 200)
    total_epoch = 10000
    batch_size = 12

    model_folder = MakeFolder(model_root + '/{}'.format(model_name))
    ClearGraphPath(model_folder)

    param_config = {
        RotateTransform.name: {'theta': ['uniform', -10, 10]},
        ShiftTransform.name: {'horizontal_shift': ['uniform', -0.05, 0.05],
                              'vertical_shift': ['uniform', -0.05, 0.05]},
        ZoomTransform.name: {'horizontal_zoom': ['uniform', 0.95, 1.05],
                             'vertical_zoom': ['uniform', 0.95, 1.05]},
        FlipTransform.name: {'horizontal_flip': ['choice', False, False]},
        BiasTransform.name: {'center': ['uniform', -1., 1., 2],
                             'drop_ratio': ['uniform', 0., 1.]},
        NoiseTransform.name: {'noise_sigma': ['uniform', 0., 0.03]},
        ContrastTransform.name: {'factor': ['uniform', 0.8, 1.2]},
        GammaTransform.name: {'gamma': ['uniform', 0.8, 1.2]},
        ElasticTransform.name: ['elastic', 1, 0.1, 256]
    }

    train_csv_list = pd.read_csv(os.path.join(data_root, 'normal_train.csv'), index_col='CaseName').index.tolist()
    train_list = ['2ch_{}'.format(case) for case in train_csv_list] + ['3ch_{}'.format(case) for case in train_csv_list]
    val_csv_list = pd.read_csv(os.path.join(data_root, 'normal_val.csv'), index_col='CaseName').index.tolist()
    val_list = ['2ch_{}'.format(case) for case in val_csv_list] + ['3ch_{}'.format(case) for case in val_csv_list]

    train_loader, train_batches = _GetLoader(data_root, train_list, param_config, input_shape, batch_size, True)
    val_loader, val_batches = _GetLoader(data_root, val_list, None, input_shape, batch_size, True)

    model = UNet(in_channels=1, out_channels=1).to(device)
    model.apply(HeWeightInit)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(store_path=str(model_folder / '{}-{:.6f}.pt'), patience=100, verbose=True)
    writer = SummaryWriter(log_dir=str(model_folder / 'log'), comment='Net')

    for epoch in range(total_epoch):
        train_loss, val_loss = 0., 0.
        train_dice_loss, val_dice_loss = 0., 0.
        train_bce_loss, val_bce_loss = 0., 0.
        train_dice, val_dice = 0., 0.

        model.train()
        for ind, (inputs, outputs) in enumerate(train_loader):
            inputs = MoveTensorsToDevice(inputs, device)
            outputs = MoveTensorsToDevice(outputs, device)

            preds = model(inputs)

            optimizer.zero_grad()

            bce = bce_loss(preds, outputs)
            dice = dice_loss(torch.sigmoid(preds), outputs)
            loss = dice + bce
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice_loss += dice.item()
            train_bce_loss += bce.item()

            train_dice += Dice(torch.sigmoid(preds.detach()), outputs.detach())

        model.eval()
        with torch.no_grad():
            for ind, (inputs, outputs) in enumerate(val_loader):
                inputs = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(outputs, device)

                preds = model(inputs)

                bce = bce_loss(preds, outputs)
                dice = dice_loss(torch.sigmoid(preds), outputs)
                loss = dice + bce

                val_loss += loss.item()
                val_dice_loss += dice.item()
                val_bce_loss += bce.item()

                val_dice += Dice(torch.sigmoid(preds.detach()), outputs.detach())

        # Save Tensor Board
        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().detach().numpy(), epoch + 1)

        writer.add_scalars('Loss', {'train_loss': train_loss / train_batches,
                                    'train_bce_loss': train_bce_loss / train_batches,
                                    'train_dice_loss': train_dice_loss / train_batches,
                                    'val_loss': val_loss / val_batches,
                                    'val_bce_loss': val_bce_loss / val_batches,
                                    'val_dice_loss': val_dice_loss / val_batches
                                    }, epoch + 1)
        writer.add_scalars('Dice', {'train_dice': train_dice / train_batches, 'val_dice': val_dice / val_batches}, epoch + 1)

        print('Epoch {}:\tloss: {:.3f}, val-loss: {:.3f}; train-dice: {:.3f}, val-dice: {:.3f}'.format(
            epoch + 1, train_loss / train_batches, val_loss / val_batches,
            train_dice / train_batches, val_dice / val_batches))

        scheduler.step(val_loss)
        early_stopping(val_loss, model, (epoch + 1, val_loss))

        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.flush()


if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/RenJi/SegModel'
    data_root = r'/home/zhangyihong/Documents/RenJi/Data/SegData'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    Train(device, model_root, 'UNet_1118', data_root)
