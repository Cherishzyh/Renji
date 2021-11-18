import os.path
import shutil
import torch.nn as nn

import pandas as pd
from scipy.ndimage import binary_dilation
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score

from MeDIT.Augment import *
from MeDIT.Others import MakeFolder, CopyFile

from T4T.Utility.Data import *
from T4T.Utility.CallBacks import EarlyStopping
from T4T.Utility.Initial import HeWeightInit

from RenJi.Network2D.ResNet2D import resnet50
# from RenJi.Network2D.ResNet2D_Focus import resnet50
# from RenJi.Network2D.AttenBlock import resnet50
# from RenJi.Network2D.ResNet2D_CBAM import resnet50
# from RenJi.Network2D.ResNeXt2D import ResNeXt
from RenJi.SegModel2D.UNet import UNet
# from RenJi.Network2D.MergeResNet import resnet50


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(data_root, sub_list, aug_param_config, input_shape, batch_size, shuffle, is_balance=True):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)

    # data.AddOne(Image2D(data_root + '/2CHNPY', shape=input_shape))
    data.AddOne(Image2D(data_root + '/3CHNPY', shape=input_shape))
    # data.AddOne(Image2D(data_root + '/2CHPredDilated', shape=input_shape, is_roi=True))
    data.AddOne(Image2D(data_root + '/3CHPredDilated', shape=input_shape, is_roi=True))

    data.AddOne(Label(data_root + '/label_2cl.csv'), is_input=False)
    if is_balance:
        data.Balance(Label(data_root + '/label_2cl.csv'))

    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def EnsembleTrain(device, model_root, model_name, data_root):
    torch.autograd.set_detect_anomaly(True)

    input_shape = (150, 150)
    total_epoch = 10000
    batch_size = 36

    model_folder = MakeFolder(model_root + '/{}'.format(model_name))
    if os.path.exists(model_folder):
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

    alltrain_list = pd.read_csv(os.path.join(data_root, 'alltrain_name.csv'), index_col='CaseName').index.tolist()
    for cv_index in range(1, 6):
        sub_model_folder = MakeFolder(model_folder / 'CV_{}'.format(cv_index))
        sub_val = pd.read_csv(os.path.join(data_root, 'train-cv{}.csv'.format(cv_index)), index_col='CaseName').index.tolist()
        sub_train = [case for case in alltrain_list if case not in sub_val]
        train_loader, train_batches = _GetLoader(data_root, sub_train, param_config, input_shape, batch_size, True)
        val_loader, val_batches = _GetLoader(data_root, sub_val, None, input_shape, batch_size, True)

        model = ResNeXt(input_channels=5, num_classes=2, num_blocks=[3, 4, 6, 3]).to(device)
        model.apply(HeWeightInit)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ce_loss = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
        early_stopping = EarlyStopping(store_path=str(sub_model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
        writer = SummaryWriter(log_dir=str(sub_model_folder / 'log'), comment='Net')

        for epoch in range(total_epoch):
            train_loss, val_loss = 0., 0.
            train_pred, val_pred = [], []
            train_label, val_label = [], []

            model.train()
            for ind, (inputs, outputs) in enumerate(train_loader):
                image = inputs[0] * inputs[1]
                image = torch.cat([image[:, 9: 12], image[:, -2:]], dim=1)
                image = MoveTensorsToDevice(image, device)
                outputs[outputs <= 1] = 0
                outputs[outputs >= 2] = 1
                outputs = MoveTensorsToDevice(outputs, device)

                preds = model(image)

                optimizer.zero_grad()

                loss = ce_loss(preds, outputs.long())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                train_pred.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().detach().numpy().tolist())
                train_label.extend(outputs.cpu().data.numpy().astype(int).squeeze().tolist())

            train_acc = sum([1 for index in range(len(train_label)) if train_label[index] == train_pred[index]]) / len(
                train_label)

            model.eval()
            with torch.no_grad():
                for ind, (inputs, outputs) in enumerate(val_loader):
                    image = inputs[0] * inputs[1]
                    image = torch.cat([image[:, 9: 12], image[:, -2:]], dim=1)
                    outputs[outputs <= 1] = 0
                    outputs[outputs >= 2] = 1
                    # outputs = torch.zeros(outputs.shape[0], 4).scatter_(1, torch.unsqueeze(outputs, dim=1).long(), 1)

                    image = MoveTensorsToDevice(image, device)
                    outputs = MoveTensorsToDevice(outputs, device)

                    preds = model(image)

                    loss = ce_loss(preds, outputs.long())

                    val_loss += loss.item()

                    val_pred.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().detach().numpy().tolist())
                    val_label.extend(outputs.cpu().data.numpy().astype(int).squeeze().tolist())

            val_acc = sum([1 for index in range(len(val_label)) if val_label[index] == val_pred[index]]) / len(
                val_label)

            # Save Tensor Board
            for index, (name, param) in enumerate(model.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name + '_data', param.cpu().detach().numpy(), epoch + 1)

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

        del writer, optimizer, scheduler, early_stopping, model


def EnsembleTrainBySeg(device, model_root, model_name, data_root):
    torch.autograd.set_detect_anomaly(True)

    input_shape = (150, 150)
    total_epoch = 10000
    batch_size = 36

    model_folder = MakeFolder(model_root + '/{}'.format(model_name))
    if os.path.exists(model_folder):
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
    # seg_model_path_1 = r'/home/zhangyihong/Documents/RenJi/SegModel/UNet_1026_2ch/84-0.119721.pt'
    # seg_model_1 = UNet(in_channels=1, out_channels=1).to(device)
    # seg_model_1.load_state_dict(torch.load(str(seg_model_path_1)))
    # seg_model_1.eval()

    alltrain_list = pd.read_csv(os.path.join(data_root, 'non_alltrain_name.csv'), index_col='CaseName').index.tolist()
    for cv_index in range(1, 6):
        sub_model_folder = MakeFolder(model_folder / 'CV_{}'.format(cv_index))
        sub_val = pd.read_csv(os.path.join(data_root, 'non_train-cv{}.csv'.format(cv_index)),
                              index_col='CaseName').index.tolist()
        sub_train = [case for case in alltrain_list if case not in sub_val]
        train_loader, train_batches = _GetLoader(data_root, sub_train, param_config, input_shape, batch_size, True)
        val_loader, val_batches = _GetLoader(data_root, sub_val, None, input_shape, batch_size, True)

        model = resnet50(input_channels=5, num_classes=1).to(device)
        model.apply(HeWeightInit)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        bce_loss = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5,
                                                               verbose=True)
        early_stopping = EarlyStopping(store_path=str(sub_model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
        writer = SummaryWriter(log_dir=str(sub_model_folder / 'log'), comment='Net')

        for epoch in range(total_epoch):
            train_loss, val_loss = 0., 0.

            model.train()
            for ind, (inputs, outputs) in enumerate(train_loader):
                # image = [torch.cat([inputs[0][:, 9: 12], inputs[0][:, -2:]], dim=1),
                #          torch.cat([inputs[1][:, 9: 12], inputs[1][:, -2:]], dim=1)]

                image = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(outputs, device)

                # with torch.no_grad():
                #     roi = torch.zeros_like(image[0]).to(device)
                #     for slice in range(image[0].shape[1]):
                #         roi[:, slice:slice+1] = torch.sigmoid(seg_model_1(image[0][:, slice:slice+1]))
                #         roi[roi >= 0.5] = 1
                #         roi[roi < 0.5] = 0
                #         dilate_roi_2ch = torch.from_numpy(binary_dilation(roi.cpu().detach().numpy(),
                #                                                       structure=np.ones((1, 1, 11, 11)))).to(device)
                #     roi = torch.zeros_like(image[1]).to(device)
                #     for slice in range(image[1].shape[1]):
                #         roi[:, slice:slice+1] = torch.sigmoid(seg_model_2(image[1][:, slice:slice+1]))
                #         roi[roi >= 0.5] = 1
                #         roi[roi < 0.5] = 0
                #         dilate_roi_3ch = torch.from_numpy(binary_dilation(roi.cpu().detach().numpy(),
                #                                                       structure=np.ones((1, 1, 11, 11)))).to(device)
                # image_2ch = image[0] * dilate_roi_2ch
                # image_3ch = image[1] * dilate_roi_3ch

                # preds = model(image_2ch, image_3ch)
                preds = model(image[0] * image[1])

                optimizer.zero_grad()

                loss = bce_loss(torch.squeeze(preds), outputs)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            model.eval()
            with torch.no_grad():
                for ind, (inputs, outputs) in enumerate(val_loader):
                    # image = [torch.cat([inputs[0][:, 9: 12], inputs[0][:, -2:]], dim=1),
                    #          torch.cat([inputs[1][:, 9: 12], inputs[1][:, -2:]], dim=1)]

                    image = MoveTensorsToDevice(inputs, device)
                    outputs = MoveTensorsToDevice(outputs, device)

                    # image_2ch = image[0] * dilate_roi_2ch
                    # image_3ch = image[1] * dilate_roi_3ch
                    #
                    # preds = model(image_2ch, image_3ch)
                    preds = model(image[0] * image[1])

                    loss = bce_loss(torch.squeeze(preds), outputs)

                    val_loss += loss.item()

            # Save Tensor Board
            for index, (name, param) in enumerate(model.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name + '_data', param.cpu().detach().numpy(), epoch + 1)

            writer.add_scalars('Loss', {'train_loss': train_loss / train_batches, 'val_loss': val_loss / val_batches},
                               epoch + 1)
            print('Epoch {}:\tloss: {:.3f}, val-loss: {:.3f}'.format(epoch + 1, train_loss / train_batches,
                                                                     val_loss / val_batches))

            scheduler.step(val_loss)
            early_stopping(val_loss, model, (epoch + 1, val_loss))

            if early_stopping.early_stop:
                print("Early stopping")
                break

            writer.flush()
        writer.close()

        del writer, optimizer, scheduler, early_stopping, model



if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/RenJi/Model2D'
    data_root = r'/home/zhangyihong/Documents/RenJi/NPY5slice'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # EnsembleTrain(device, model_root, 'ResNeXt_0921_5slice_cv_2cl', data_root)
    EnsembleTrainBySeg(device, model_root, 'ResNet_1028_5slice_cv_2cl_3ch', data_root)
