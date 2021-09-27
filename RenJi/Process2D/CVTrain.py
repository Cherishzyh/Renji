import os.path
import shutil

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

# from RenJi.Network2D.ResNet2D import resnet50
# from RenJi.Network2D.ResNet2D_Focus import resnet50
# from RenJi.Network2D.AttenBlock import resnet50
# from RenJi.Network2D.ResNet2D_CBAM import resnet50
from RenJi.Network2D.ResNeXt2D import ResNeXt
from RenJi.SegModel2D.UNet import UNet


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(data_root, sub_list, aug_param_config, input_shape, batch_size, shuffle, is_balance=True):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)

    data.AddOne(Image2D(data_root + '/NPY', shape=input_shape))
    # data.AddOne(Image2D(data_root + '/RoiNPY_Dilation', shape=input_shape, is_roi=True))
    # data.AddOne(Label(data_root + '/label_norm.csv'), is_input=False)
    # if is_balance:
    #     data.Balance(Label(data_root + '/label_norm.csv'))
    data.AddOne(Label(data_root + '/label_2cl.csv'), is_input=False)
    if is_balance:
        data.Balance(Label(data_root + '/label_2cl.csv'))

    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=36, pin_memory=True)
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
    seg_model_path = r'/home/zhangyihong/Documents/RenJi/SegModel/UNet_0922/90-0.156826.pt'
    seg_model = UNet(in_channels=1, out_channels=1).to(device)
    seg_model.load_state_dict(torch.load(str(seg_model_path)))
    seg_model.eval()

    alltrain_list = pd.read_csv(os.path.join(data_root, 'non_alltrain_name.csv'), index_col='CaseName').index.tolist()
    for cv_index in range(1, 6):
        sub_model_folder = MakeFolder(model_folder / 'CV_{}'.format(cv_index))
        sub_val = pd.read_csv(os.path.join(data_root, 'non_train-cv{}.csv'.format(cv_index)), index_col='CaseName').index.tolist()
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
                image = torch.cat([inputs[:, 9: 12], inputs[:, -2:]], dim=1)
                roi = torch.zeros_like(image)
                image = MoveTensorsToDevice(image, device)
                roi = MoveTensorsToDevice(roi, device)
                outputs = MoveTensorsToDevice(outputs, device)

                with torch.no_grad():
                    for slice in range(image.shape[1]):
                        roi[:, slice:slice+1] = torch.sigmoid(seg_model(image[:, slice:slice+1]))
                roi[roi >= 0.5] = 1
                roi[roi < 0.5] = 0
                dilate_roi = torch.from_numpy(binary_dilation(roi.cpu().detach().numpy(),
                                                              structure=np.ones((1, 1, 11, 11)))).to(device)

                preds = model(image*dilate_roi)

                optimizer.zero_grad()

                loss = ce_loss(preds, outputs.long())
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                train_pred.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().detach())
                train_label.extend(outputs.int().cpu().detach())

            train_acc = sum([1 for index in range(len(train_label)) if train_label[index] == train_pred[index]]) / len(
                train_label)

            model.eval()
            with torch.no_grad():
                for ind, (inputs, outputs) in enumerate(val_loader):
                    image = torch.cat([inputs[:, 9: 12], inputs[:, -2:]], dim=1)
                    roi = torch.zeros_like(image)
                    image = MoveTensorsToDevice(image, device)
                    roi = MoveTensorsToDevice(roi, device)
                    outputs = MoveTensorsToDevice(outputs, device)

                    with torch.no_grad():
                        for slice in range(image.shape[1]):
                            roi[:, slice:slice + 1] = torch.sigmoid(seg_model(image[:, slice:slice + 1]))

                    roi[roi >= 0.5] = 1
                    roi[roi < 0.5] = 0
                    dilate_roi = torch.from_numpy(binary_dilation(roi.cpu().detach().numpy(),
                                                                  structure=np.ones((1, 1, 11, 11)))).to(device)

                    preds = model(image * dilate_roi)

                    loss = ce_loss(preds, outputs.long())

                    val_loss += loss.item()

                    val_pred.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().detach())
                    val_label.extend(outputs.int().cpu().detach())

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



if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/RenJi/Model2D'
    data_root = r'/home/zhangyihong/Documents/RenJi/CaseWithROI'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # EnsembleTrain(device, model_root, 'ResNeXt_0921_5slice_cv_2cl', data_root)
    EnsembleTrainBySeg(device, model_root, 'ResNeXt_0922_5slice_cv_2cl', data_root)
