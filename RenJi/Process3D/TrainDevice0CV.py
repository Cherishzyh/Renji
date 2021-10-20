import os
import shutil
from scipy.ndimage import binary_dilation

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from CnnTools.T4T.Utility.Data import *
from CnnTools.T4T.Utility.CallBacks import EarlyStopping
from CnnTools.T4T.Utility.Initial import HeWeightInit

from BasicTool.MeDIT.Others import MakeFolder, CopyFile
from BasicTool.MeDIT.Augment import *

from RenJi.Network3D.ResNet3D import GenerateModel
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


def _GetCV(label_root, cv):
    all_label_df = os.path.join(label_root, 'alltrain_name.csv')
    all_case = pd.read_csv(all_label_df, index_col='CaseName')
    all_case = all_case.index.tolist()

    cv_label = os.path.join(label_root, 'train-cv{}.csv'.format(cv))
    cv_val = pd.read_csv(cv_label)
    cv_val = cv_val.loc[:, 'CaseName'].tolist()

    cv_train = [case for case in all_case if case not in cv_val]

    return cv_train, cv_val


def EnsembleTrain(device, model_root, model_name, data_root):
    torch.autograd.set_detect_anomaly(True)

    device = device
    input_shape = (150, 150)
    total_epoch = 10000
    batch_size = 12
    model_folder = os.path.join(model_root, model_name)
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

    for cv_index in range(1, 6):
        sub_model_folder = os.path.join(model_folder, 'CV_{}'.format(cv_index))
        sub_train, sub_val = _GetCV(data_root, cv_index)
        train_loader, train_batches = _GetLoader(data_root, sub_train, param_config, input_shape, batch_size, True)
        val_loader, val_batches = _GetLoader(data_root, sub_val, param_config, input_shape, batch_size, True)

        # no softmax or sigmoid
        model = GenerateModel(50, n_input_channels=1, n_classes=2).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        ce_loss = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)
        early_stopping = EarlyStopping(store_path=str(Path(sub_model_folder) / '{}-{:.6f}.pt'), patience=50, verbose=True)
        writer = SummaryWriter(log_dir=str(Path(sub_model_folder) / 'log'), comment='Net')

        for epoch in range(total_epoch):
            train_loss, val_loss = 0., 0.
            train_pred, val_pred = [], []
            train_label, val_label = [], []

            model.train()
            for ind, (inputs, outputs) in enumerate(train_loader):
                image = inputs[0] * inputs[1]
                image = torch.unsqueeze(image, dim=1)
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
                    image = torch.unsqueeze(image, dim=1)
                    outputs[outputs <= 1] = 0
                    outputs[outputs >= 2] = 1

                    image = MoveTensorsToDevice(image, device)
                    outputs = MoveTensorsToDevice(outputs, device)

                    preds = model(image)

                    loss = ce_loss(preds, outputs.long())

                    val_loss += loss.item()

                    val_pred.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().detach().numpy().tolist())
                    val_label.extend(outputs.cpu().data.numpy().astype(int).squeeze().tolist())

            val_acc = sum([1 for index in range(len(val_label)) if val_label[index] == val_pred[index]]) / len(val_label)

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

        model = GenerateModel(50, n_input_channels=1, n_classes=2).to(device)
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
                roi = torch.zeros_like(inputs)
                inputs = MoveTensorsToDevice(inputs, device)
                roi = MoveTensorsToDevice(roi, device)
                outputs = MoveTensorsToDevice(outputs, device)

                with torch.no_grad():
                    for slice in range(inputs.shape[1]):
                        roi[:, slice:slice+1] = torch.sigmoid(seg_model(inputs[:, slice:slice+1]))
                roi[roi >= 0.5] = 1
                roi[roi < 0.5] = 0
                dilate_roi = torch.from_numpy(binary_dilation(roi.cpu().detach().numpy(),
                                                              structure=np.ones((1, 1, 11, 11)))).to(device)

                preds = model(torch.unsqueeze(inputs*dilate_roi, dim=1))

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
                    roi = torch.zeros_like(inputs)
                    inputs = MoveTensorsToDevice(inputs, device)
                    roi = MoveTensorsToDevice(roi, device)
                    outputs = MoveTensorsToDevice(outputs, device)

                    with torch.no_grad():
                        for slice in range(inputs.shape[1]):
                            roi[:, slice:slice + 1] = torch.sigmoid(seg_model(inputs[:, slice:slice + 1]))

                    roi[roi >= 0.5] = 1
                    roi[roi < 0.5] = 0
                    dilate_roi = torch.from_numpy(binary_dilation(roi.cpu().detach().numpy(),
                                                                  structure=np.ones((1, 1, 11, 11)))).to(device)

                    preds = model(torch.unsqueeze(inputs*dilate_roi, dim=1))

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
    model_root = r'/home/zhangyihong/Documents/RenJi/Model'
    data_root = r'/home/zhangyihong/Documents/RenJi/CaseWithROI'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # EnsembleTrain(device, model_root, 'ResNet3D_0921_mask_cv_2cl', data_root)
    EnsembleTrainBySeg(device, model_root, 'ResNet3D_0922_mask_cv_2cl', data_root)
