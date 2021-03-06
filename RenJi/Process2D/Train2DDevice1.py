import shutil
from scipy.ndimage import binary_dilation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from T4T.Utility.Data import *
from T4T.Utility.CallBacks import EarlyStopping
from T4T.Utility.Initial import HeWeightInit

from MeDIT.Others import MakeFolder, CopyFile
from MeDIT.Augment import *

from RenJi.Network3D.MergeResNet3D import GenerateModel
from RenJi.SegModel2D.UNet import UNet


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(data_root, sub_list, aug_param_config, input_shape, batch_size, shuffle, is_balance=True):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)

    data.AddOne(Image2D(data_root + '/2CHNPY', shape=input_shape))
    data.AddOne(Image2D(data_root + '/3CHNPY', shape=input_shape))

    data.AddOne(Label(data_root + '/label_2cl.csv'), is_input=False)
    if is_balance:
        data.Balance(Label(data_root + '/label_2cl.csv'))

    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=36, pin_memory=True)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def Train(device, model_root, model_name, data_root):
    torch.autograd.set_detect_anomaly(True)

    input_shape = (200, 200)
    total_epoch = 10000
    batch_size = 24
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
    val_loader, val_batches = _GetLoader(data_root, val_case, param_config, input_shape, batch_size, True, False)

    model = resnet50(input_channels=5, num_classes=4).to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
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
            inputs = torch.cat([inputs[:, 10: 13], inputs[:, -2:]], dim=1)

            inputs = MoveTensorsToDevice(inputs, device)
            outputs = MoveTensorsToDevice(outputs, device)

            preds = model(inputs)

            loss = ce_loss(preds, outputs.long())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_pred.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().data.numpy().tolist())
            train_label.extend(outputs.cpu().data.numpy().astype(int).squeeze().tolist())

        train_acc = sum([1 for index in range(len(train_label)) if train_label[index] == train_pred[index]]) / len(train_label)

        model.eval()
        with torch.no_grad():
            for ind, (inputs, outputs) in enumerate(val_loader):
                inputs = torch.cat([inputs[:, 10: 13], inputs[:, -2:]], dim=1)
                inputs = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(outputs, device)

                preds = model(inputs)

                loss = ce_loss(preds, outputs.long())

                val_loss += loss.item()

                val_pred.extend(torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().data.numpy().tolist())
                val_label.extend(outputs.cpu().data.numpy().astype(int).squeeze().tolist())

        val_acc = sum([1 for index in range(len(val_label)) if val_label[index] == val_pred[index]]) / len(val_label)

        # Save Tensor Board
        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Loss', {'train_loss': train_loss / train_batches, 'val_loss': val_loss / val_batches}, epoch + 1)
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

def EnsembleTrainBySeg(device, model_root, model_name, data_root):
    torch.autograd.set_detect_anomaly(True)

    device = device
    input_shape = (150, 150)
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

    seg_model_path_1 = r'/home/zhangyihong/Documents/RenJi/SegModel/UNet_1026_mix23/70-0.302015.pt'
    seg_model = UNet(in_channels=1, out_channels=1).to(device)
    seg_model.load_state_dict(torch.load(str(seg_model_path_1)))
    seg_model.eval()

    alltrain_list = pd.read_csv(os.path.join(data_root, 'non_alltrain_name.csv'), index_col='CaseName').index.tolist()
    for cv_index in range(1, 6):
        sub_model_folder = MakeFolder(model_folder / 'CV_{}'.format(cv_index))
        sub_val = pd.read_csv(os.path.join(data_root, 'non_train-cv{}.csv'.format(cv_index)),
                              index_col='CaseName').index.tolist()
        sub_train = [case for case in alltrain_list if case not in sub_val]
        train_loader, train_batches = _GetLoader(data_root, sub_train, param_config, input_shape, batch_size, True)
        val_loader, val_batches = _GetLoader(data_root, sub_val, None, input_shape, batch_size, True)

        # no softmax or sigmoid
        model = GenerateModel(50, n_input_channels=1, n_classes=1).to(device)
        model.apply(HeWeightInit)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        bce_loss = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5,
                                                               verbose=True)
        early_stopping = EarlyStopping(store_path=str(Path(sub_model_folder) / '{}-{:.6f}.pt'), patience=50,
                                       verbose=True)
        writer = SummaryWriter(log_dir=str(Path(sub_model_folder) / 'log'), comment='Net')

        for epoch in range(total_epoch):
            train_loss, val_loss = 0., 0.

            model.train()
            for ind, (inputs, outputs) in enumerate(train_loader):
                image = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(outputs, device)

                with torch.no_grad():
                    roi = torch.zeros_like(image[0]).to(device)
                    for slice in range(image[0].shape[1]):
                        roi[:, slice:slice + 1] = torch.sigmoid(seg_model(image[0][:, slice:slice + 1]))
                        roi[roi >= 0.5] = 1
                        roi[roi < 0.5] = 0
                        dilate_roi_2ch = torch.from_numpy(binary_dilation(roi.cpu().detach().numpy(),
                                                                          structure=np.ones((1, 1, 11, 11)))).to(device)
                    roi = torch.zeros_like(image[1]).to(device)
                    for slice in range(image[1].shape[1]):
                        roi[:, slice:slice + 1] = torch.sigmoid(seg_model(image[1][:, slice:slice + 1]))
                        roi[roi >= 0.5] = 1
                        roi[roi < 0.5] = 0
                        dilate_roi_3ch = torch.from_numpy(binary_dilation(roi.cpu().detach().numpy(),
                                                                          structure=np.ones((1, 1, 11, 11)))).to(device)
                image_2ch = image[0] * dilate_roi_2ch
                image_2ch = torch.unsqueeze(image_2ch, dim=1)
                image_3ch = image[1] * dilate_roi_3ch
                image_3ch = torch.unsqueeze(image_3ch, dim=1)

                preds = model(image_2ch, image_3ch)

                optimizer.zero_grad()
                loss = bce_loss(torch.squeeze(preds), outputs)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            model.eval()
            with torch.no_grad():
                for ind, (inputs, outputs) in enumerate(val_loader):
                    image = MoveTensorsToDevice(inputs, device)
                    outputs = MoveTensorsToDevice(outputs, device)

                    with torch.no_grad():
                        roi = torch.zeros_like(image[0]).to(device)
                        for slice in range(image[0].shape[1]):
                            roi[:, slice:slice + 1] = torch.sigmoid(seg_model(image[0][:, slice:slice + 1]))
                            roi[roi >= 0.5] = 1
                            roi[roi < 0.5] = 0
                            dilate_roi_2ch = torch.from_numpy(binary_dilation(roi.cpu().detach().numpy(),
                                                                              structure=np.ones((1, 1, 11, 11)))).to(
                                device)
                        roi = torch.zeros_like(image[1]).to(device)
                        for slice in range(image[1].shape[1]):
                            roi[:, slice:slice + 1] = torch.sigmoid(seg_model(image[1][:, slice:slice + 1]))
                            roi[roi >= 0.5] = 1
                            roi[roi < 0.5] = 0
                            dilate_roi_3ch = torch.from_numpy(binary_dilation(roi.cpu().detach().numpy(),
                                                                              structure=np.ones((1, 1, 11, 11)))).to(
                                device)
                    image_2ch = image[0] * dilate_roi_2ch
                    image_2ch = torch.unsqueeze(image_2ch, dim=1)
                    image_3ch = image[1] * dilate_roi_3ch
                    image_3ch = torch.unsqueeze(image_3ch, dim=1)

                    preds = model(image_2ch, image_3ch)

                    loss = bce_loss(torch.squeeze(preds), outputs)

                    val_loss += loss.item()

            for index, (name, param) in enumerate(model.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name + '_data', param.cpu().detach().numpy(), epoch + 1)

            writer.add_scalars('Loss', {'train_loss': train_loss / train_batches, 'val_loss': val_loss / val_batches},
                               epoch + 1)

            print('Epoch {}:\tloss: {:.3f}, val-loss: {:.3f}'.format(
                epoch + 1, train_loss / train_batches, val_loss / val_batches))

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
    data_root = r'/home/zhangyihong/Documents/RenJi'
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    # EnsembleTrain(device, model_root, 'ResNet3D_0921_mask_cv_2cl', data_root)
    EnsembleTrainBySeg(device, model_root, 'ResNet3D_1027_mask_cv_2cl_2', data_root)