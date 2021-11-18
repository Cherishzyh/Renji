import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import binary_dilation

from MeDIT.Others import IterateCase
from T4T.Utility.Data import *

from RenJi.SegModel2D.UNet import UNet
from RenJi.Metric.ROC import Dice, Dice4Numpy
from Tools import KeepLargest


def EnsembleInference(model_root, data_root, model_name, data_type, weights_list=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (150, 150)
    batch_size = 24
    model_folder = os.path.join(model_root, model_name)

    all_list = pd.read_csv(os.path.join(data_root, '{}_name.csv'.format(data_type)), index_col='CaseName').index.tolist()

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []
    for cv_index, cv_folder in enumerate(cv_folder_list):
        if data_type == 'val':
            sub_list = pd.read_csv(os.path.join(data_root, 'train-cv{}.csv'.format(cv_index+1)), index_col='CaseName').index.tolist()
        elif data_type == 'train':
            sub_val = pd.read_csv(os.path.join(data_root, 'train-cv{}.csv'.format(cv_index + 1)), index_col='CaseName').index.tolist()
            sub_list = [case for case in all_list if case not in sub_val]
        else:
            sub_list = all_list

        data = DataManager(sub_list=sub_list)

        data.AddOne(Image2D(data_root + '/NPY', shape=input_shape))
        data.AddOne(Image2D(data_root + '/RoiNPY', shape=input_shape, is_roi=True), is_input=False)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=36, pin_memory=True)

        model = UNet(in_channels=1, out_channels=1).to(device)
        if weights_list is None:
            one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if one.is_file()]
            one_fold_weights_list = [one for one in one_fold_weights_list if str(one).endswith('.pt')]
            one_fold_weights_list = sorted(one_fold_weights_list,  key=lambda x: os.path.getctime(str(x)))
            weights_path = one_fold_weights_list[-1]
        else:
            weights_path = weights_list[cv_index]

        print(weights_path.name, end='\t')
        model.load_state_dict(torch.load(str(weights_path)))

        t2_list, pred_list, label_list = [], [], []
        model.eval()
        for inputs, outputs in data_loader:
            image = inputs[:, :1]
            outputs = outputs[:, :1]
            t2_list.append(image)
            image = MoveTensorsToDevice(image, device)

            preds = model(image)

            pred_list.append(torch.sigmoid(preds).cpu().detach())
            label_list.append(outputs)


        cv_pred_list.append(torch.cat(pred_list, dim=0))
        cv_label_list.append(torch.cat(label_list, dim=0))
        np.save(os.path.join(cv_folder, '{}_preds.npy'.format(data_type)), torch.cat(pred_list, dim=0).numpy())
        np.save(os.path.join(cv_folder, '{}_label.npy'.format(data_type)), torch.cat(label_list, dim=0).numpy())
        np.save(os.path.join(cv_folder, '{}_image.npy'.format(data_type)), torch.cat(t2_list, dim=0).numpy())

        cv_dice = np.mean([Dice(torch.cat(pred_list, dim=0)[index], torch.cat(label_list, dim=0)[index]).numpy()
                        for index in range(torch.cat(pred_list, dim=0).shape[0])])
        print(float('{:.3f}'.format(cv_dice)))

        del model, weights_path

    if data_type == 'alltrain' or data_type == 'test':
        cv_pred = torch.cat(cv_pred_list, dim=1)
        cv_label = torch.cat(cv_label_list, dim=1)
        mean_pred = torch.mean(cv_pred, dim=1, keepdim=False)
        mean_label = torch.mean(cv_label, dim=1, keepdim=False)
        np.save(os.path.join(model_folder, '{}_preds.npy'.format(data_type)), mean_pred.numpy())
        np.save(os.path.join(model_folder, '{}_label.npy'.format(data_type)), mean_label.numpy())

        dice = np.mean([Dice(mean_pred[index], mean_label[index]).numpy() for index in range(mean_pred.shape[0])])
        print(float('{:.3f}'.format(dice)))

        return mean_label, mean_pred


def Inference(model_root, data_root, model_name, data_type):
    input_shape = (200, 200)
    batch_size = 12
    model_folder = os.path.join(model_root, model_name)

    sub_list = pd.read_csv(os.path.join(data_root, 'normal_{}.csv'.format(data_type)), index_col='CaseName').index.tolist()
    sub_list = ['2ch_{}'.format(case) for case in sub_list] + ['3ch_{}'.format(case) for case in sub_list]
    # sub_list = ['2ch_{}'.format(case) for case in sub_list]

    data = DataManager(sub_list=sub_list)

    data.AddOne(Image2D(data_root + '/NPY', shape=input_shape))
    data.AddOne(Image2D(data_root + '/RoiNPY', shape=input_shape, is_roi=True), is_input=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=36, pin_memory=True)

    model = UNet(in_channels=1, out_channels=1).to(device)

    one_fold_weights_list = [one for one in IterateCase(model_folder, only_folder=False, verbose=0) if one.is_file()]
    one_fold_weights_list = [one for one in one_fold_weights_list if str(one).endswith('.pt')]
    one_fold_weights_list = sorted(one_fold_weights_list,  key=lambda x: os.path.getctime(str(x)))
    weights_path = one_fold_weights_list[-1]

    print(weights_path.name, end='\t')
    model.load_state_dict(torch.load(str(weights_path)))

    t2_list, pred_list, label_list = [], [], []
    model.eval()
    for inputs, outputs in data_loader:
        image = inputs[:, :1]
        outputs = outputs[:, :1]
        t2_list.append(image)
        image = MoveTensorsToDevice(image, device)

        preds = model(image)

        pred_list.append(torch.sigmoid(preds).cpu().detach())
        label_list.append(outputs)

    np.save(os.path.join(model_folder, '{}_preds.npy'.format(data_type)), torch.cat(pred_list, dim=0).numpy())
    np.save(os.path.join(model_folder, '{}_label.npy'.format(data_type)), torch.cat(label_list, dim=0).numpy())
    np.save(os.path.join(model_folder, '{}_image.npy'.format(data_type)), torch.cat(t2_list, dim=0).numpy())

    dice = np.mean([Dice(torch.cat(pred_list, dim=0)[index], torch.cat(label_list, dim=0)[index]) for index in range(torch.cat(label_list, dim=0).shape[0])])
    print(float('{:.3f}'.format(dice)))


def ShowHist(model_folder, data_type='test'):
    label = np.load(os.path.join(model_folder, '{}_label.npy'.format(data_type)))
    preds = np.load(os.path.join(model_folder, '{}_preds.npy'.format(data_type)))

    dice_list = [Dice4Numpy(preds[index], label[index]) for index in range(preds.shape[0])]
    print('mean_dice: {}'.format(sum(dice_list)/len(dice_list)))

    plt.hist(dice_list, bins=50)
    plt.show()


def Visualization(npy_folder, data_type='test'):
    label = np.squeeze(np.load(os.path.join(npy_folder, '{}_label.npy'.format(data_type))))
    preds = np.squeeze(np.load(os.path.join(npy_folder, '{}_preds.npy'.format(data_type))))
    image = np.squeeze(np.load(os.path.join(npy_folder, '{}_image.npy'.format(data_type))))

    dice_list = [Dice4Numpy(preds[index], label[index]) for index in range(preds.shape[0])]
    print(sum(dice_list) / len(dice_list))

    sort_index = np.argsort(np.array(dice_list))
    for index in [sort_index[0], sort_index[sort_index.shape[0]//2], sort_index[-1]]:
        print('Dice: {:.3f}'.format(Dice4Numpy(preds[index], label[index])))
        binary_pred = deepcopy(preds)
        binary_pred[binary_pred >= 0.5] = 1
        binary_pred[binary_pred < 0.5] = 0
        plt.figure(figsize=(8, 8))
        plt.imshow(image[index], cmap='gray')
        plt.contour(label[index], colors='r')
        plt.contour(binary_pred[index], colors='y')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.show()
        plt.close()


def InferenceByCase(model_root, data_root, model_name, data_type):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (200, 200)
    batch_size = 1
    model_folder = os.path.join(model_root, model_name)

    sub_list = pd.read_csv(os.path.join(data_root, '{}_name.csv'.format(data_type)), index_col='CaseName').index.tolist()

    data = DataManager(sub_list=sub_list)

    data.AddOne(Image2D(data_root + '/Example/NPY', shape=input_shape))
    data.AddOne(Image2D(data_root + '/Example/RoiNPY_Dilation', shape=input_shape, is_roi=True), is_input=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=36, pin_memory=True)

    model = UNet(in_channels=1, out_channels=1).to(device)

    one_fold_weights_list = [one for one in IterateCase(model_folder, only_folder=False, verbose=0) if one.is_file()]
    one_fold_weights_list = [one for one in one_fold_weights_list if str(one).endswith('.pt')]
    one_fold_weights_list = sorted(one_fold_weights_list,  key=lambda x: os.path.getctime(str(x)))
    weights_path = one_fold_weights_list[-1]

    print(weights_path.name, end='\t')
    model.load_state_dict(torch.load(str(weights_path)))
    model.eval()

    t2_list, pred_list, label_list = [], [], []
    with torch.no_grad():
        for inputs, outputs in data_loader:
            t2_list.append(torch.squeeze(inputs))
            inputs = MoveTensorsToDevice(inputs, device)
            preds = torch.zeros_like(inputs)
            for slice in range(inputs.shape[1]):
                preds[:, slice:slice + 1] = torch.sigmoid(model(inputs[:, slice:slice + 1]))

        pred_list.append(torch.squeeze(preds).cpu().detach())
        label_list.append(torch.squeeze(outputs))

    np.save(os.path.join(model_folder, '{}_preds.npy'.format(data_type)), pred_list[0].numpy())
    np.save(os.path.join(model_folder, '{}_label.npy'.format(data_type)), label_list[0].numpy())
    np.save(os.path.join(model_folder, '{}_image.npy'.format(data_type)), t2_list[0].numpy())

    dice = Dice(pred_list[0], label_list[0])
    print(float('{:.3f}'.format(dice)))


if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/RenJi/SegModel'
    data_root = r'/home/zhangyihong/Documents/RenJi/CaseWithROI'
    model_name = 'UNet_1026_mix23_use'

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    #
    # Inference(model_root, data_root, model_name, data_type='train')
    # print()
    # Inference(model_root, data_root, model_name, data_type='val')
    # print()
    # Inference(model_root, data_root, model_name, data_type='test')
    #
    # ShowHist(os.path.join(model_root, model_name), data_type='train')
    # ShowHist(os.path.join(model_root, model_name), data_type='val')
    # ShowHist(os.path.join(model_root, model_name), data_type='test')
    # Visualization(os.path.join(model_root, model_name), data_type='test')

    # Inference(model_root, data_root, model_name, data_type='non_alltrain')
    # ShowHist(os.path.join(model_root, model_name), data_type='non_alltrain')
    # Visualization(os.path.join(model_root, model_name), data_type='non_alltrain')
    # InferenceByCase(model_root, data_root, model_name, 'non_alltrain')
    #
    # os.mkdir(os.path.join(model_root, '{}/Image'.format(model_name)))
    # os.mkdir(os.path.join(model_root, '{}/Image/Train'.format(model_name)))
    # os.mkdir(os.path.join(model_root, '{}/Image/Val'.format(model_name)))
    # os.mkdir(os.path.join(model_root, '{}/Image/Test'.format(model_name)))
    # name = ['Train', 'Val', 'Test']
    # for index, type in enumerate(['train', 'val', 'test']):
    #     label = np.squeeze(np.load(os.path.join(os.path.join(model_root, model_name), '{}_label.npy'.format(type))))
    #     preds = np.squeeze(np.load(os.path.join(os.path.join(model_root, model_name), '{}_preds.npy'.format(type))))
    #     image = np.squeeze(np.load(os.path.join(os.path.join(model_root, model_name), '{}_image.npy'.format(type))))
    #     preds[preds >= 0.5] = 1
    #     preds[preds < 0.5] = 0
    #     preds = KeepLargest(preds)
    #
    #     for slice in range(label.shape[0]):
    #
    #         plt.figure(figsize=(8, 8))
    #         plt.imshow(image[slice], cmap='gray')
    #         plt.contour(preds[slice], colors='y')
    #         plt.contour(label[slice], colors='r')
    #         plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #         plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #         plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #         plt.savefig(os.path.join(os.path.join(os.path.join(model_root, '{}/Image/{}'.format(model_name, name[index])), '{}.jpg'.format(slice))))
    #         # plt.show()
    #         plt.close()
    #
    Inference(model_root, data_root, model_name, data_type='external')
