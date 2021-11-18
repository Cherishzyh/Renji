import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from scipy import ndimage
from scipy.ndimage import binary_dilation
from scipy.ndimage import median_filter

from MeDIT.Others import IterateCase
from T4T.Utility.Data import *

from RenJi.SegModel2D.UNet import UNet
from RenJi.Metric.ROC import Dice, Dice4Numpy


class InferenceByCase():
    def __init__(self):
        super().__init__()

    def KeepLargest(self, mask):
        new_mask = np.zeros(mask.shape)
        label_im, nb_labels = ndimage.label(mask)
        max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
        new_mask[label_im == np.argmax(max_volume) + 1] = 1
        return new_mask

    def CropData(self, data, crop_shape, center, is_roi=False, slice_num=1):
        from MeDIT.ArrayProcess import ExtractPatch
        from MeDIT.ArrayProcess import ExtractBlock

        # Crop
        if len(data.shape) == 2:
            t2_crop, _ = ExtractPatch(data, crop_shape, center_point=center)
        elif len(data.shape) == 3:
            center = [center[0], center[1], -1]
            crop_shape = [crop_shape[0], crop_shape[1], slice_num]
            t2_crop, _ = ExtractBlock(data, crop_shape, center_point=center)
        else:
            raise Exception

        if not is_roi:
            # Normalization
            t2_crop -= np.mean(t2_crop)
            t2_crop /= np.std(t2_crop)

        return t2_crop

    def GetCenter(self, roi):
        roi = np.squeeze(roi)
        if np.ndim(roi) == 3:
            roi = roi[0]
        non_zero = np.nonzero(roi)
        center_x = int(np.median(np.unique(non_zero[1])))
        center_y = int(np.median(np.unique(non_zero[0])))
        return (center_x, center_y)

    def NPY2NPY(self, case, data_path):
        # ch2_path = os.path.join(data_path, '2CHNPYNorm/{}'.format(case))
        # ch3_path = os.path.join(data_path, '3CHNPY/{}'.format(case))
        ch2_label_path = os.path.join(data_path, '2CHROINPY/{}'.format(case))
        ch3_label_path = os.path.join(data_path, '3CHROINPY/{}'.format(case))
        ch2_path = os.path.join(data_path, '2CHNPY_5Slice/{}'.format(case))
        ch3_path = os.path.join(data_path, '3CHNPY_5Slice/{}'.format(case))


        if os.path.exists(ch2_label_path):
            ch2_label_arr = np.squeeze(np.load(ch2_label_path))
        else:
            ch2_label_arr = None

        if os.path.exists(ch3_label_path):
            ch3_label_arr = np.squeeze(np.load(ch3_label_path))
        else:
            ch3_label_arr = None

        if os.path.exists(ch2_path):
            ch2_arr = np.load(ch2_path)
        else:
            ch2_arr = None

        if os.path.exists(ch3_path):
            ch3_arr = np.load(ch3_path)
        else:
            ch3_arr = None

        return ch2_arr, ch3_arr, ch2_label_arr, ch3_label_arr

    def run(self, case, ch2_data, ch3_data, model, is_save=False):

        if isinstance(ch2_data, np.ndarray):
            ch2_data = torch.from_numpy(ch2_data[:, np.newaxis])
        if isinstance(ch3_data, np.ndarray):
            ch3_data = torch.from_numpy(ch3_data[:, np.newaxis])

        ch2_data = MoveTensorsToDevice(ch2_data, device)
        ch3_data = MoveTensorsToDevice(ch3_data, device)

        with torch.no_grad():
            preds_2ch = model(ch2_data)
            preds_2ch = torch.squeeze(torch.sigmoid(preds_2ch)).cpu().detach().numpy()
            preds_2ch[preds_2ch >= 0.5] = 1
            preds_2ch[preds_2ch < 0.5] = 0
            # preds_2ch = self.KeepLargest(preds_2ch)
            # preds_2ch = median_filter(preds_2ch, size=(1, 3, 3))
            preds_3ch = model(ch3_data)
            preds_3ch = torch.squeeze(torch.sigmoid(preds_3ch)).cpu().detach().numpy()
            preds_3ch[preds_3ch >= 0.5] = 1
            preds_3ch[preds_3ch < 0.5] = 0
            # preds_3ch = self.KeepLargest(preds_3ch)
            # preds_3ch = median_filter(preds_3ch, size=(1, 3, 3))

        if is_save:
            if not os.path.exists(os.path.join(data_root, '2CHPredROI')):
                os.mkdir(os.path.join(data_root, '2CHPredROI'))
            if not os.path.exists(os.path.join(data_root, '2CHPredROIDilated')):
                os.mkdir(os.path.join(data_root, '2CHPredROIDilated'))
            if not os.path.exists(os.path.join(data_root, '3CHPredROI')):
                os.mkdir(os.path.join(data_root, '3CHPredROI'))
            if not os.path.exists(os.path.join(data_root, '3CHPredROIDilated')):
                os.mkdir(os.path.join(data_root, '3CHPredROIDilated'))

            np.save(os.path.join(data_root, '2CHPredROI/{}'.format(case)), preds_2ch)
            np.save(os.path.join(data_root, '2CHPredROIDilated/{}'.format(case)),
                    np.squeeze(binary_dilation(preds_2ch, structure=np.ones((1, 11, 11)))))
            np.save(os.path.join(data_root, '3CHPredROI/{}'.format(case)), preds_3ch)
            np.save(os.path.join(data_root, '3CHPredROIDilated/{}'.format(case)),
                    np.squeeze(binary_dilation(preds_3ch, structure=np.ones((1, 11, 11)))))

        return preds_2ch, preds_3ch


def ShowFlatten(data, roi, pred, save_path=r''):
    data_flatten = FlattenImages(data, is_show=False)
    pred_flatten = FlattenImages(pred, is_show=False)

    plt.close()
    plt.figure(figsize=(8, 8), dpi=100)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(data_flatten, cmap='gray')
    if isinstance(roi, np.ndarray):
        roi_flatten = FlattenImages(roi, is_show=False)
        plt.contour(roi_flatten, colors='r')
    plt.contour(pred_flatten, colors='y')
    if save_path:
        plt.savefig(save_path, pad_inches=0)
        # plt.show()
    plt.close()



if __name__ == '__main__':
    from MeDIT.Visualization import FlattenImages
    Seg = InferenceByCase()

    data_root = r'/home/zhangyihong/Documents/RenJi/ExternalTest'
    model_folder = r'/home/zhangyihong/Documents/RenJi/SegModel'
    save_folder = r'/home/zhangyihong/Documents/RenJi/SegModel/UNet_1026_mix23_use/ExternalImage'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    model_path = os.path.join(model_folder, 'UNet_1026_mix23_use')
    weights_list = None
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=1).to(device)
    if weights_list is None:
        weights_list = [one for one in IterateCase(model_path, only_folder=False, verbose=0) if one.is_file()]
        weights_list = [one for one in weights_list if str(one).endswith('.pt')]
        if len(weights_list) == 0:
            raise Exception
        weights_list = sorted(weights_list, key=lambda x: os.path.getctime(str(x)))
        weights_path = weights_list[-1]
    else:
        weights_path = weights_list

    weights_path = os.path.join(model_path, weights_path)
    model.to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    data_num = len(os.listdir(r'/home/zhangyihong/Documents/RenJi/ExternalTest/2CHNPY_5Slice'))
    for index, case in enumerate(sorted(os.listdir(r'/home/zhangyihong/Documents/RenJi/ExternalTest/2CHNPY_5Slice'))):
        print('********************** {} / {} | predicting {} **********************'.format(index+1, data_num, case.split('.npy')[0]))
        ch2_arr, ch3_arr, ch2_label_arr, ch3_label_arr = Seg.NPY2NPY(case, data_root)

        preds_2ch, preds_3ch = Seg.run(case, ch2_arr, ch3_arr, model, is_save=True)

        # ShowFlatten(ch2_arr, ch2_label_arr, preds_2ch, save_path=os.path.join(save_folder, '2ch_{}.jpg'.format(case.split('.npy')[0])))
        # ShowFlatten(ch3_arr, ch3_label_arr, preds_3ch, save_path=os.path.join(save_folder, '3ch_{}.jpg'.format(case.split('.npy')[0])))


