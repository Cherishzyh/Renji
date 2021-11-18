import os
import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from scipy import ndimage
import scipy.signal as signal

from T4T.Utility.Data import *

from RenJi.SuccessfulModel.ConfigInterpretor import BaseImageOutModel


class InferenceByCase(BaseImageOutModel):
    def __init__(self):
        super(InferenceByCase).__init__()

    def __KeepLargest(self, mask):
        new_mask = np.zeros(mask.shape)
        label_im, nb_labels = ndimage.label(mask)
        max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
        new_mask[label_im == np.argmax(max_volume) + 1] = 1
        return new_mask

    def run(self, data, save_nii=r'', save_jpg=r''):
        resolution = self.config.GetResolution()
        resolution = [1.0, resolution[0], resolution[1]]

        data_cropped = self.config.CropDataShape(data, resolution)

        with torch.no_grad():
            inputs = torch.from_numpy(data_cropped[:, np.newaxis, ...]).to(self.device)
            preds_list = self.model(inputs)

        pred = torch.sigmoid(preds_list).cpu().detach().numpy()
        pred = np.squeeze(pred)
        pred = self.config.RecoverDataShape(pred, resolution)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = self.__KeepLargest(pred)
        pred = signal.medfilt(pred, 5)

        if save_jpg:
            data_flatten = FlattenImages(data, is_show=False)
            pred_flatten = FlattenImages(pred, is_show=False)
            plt.figure(figsize=(8, 8), dpi=100)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.imshow(data_flatten, cmap='gray', vmin=0.)
            plt.contour(pred_flatten, colors='y')
            plt.savefig(save_jpg, pad_inches=0)
            plt.close()

        if save_nii:
            data_image = sitk.GetImageFromArray(data)
            data_image.SetSpacing(resolution)
            mask_image = sitk.GetImageFromArray(pred)
            mask_image.SetSpacing(resolution)

            sitk.WriteImage(data_image, os.path.join(save_nii, 'Image.nii.gz'))
            sitk.WriteImage(mask_image, os.path.join(save_nii, 'Mask.nii.gz'))
        return pred


if __name__ == '__main__':
    from MeDIT.Visualization import FlattenImages
    from pathlib import Path
    model_folder = r'/home/zhangyihong/Documents/RenJi/SuccessfulModel/UNet_1118'
    pred_folder = r'/home/zhangyihong/Documents/RenJi/Data/PredData'

    root_folder = Path(r'/home/zhangyihong/Documents/RenJi/Data/ClassData/2CHNPY')

    segmentor = InferenceByCase()
    segmentor.LoadConfigAndModel(model_folder)

    nii_folder = os.path.join(pred_folder, '2CH')
    jpg_folder = os.path.join(pred_folder, 'Image')
    if not os.path.exists(nii_folder): os.mkdir(nii_folder)
    if not os.path.exists(jpg_folder): os.mkdir(jpg_folder)

    for case in sorted(root_folder.iterdir()):
        pred_folder = os.path.join(nii_folder, str(case.name).split('.npy')[0])
        if not os.path.exists(pred_folder): os.mkdir(pred_folder)

        data = np.load(os.path.join(root_folder, case))

        segmentor.run(data, save_nii=pred_folder,
                      save_jpg=os.path.join(jpg_folder, '{}.jpg'.format(str(case.name).split('.npy')[0])))

        print('successful predict {}'.format(str(case.name).split('.npy')[0]))

