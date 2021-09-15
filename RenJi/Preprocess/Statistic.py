import os
import numpy as np
import SimpleITK as sitk

from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01


def MaxRowCol():
    folder = r'Z:\RenJi\LVA finished 2CH_MASK 20210910\LVA finished 2CH_MASK'
    max_row = []
    max_col = []
    square_list = []
    for case in sorted(os.listdir(folder)):
        case_folder = os.path.join(folder, case)
        mask_path = os.path.join(case_folder, 'mask.nii.gz')
        mask = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(mask_path)))
        if np.ndim(mask) == 3:
            max_row.append(np.max(np.sum(mask[0], axis=0)))
            max_col.append(np.max(np.sum(mask[0], axis=1)))
            square_list.append(int(np.sum(mask[0])))
    print(max(max_row))
    print(max(max_col))
    print(sorted(os.listdir(folder))[max_row.index(max(max_row))])
    print(sorted(os.listdir(folder))[max_col.index(max(max_col))])
    print(max(square_list))
    print(sorted(os.listdir(folder))[square_list.index(max(square_list))])

    # case_folder = os.path.join(folder, '20190502 sudaming')
    # mask = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, '2CH_MASK.nii.gz'))))
    # image = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(case_folder, '601_B-TFE_2CH_t857.nii'))))
    # Imshow3DArray(Normalize01(image[:, 2].transpose((1, 2, 0))), roi=Normalize01(np.expand_dims(mask[2], 0).repeat(30, axis=0).transpose((1, 2, 0))))
# MaxRowCol()


def Resolution():
    resolution_list = []
    folder = r'Z:\RenJi\LVA finished 2CH_MASK 20210910\LVA finished 2CH_MASK'
    for case in sorted(os.listdir(folder)):
        case_folder = os.path.join(folder, case)
        # name_list = [name for name in os.listdir(case_folder) if 'MASK' not in name]
        # image_path = os.path.join(case_folder, name_list[0])
        image_path = os.path.join(case_folder, '2CH_MASK.nii.gz')
        image = sitk.ReadImage(image_path)
        print(image.GetSpacing(), image.GetSize())
        # resolution_list.append([image.GetSpacing()[0], image.GetSpacing()[1]])
    # print(max(resolution_list, key=resolution_list.count))
# Resolution()


def AAAA():
    folder = r'Z:\RenJi\LVA finished 2CH_MASK 20210910\LVA finished 2CH_MASK\20140923 suyongming'
    image_path = os.path.join(folder, 'resize.nii.gz')
    mask_path = os.path.join(folder, 'mask.nii.gz')
    image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    Imshow3DArray(Normalize01(image.transpose((1, 2, 0))), roi=Normalize01(mask.transpose((1, 2, 0))))
# AAAA()