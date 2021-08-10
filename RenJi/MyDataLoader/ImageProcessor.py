"""
BaseProcess
class for image or other dataset process, which could be used for data augmentation.

Author: Yang Song [songyangmri@gmail.com]
All rights reserved.
"""

from copy import deepcopy
import numpy as np
import h5py

from BasicTool.MeDIT.DataAugmentor import DataAugmentor2D, AugmentParametersGenerator
from BasicTool.MeDIT.Augment import TransformManager
from BasicTool.MeDIT.ArrayProcess import ExtractPatch, Crop2DArray, Crop3DArray

from MyDataLoader.DataAugmentor import DataAugmentor3D

class ImageProcess2D():
    def __init__(self, **kwargs):
        if 'augment_param' in kwargs.keys():
            self.augment_param = kwargs['augment_param']
            kwargs.pop('augment_param')
        else:
            self.augment_param = {}
        self.param_generator = AugmentParametersGenerator()
        self.augmentor = DataAugmentor2D()

    def SetAugmentParameter(self, augment_parameter):
        self.augment_param = augment_parameter

    def AugmentDataList2D(self, data_list, interp='linear', not_roi_list=None):
        aug_data_list = []
        if not_roi_list is None:
            not_roi_list = [True for index in range(len(data_list))]

        for data, not_roi in zip(data_list, not_roi_list):
            aug_data = np.zeros_like(data)
            for channel_index in range(data.shape[0]):
                aug_data[channel_index, ...] = self.augmentor.Execute(data[channel_index, ...],
                                                                      interpolation=interp,
                                                                      not_roi=not_roi)
            aug_data_list.append(aug_data)
        return aug_data_list

    def CropDataList2D(self, data_list, shape_list):
        crop_data_list = []
        for data, shape in zip(data_list, shape_list):
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            crop_data = np.zeros((data.shape[0], shape[1], shape[2]))
            for channel_index in range(data.shape[0]):
                crop_data[channel_index, ...], _ = ExtractPatch(data[channel_index, ...], shape[1:])
            crop_data_list.append(crop_data)
        return crop_data_list

    def GetOneRawSample(self, data_path, input_number, output_number):
        input_data_list, output_data_list = [], []
        file = h5py.File(data_path, 'r')
        for input_number_index in range(input_number):
            temp_data = np.asarray(file['input_' + str(input_number_index)])
            if temp_data.ndim == 2:
                temp_data = temp_data[np.newaxis, ...]
            input_data_list.append(temp_data)
        for output_number_index in range(output_number):
            temp_data = np.asarray(file['output_' + str(output_number_index)])
            if temp_data.ndim == 2:
                temp_data = temp_data[np.newaxis, ...]
            output_data_list.append(temp_data)
        file.close()
        return input_data_list, output_data_list

# class OneImageProcess2D(object):
#     def __init__(self, shape=None, interp='linear', is_not_roi=True):
#         self.shape = shape
#         self.interp = interp
#         self.is_not_roi = is_not_roi
#         self.augmentor = DataAugmentor2D()
#
#     def DataTransform(self, data, augment_param=None):
#         assert(data.ndim == 2)
#         if augment_param is not None:
#             self.augmentor.SetParameter(parameter_dict=augment_param)
#             transform_data = self.augmentor.Execute(data, interpolation=self.interp, not_roi=self.is_not_roi)
#         transform_data = Crop2DArray(transform_data, self.shape)
#         return transform_data


class OneImageProcess2D(object):
    def __init__(self, shape=None, is_not_roi=True, transform_sequence=None):
        self.shape = shape
        self.is_not_roi = is_not_roi
        self.transformer = TransformManager()

    def SetTransform(self, transform_sequence):
        self.transformer = TransformManager(transform_sequence=transform_sequence)

    def DataTransform(self, data, param=None):
        assert (data.ndim == 2)
        if param is not None:
            transform_data = self.transformer.Transform(data, param=param, skip=not self.is_not_roi)
        else:
            transform_data = deepcopy(data)
        transform_data = Crop2DArray(transform_data, self.shape)
        return transform_data


class OneImageProcess3D(object):
    def __init__(self, shape=None, is_not_roi=True, transform_sequence=None):
        self.shape = shape
        self.is_not_roi = is_not_roi
        self.transformer = DataAugmentor3D()

    def SetTransform(self, transform_sequence):
        self.transformer = DataAugmentor3D()

    def DataTransform(self, data, param=None):
        assert (data.ndim == 3)
        if param is not None:
            transform_data = self.transformer.Execute(data, parameter_dict=param, is_not_roi=self.is_not_roi)
        else:
            transform_data = deepcopy(data)
        transform_data = Crop3DArray(transform_data, self.shape)
        return transform_data