'''
MIP4AIM.NiiProcess.Resampler
Utility for Resampling.

author: Yang Song
All right reserved
'''

import os
from pathlib import Path
from copy import deepcopy

import numpy as np
import SimpleITK as sitk

class Resampler():
    def __init__(self):
        pass

    def _GenerateFileName(self, file_path, name):
        store_path = ''
        if os.path.splitext(file_path)[1] == '.nii':
            store_path = file_path[:-4] + '_' + name + '.nii'
        elif os.path.splitext(file_path)[1] == '.gz':
            store_path = file_path[:-7] + '_' + name + '.nii.gz'
        else:
            print('the input file should be suffix .nii or .nii.gz')

        return store_path

    def Get2DTransform(image, rotate_angle=0, scale_width=1, scale_height=1, shear=0, shift_left=0, shift_top=0):
        input_shape = image.GetSize()
        input_shape = np.asarray(input_shape)
        affine_transform = sitk.AffineTransform(2)
        affine_transform.SetCenter(image.TransformIndexToPhysicalPoint([index // 2 for index in image.GetSize()]))
        affine_transform.Rotate(0, 1, 3.1415926 * rotate_angle / 180)
        affine_transform.Scale([2 - scale_width, 2 - scale_height])
        affine_transform.Shear(1, 0, shear)
        affine_transform.Translate([shift_left, shift_top])
        return affine_transform

    def Get3DTransform(image, rotate_angle_xy=0, rotate_angle_zx=0, rotate_angle_yz=0,
                       scale_x=1, scale_y=1, scale_z=1,
                       shear=0,
                       shift_x=0, shift_y=0, shift_z=0):
        input_shape = image.GetSize()
        input_shape = np.asarray(input_shape)
        affine_transform = sitk.AffineTransform(3)
        center_point = image.TransformIndexToPhysicalPoint([index // 2 for index in image.GetSize()])
        affine_transform.SetCenter(center_point)

        affine_transform.Rotate(0, 1, 3.1415926 * rotate_angle_xy / 180)
        affine_transform.Rotate(1, 2, 3.1415926 * rotate_angle_yz / 180)
        affine_transform.Rotate(2, 0, 3.1415926 * rotate_angle_zx / 180)
        affine_transform.Scale([1/scale_x, 1/scale_y, 1/scale_z])
        affine_transform.Shear(1, 0, shear)
        affine_transform.Translate([shift_x, shift_y, shift_z])
        return affine_transform

    def ApplyTransform(data, transform, size=None, interpolate_method=sitk.sitkBSpline):
        if isinstance(data, np.ndarray):
            image_data = sitk.GetImageFromArray(data)
        elif isinstance(data, sitk.Image):
            image_data = data

        if not (isinstance(size, list) or isinstance(size, list)):
            size = image_data.GetSize()

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image_data)
        resampler.SetTransform(transform)
        resampler.SetInterpolator(interpolate_method)
        print(resampler.GetOutputSpacing())
        # resampler.SetOutputSpacing([index * 2 for index in data.GetSpacing()])

        temp = resampler.Execute(image_data)
        print(temp.GetSpacing())
        print(temp.GetSize())
        result = temp

        # shift = [128, -128, 0]
        # shift_transform = Get3DTransform(temp, shift_x=shift[0], shift_y=shift[1], shift_z=shift[2])
        # print(shift_transform)
        # resampler.SetReferenceImage(temp)
        # resampler.SetSize(size)
        # resampler.SetTransform(shift_transform)
        # result = resampler.Execute(temp)
        # result = sitk.GetArrayFromImage(result)
        return result

    def ResizeSipmleITKImage(self, image, is_roi=False, ref_image='', expected_resolution=None, expected_shape=None, method=sitk.sitkBSpline,
                             dtype=sitk.sitkFloat32, store_path=''):
        if (expected_resolution is None) and (expected_shape is None) and (ref_image == ''):
            print('Give at least one parameters. ')
            return image

        if isinstance(image, Path):
            image = str(image)
        if isinstance(image, str) and os.path.exists(image):
            image_path = deepcopy(image)
            image = sitk.ReadImage(image)
        else:
            image_path = ''
        shape = image.GetSize()
        resolution = image.GetSpacing()

        if isinstance(ref_image, Path) and ref_image.exists():
            ref_image = sitk.ReadImage(str(ref_image))
        elif isinstance(ref_image, str) and os.path.exists(ref_image):
            ref_image = sitk.ReadImage(ref_image)

        if isinstance(ref_image, sitk.Image):
            expected_shape = ref_image.GetSize()
            expected_resolution = ref_image.GetSpacing()
        else:
            if expected_resolution is None:
                expected_shape = list(expected_shape)
                dim_0, dim_1, dim_2 = False, False, False
                if expected_shape[0] == 0:
                    expected_shape[0] = shape[0]
                    dim_0 = True
                if expected_shape[1] == 0:
                    expected_shape[1] = shape[1]
                    dim_1 = True
                if expected_shape[2] == 0:
                    expected_shape[2] = shape[2]
                    dim_2 = True
                expected_resolution = [raw_resolution * raw_size / dest_size for dest_size, raw_size, raw_resolution in
                                       zip(expected_shape, shape, resolution)]
                if dim_0: expected_resolution[0] = resolution[0]
                if dim_1: expected_resolution[1] = resolution[1]
                if dim_2: expected_resolution[2] = resolution[2]
                expected_resolution = tuple(expected_resolution)

            elif expected_shape is None:
                expected_resolution = list(expected_resolution)
                dim_0, dim_1, dim_2 = False, False, False
                if expected_resolution[0] < 1e-6:
                    expected_resolution[0] = resolution[0]
                    dim_0 = True
                if expected_resolution[1] < 1e-6:
                    expected_resolution[1] = resolution[1]
                    dim_1 = True
                if expected_resolution[2] < 1e-6:
                    expected_resolution[2] = resolution[2]
                    dim_2 = True
                expected_shape = [int(raw_resolution * raw_size / dest_resolution) for
                                  dest_resolution, raw_size, raw_resolution in zip(expected_resolution, shape, resolution)]
                if dim_0: expected_shape[0] = shape[0]
                if dim_1: expected_shape[1] = shape[1]
                if dim_2: expected_shape[2] = shape[2]
                expected_shape = tuple(expected_shape)

        resample_filter = sitk.ResampleImageFilter()

        if is_roi:
            temp_output = resample_filter.Execute(image, expected_shape, sitk.AffineTransform(len(shape)), sitk.sitkLinear,
                                             image.GetOrigin(), expected_resolution, image.GetDirection(), 0.0, dtype)
            roi_data = sitk.GetArrayFromImage(temp_output)

            new_data = np.zeros(roi_data.shape, dtype=np.uint8)
            pixels = np.unique(np.asarray(sitk.GetArrayFromImage(image), dtype=int))
            for i in range(len(pixels)):
                if i == (len(pixels) - 1):
                    max = pixels[i]
                    min = (pixels[i - 1] + pixels[i]) / 2
                elif i == 0:
                    max = (pixels[i] + pixels[i + 1]) / 2
                    min = pixels[i]
                else:
                    max = (pixels[i] + pixels[i + 1]) / 2
                    min = (pixels[i - 1] + pixels[i]) / 2
                new_data[np.bitwise_and(roi_data > min, roi_data <= max)] = pixels[i]
            output = sitk.GetImageFromArray(new_data)
            output.CopyInformation(temp_output)
        else:
            output = resample_filter.Execute(image, expected_shape, sitk.AffineTransform(len(shape)), method,
                                             image.GetOrigin(), expected_resolution, image.GetDirection(), 0.0, dtype)

        if store_path and store_path.endswith(('.nii', '.nii.gz')):
            sitk.WriteImage(output, store_path)
        elif store_path and image_path:
            sitk.WriteImage(output, self._GenerateFileName(image_path, 'Resize'))

        return output


def ResizeSipmleITKImage(image, expected_resolution=None, method=sitk.sitkBSpline, dtype=np.float32, store_path=''):
    if (expected_resolution is None):
        print('Give at least one parameters. ')
        return image

    shape = (image.GetSize()[0], image.GetSize()[1])
    resolution = (image.GetSpacing()[0], image.GetSpacing()[1])

    expected_resolution = list(expected_resolution)
    dim_0, dim_1 = False, False
    if expected_resolution[0] < 1e-6:
        expected_resolution[0] = resolution[0]
        dim_0 = True
    if expected_resolution[1] < 1e-6:
        expected_resolution[1] = resolution[1]
        dim_1 = True

    expected_shape = [int(raw_resolution * raw_size / dest_resolution) for
                      dest_resolution, raw_size, raw_resolution in
                      zip(expected_resolution, shape, resolution)]
    if dim_0: expected_shape[0] = shape[0]
    if dim_1: expected_shape[1] = shape[1]

    data = np.squeeze(sitk.GetArrayFromImage(image))
    if np.ndim(data) == 4:
        data = data[:, 2, ...]
    output_list = []
    for time in range(data.shape[0]):
        data_time = data[time, ...]
        image_time = sitk.GetImageFromArray(data_time)
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing(expected_resolution)
        resample_filter.SetInterpolator(method)
        resample_filter.SetSize(expected_shape)
        image_resize = resample_filter.Execute(image_time)
        data_resize = sitk.GetArrayFromImage(image_resize)
        output_list.append(data_resize)

    output = np.array(output_list, dtype=dtype)
    output = output[:, np.newaxis, ...]
    output = sitk.GetImageFromArray(output)
    output.SetSpacing((expected_resolution[0], expected_resolution[1], image.GetSpacing()[2]))

    if store_path and store_path.endswith(('.nii', '.nii.gz')):
        sitk.WriteImage(output, store_path)
    return output


if __name__ == '__main__':
    from MeDIT.SaveAndLoad import LoadNiiData
    ref_path = r'e:\PCaData\JSPH_PCa\DWI1500\2019-CA-formal-DING YONG MING\t2_Resize.nii'
    roi_path = r'e:\PCaData\JSPH_PCa\DWI1500\2019-CA-formal-DING YONG MING\roi.nii'

    resampler = Resampler()
    resampler.ResizeSipmleITKImage(roi_path, is_roi=True, ref_image=ref_path, store_path='save')