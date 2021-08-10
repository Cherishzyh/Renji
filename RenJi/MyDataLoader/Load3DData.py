from CnnTools.T4T.Utility.Data import *
from MyDataLoader.ImageProcessor import OneImageProcess3D


class Image3D(OneSample):
    def __init__(self, data_folder, shape=None, is_roi=False, sub_list=None, dtype=np.float32,
                 transform_sequence=None):
        super(Image3D, self).__init__()
        self.dtype = dtype
        self.image_processor = OneImageProcess3D(shape=shape, is_not_roi=not is_roi)

        for file_name in sorted(os.listdir(data_folder)):
            name, suffext = SplitPathWithSuffex(file_name)

            if sub_list is None:
                self.indexes[name] = os.path.join(data_folder, file_name)
            else:
                if name in sub_list:
                    self.indexes[name] = os.path.join(data_folder, file_name)
        assert(len(self.indexes) > 0)

    def SetTransform(self, transform_sequence):
        self.transformer = OneImageProcess3D(transform_sequence=transform_sequence)

    def GetOneSample(self, index, augment_param=None):
        assert(index in self.indexes.keys())
        data = np.load(self.indexes[index])
        transform_data = []
        for channel_index in range(data.shape[0]):
            transform_data.append(self.image_processor.DataTransform(
                data[channel_index, ...], param=augment_param)
            )
        return np.asarray(transform_data, dtype=self.dtype)