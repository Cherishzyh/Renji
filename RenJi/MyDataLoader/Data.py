"""
All rights reserved.
--Yang Song. [songyangmri@gmail.com]
"""

import os
from abc import abstractmethod
from random import shuffle, sample

from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


from MeDIT.Others import SplitPathWithSuffex, AddNameInEnd
from MeDIT.Augment import ParamGenerate, map_dict
from T4T.Utility.ImageProcessor import OneImageProcess2D
from T4T.Utility import FILE_SEPARATOR

from RenJi.MyDataLoader.DataAugmentor import DataAugmentor3D
from RenJi.MyDataLoader.ImageProcessor import OneImageProcess3D


def MoveTensorsToDevice(tensor, device):
    if isinstance(tensor, list):
        return [one.to(device) for one in tensor]
    else:
        return tensor.to(device)

def GetFileLevel(data_folder):
    for case in data_folder.iterdir():
        if case.name.endswith('.npy'):
            return len(case.name.split(FILE_SEPARATOR))


class DataSpliter(object):
    def __init__(self):
        pass

    def _AddAccordingFileSeparator(self, root, case_list):
        # 输入.npy文件，根据分隔符第一个字段进行拆分，然后扩充至倒数第二字段，例如：
        # zhang3_-_slice5_-_roi0.npy，根据zhang3进行拆分，然后list扩充到zhang3_-_slice5
        all = []
        for one in case_list:
            candidate = [FILE_SEPARATOR.join(temp.name[:-4].split(FILE_SEPARATOR)[:-1])
                         for temp in root.iterdir() if (temp.name.endswith('npy') and one in temp.name)]
            all.extend(candidate)
        return list(set(all))

    def _SaveIndex(self, df, store_path, store_name):
        if isinstance(df, list):
            df = pd.DataFrame({store_name: df}, index=range(len(df)))
        if isinstance(df, pd.Series):
            df = pd.DataFrame({store_name: df.values}, index=df.index)

        assert(isinstance(df, pd.DataFrame))
        df.to_csv(store_path)

    def SplitFolder(self, data_folder, train_ratio=0.8, store_name=('train', 'val')):
        root = Path(data_folder)
        case_list = [case.name[:-4].split(FILE_SEPARATOR)[0] for case in root.iterdir() if case.name.endswith('npy')]
        case_list = list(set(case_list))
        shuffle(case_list)
        sub1, sub2 = case_list[:round(len(case_list) * train_ratio)], case_list[round(len(case_list) * train_ratio):]

        file_level = GetFileLevel(root)
        if file_level > 2:
            sub1 = self._AddAccordingFileSeparator(root, sub1)
            sub2 = self._AddAccordingFileSeparator(root, sub2)

        self._SaveIndex(sub1, str(root.parent / '{}-name.csv'.format(store_name[0])), store_name[0])
        self._SaveIndex(sub2, str(root.parent / '{}-name.csv'.format(store_name[1])), store_name[1])

    def SplitFolderLabelBased(self, save_dir, label_data, train_ratio=0.8, val_ratio=0.2, store_name=('train', 'val', 'test')):
        label_dict = label_data.indexes
        label_dict = {x:label_dict[x].item() for x in label_dict.keys()}
        label_set = set(list(label_dict.values()))

        train, val, test = [], [], []
        for label in label_set:
            case_names = [x for x in label_dict.keys() if label_dict[x]==label]
            case_list = [x+'.npy' for x in case_names]
            shuffle(case_list)

            total_num = round(len(case_list))
            train_num = int(total_num*train_ratio)
            val_num = int(total_num*val_ratio)

            train.extend(case_list[:train_num])
            if train_ratio+val_ratio < 1.0:
                val.extend(case_list[train_num: train_num+val_num])
                test.extend(case_list[train_num+val_num:])
            else:
                val.extend(case_list[train_num:])

        train_df = pd.DataFrame({'train': train}).T
        val_df = pd.DataFrame({'val': val}).T

        train_df.to_csv(os.path.join(save_dir, '{}-name.csv'.format(store_name[0])))
        val_df.to_csv(os.path.join(save_dir, '{}-name.csv'.format(store_name[1])))

        if train_ratio + val_ratio < 1.0:
            test_df = pd.DataFrame({'test': test}).T
            test_df.to_csv(os.path.join(save_dir, '{}-name.csv'.format(store_name[2])))

    def LoadName(self, file_path, contain_label=False):
        df = pd.read_csv(str(file_path), index_col=0).squeeze()
        if contain_label:
            return list(df.index)
        else:
            return list(df)

    def SplitLabel(self, label_path, ratio, store_name=('train', 'val'), is_store=True):
        label = pd.read_csv(str(label_path), index_col=0).squeeze()

        sub1, sub2 = [], []
        for value in label.unique():
            sub_df = label[label == value]
            sub = list(sub_df.index)
            shuffle(sub)
            sub1.extend(sub[:int(len(sub) * ratio)])
            sub2.extend(sub[int(len(sub) * ratio):])

        if is_store:
            self._SaveIndex(label[sub1], str(AddNameInEnd(label_path, store_name[0])), store_name[0])
            self._SaveIndex(label[sub2], str(AddNameInEnd(label_path, store_name[1])), store_name[1])

        return sub1, sub2

    def SplitLabelCV(self, label_path, fold=5, store_name=('train', 'val'), store_root=None):
        label = pd.read_csv(str(label_path), index_col=0).squeeze()
        sub_sets = []
        for value in label.unique():
            one = list(label[label == value].index)
            shuffle(one)
            sub_sets.append(one)

        for fold_index in range(fold):
            print('\n#####################################################################')
            print('Start running CV-{}'.format(fold_index))
            sub1, sub2 = [], []
            for one_set in sub_sets:
                sub1.extend(one_set[:round(fold_index / fold * len(one_set))])
                sub1.extend(one_set[round((fold_index + 1) / fold * len(one_set)):])

                sub2.extend(one_set[round(fold_index / fold * len(one_set)):
                                    round((fold_index + 1) / fold * len(one_set))])

            if store_root is not None:
                self._SaveIndex(label[sub1], str(store_root / '{}-cv{}.csv'.format(store_name[0], fold_index + 1)),
                                store_name[0])
                self._SaveIndex(label[sub2], str(AddNameInEnd(label_path, store_name[1])), store_name[1])

            print('{} Num: {}, {} Num: {}'.format(store_name[0], len(sub1), store_name[1], len(sub2)))
            yield sub1, sub2


class DataManager(Dataset):
    def __init__(self, augment_param=None, sub_list=None):
        self.indexes = []
        self.inputs = []
        self.outputs = []
        self.dataset = {}
        self.num_inputs = 0
        self.num_targets = 0

        if augment_param is None:
            augment_param = {}
        self.augment_generator = ParamGenerate(augment_param)
        self.augment_sequence = [map_dict[key] for key in augment_param.keys() if key in map_dict.keys()]

        self.sub_list = sub_list

    def _MergeTwoList(self, index1, index2):
        """
        如果输入是
         l1 = ['a', 'b', 'c', 'd', 'e']
         l2 = ['a_-_1', 'a_-_2', 'a_-_3', 'd_-_1', 'd_-_2', 'f_-_1']
        输出是：
         ['a_-_1', 'a_-_2', 'a_-_3', 'd_-_1', 'd_-_2']
        """
        l1_groups = tuple(set([len(one.split(FILE_SEPARATOR)) for one in index1]))
        l2_groups = tuple(set([len(one.split(FILE_SEPARATOR)) for one in index2]))
        assert (len(l1_groups) == 1)
        assert (len(l2_groups) == 1)

        if l1_groups[0] == l2_groups[0]:
            index = [one for one in index1 if one in index2]
        else:
            if l1_groups[0] > l2_groups[0]:
                smaller, larger = index2, index1
            else:
                smaller, larger = index1, index2

            large_dict = {}
            for one in larger:
                major, minor = FILE_SEPARATOR.join(one.split(FILE_SEPARATOR)[:-1]), one.split(FILE_SEPARATOR)[-1]
                if major in large_dict.keys():
                    large_dict[major].append(one)
                else:
                    large_dict[major] = [one]

            index = []
            for one in smaller:
                if one in large_dict.keys():
                    index.extend(large_dict[one])
        if len(index) == 0:
            print('There is no index when merging.')
            raise IndexError

        return index

    def AddOne(self, one_sample, is_input=True):
        if len(self.indexes) == 0 and (self.sub_list is not None):
            self.indexes = one_sample.indexes.keys()
            self.indexes = self._MergeTwoList(self.indexes, self.sub_list)
        elif len(self.indexes) == 0 and (self.sub_list is None):
            self.indexes = one_sample.indexes.keys()
        else:
            self.indexes = self._MergeTwoList(self.indexes, one_sample.indexes.keys())
            one_sample.ExtendData(self.indexes)

        if isinstance(one_sample, Image2D):
            one_sample.image_processor.SetTransform(self.augment_sequence)
        if isinstance(one_sample, Image3D):
            one_sample.image_processor.SetTransform(self.augment_sequence)

        if is_input:
            self.inputs.append(one_sample)
            self.num_inputs += 1
        else:
            self.outputs.append(one_sample)
            self.num_targets += 1

        for one in self.inputs:
            one.ExtendData(self.indexes)
        for one in self.outputs:
            one.ExtendData(self.indexes)

        print('Total Cases: {}'.format(len(self.indexes)))

    def Balance(self, label_sample):
        def _SampleList(one_list, num):
            new_list = [one for one in one_list]
            while len(new_list) < num:
                new_list.append(sample(one_list, 1)[0])
            return new_list

        indexes = self._MergeTwoList(self.indexes, label_sample.indexes.keys())
        label_sample.ExtendData(indexes)

        df = pd.Series(label_sample.indexes)
        state = pd.value_counts(df)

        balance_index = []
        for one_label in state.index:
            index = list(df[df.values == one_label].index)
            index = _SampleList(index, state.max())
            balance_index.extend(index)

        self.indexes = balance_index

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index, **kwargs):
        augment_param = self.augment_generator.Generate()

        inputs, outputs = [], []
        for one_sample in self.inputs:
            if isinstance(one_sample, Image2D):
                inputs.append(one_sample.GetOneSample(self.indexes[index], augment_param))
            if isinstance(one_sample, Image3D):
                inputs.append(one_sample.GetOneSample(self.indexes[index]))
            elif isinstance(one_sample, Feature):
                inputs.append(one_sample.GetOneSample(self.indexes[index]))
            elif isinstance(one_sample, Label):
                outputs.append(one_sample.GetOneSample(self.indexes[index]))
        for one_sample in self.outputs:
            if isinstance(one_sample, Image2D):
                outputs.append(one_sample.GetOneSample(self.indexes[index], augment_param))
            if isinstance(one_sample, Image3D):
                inputs.append(one_sample.GetOneSample(self.indexes[index], augment_param))
            elif isinstance(one_sample, Feature):
                outputs.append(one_sample.GetOneSample(self.indexes[index]))
            elif isinstance(one_sample, Label):
                outputs.append(one_sample.GetOneSample(self.indexes[index]))

        if len(inputs) == 1:
            inputs = inputs[0]
        if len(outputs) == 1:
            outputs = outputs[0]

        return inputs, outputs


class OneSample(object):
    def __init__(self):
        self.indexes = {}

    def ExtendData(self, index_list):
        raw_group = tuple(set([len(one.split(FILE_SEPARATOR)) for one in self.indexes.keys()]))
        target_group = tuple(set([len(one.split(FILE_SEPARATOR)) for one in index_list]))
        if target_group[0] - raw_group[0] == 1:
            new_indexes = {}
            large_dict = {}
            for one in index_list:
                major, minor = FILE_SEPARATOR.join(one.split(FILE_SEPARATOR)[:-1]), one.split(FILE_SEPARATOR)[-1]
                if major in large_dict.keys():
                    large_dict[major].append(one)
                else:
                    large_dict[major] = [one]

            for key, content in self.indexes.items():
                if key in large_dict.keys():
                    for extend_key in large_dict[key]:
                        new_indexes[extend_key] = content

            self.indexes = new_indexes

        elif target_group[0] == raw_group[0]:
            new_indexes = {}
            large_dict = {}
            for one in index_list:
                if one in self.indexes.keys():
                    new_indexes[one] = self.indexes[one]
            self.indexes = new_indexes


    @abstractmethod
    def GetOneSample(self, index, **kwargs):
        pass


class Image2D(OneSample):
    def __init__(self, data_folder, shape=None, is_roi=False, sub_list=None, dtype=np.float32,
                 transform_sequence=None):
        super(Image2D, self).__init__()
        self.dtype = dtype
        self.image_processor = OneImageProcess2D(shape=shape, is_not_roi=not is_roi)

        for file_name in sorted(os.listdir(data_folder)):
            name, suffext = SplitPathWithSuffex(file_name)
            if sub_list is None:
                self.indexes[name] = os.path.join(data_folder, file_name)
            else:
                if name in sub_list:
                    self.indexes[name] = os.path.join(data_folder, file_name)
        assert(len(self.indexes) > 0)

    def GetOneSample(self, index, augment_param=None):
        assert(index in self.indexes.keys())
        data = np.load(self.indexes[index])
        transform_data = []
        for channel_index in range(data.shape[0]):
            transform_data.append(self.image_processor.DataTransform(
                data[channel_index, ...], param=augment_param)
            )
        return np.asarray(transform_data, dtype=self.dtype)


class Feature(OneSample):
    def __init__(self, file_path, sub_list=None, dtype=np.float32):
        super(Feature, self).__init__()
        self.dtype=dtype
        df = pd.read_csv(file_path, index_col=0)

        # 判断sub中是否有不合理的数据
        if sub_list is not None:
            for one in sub_list:
                if one not in df.index:
                    print('The item {} does not in the feature files'.format(one))
                    raise KeyError
            # 使用部分数据，通常见于train, val, test
            df = df.loc[sub_list]

        df = df.sort_index()
        for index in df.index:
            feature = df.loc[index].values
            assert (feature.ndim == 1)
            self.indexes[index] = feature.astype(self.dtype)

    def GetOneSample(self, index, **kwargs):
        assert(index in self.indexes.keys())
        return self.indexes[index]


class Label(OneSample):
    def __init__(self, file_path, label_tag=None, dtype=np.float):
        super(Label, self).__init__()
        self.dtype = dtype
        df = pd.read_csv(file_path, index_col=0)

        if label_tag is None:
            label_tag = df.columns[0]

        df = df[label_tag]
        df = df.sort_index()
        for index in df.index:
            self.indexes[index] = np.array(df[index], dtype=self.dtype)

    def GetOneSample(self, index, **kwargs):
        assert(index in self.indexes.keys())
        return self.indexes[index]


class Image3D(OneSample):
    def __init__(self, data_folder, aug_param, shape=None, is_roi=False, sub_list=None, dtype=np.float32,
                 transform_sequence=None):
        super(Image3D, self).__init__()
        self.dtype = dtype
        self.image_processor = OneImageProcess3D(shape=shape, is_not_roi=not is_roi)
        self.aug_param = aug_param

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

    # def GetOneSample(self, index, augment_param=None):
    def GetOneSample(self, index, **kwargs):
        assert(index in self.indexes.keys())
        data = np.load(self.indexes[index])
        transform_data = []
        for channel_index in range(data.shape[0]):
            transform_data.append(self.image_processor.DataTransform(
                data[channel_index, ...], param=self.aug_param)
            )
        return np.asarray(transform_data, dtype=self.dtype)


if __name__ == '__main__':
    spliter = DataSpliter()
    # spliter.SplitLabel(r'w:\CNNFormatData\ProsateX\cs_label.csv', ratio=0.8, store_name=('train', 'test'))
    # print(spliter.LoadName(r'w:\CNNFormatData\CsPCaDetectionTrain\val-name.csv'))
    # print(spliter.LoadName(r'w:\CNNFormatData\ProsateX\cs_label_train.csv', contain_label=True))
    # print(spliter.LoadName(r'w:\CNNFormatData\ProsateX\cs_label_test.csv', contain_label=True))
    # spliter.SplitLabel(r'w:\CNNFormatData\ProsateX\cs_label_train.csv', ratio=0.8, store_name=('train', 'val'))
    # print(spliter.LoadName(r'w:\CNNFormatData\ProsateX\cs_label_train_train.csv', contain_label=True))
    # print(spliter.LoadName(r'w:\CNNFormatData\ProsateX\cs_label_train_val.csv', contain_label=True))

    cv = spliter.SplitLabelCV(r'w:\CNNFormatData\ProstateX\cs_label.csv', fold=5)
    for train, val in cv:
        print(train)
        print(val)
        print(len(train), len(val))
