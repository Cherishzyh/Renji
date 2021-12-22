import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from MeDIT.Others import IterateCase
from T4T.Utility.Data import MoveTensorsToDevice
from MeDIT.Statistics import BinaryClassification

from RenJi.Network2D.MergeResNet import resnet50

class InferenceByCase():
    def __init__(self, input_shape):
        super().__init__()
        self.shape = input_shape

    def NPY2NPY(self, case, data_path):
        from MeDIT.ArrayProcess import ExtractBlock
        ch2_path = os.path.join(data_path, '2CHNPYNorm/{}.npy'.format(case))
        ch3_path = os.path.join(data_path, '3CHNPY/{}.npy'.format(case))
        ch2_label_path = os.path.join(data_path, '2CHPredROIDilated/{}.npy'.format(case))
        ch3_label_path = os.path.join(data_path, '3CHPredROIDilated/{}.npy'.format(case))
        if not os.path.exists(ch2_path) or not os.path.exists(ch3_path) or not os.path.exists(ch2_label_path) or not os.path.exists(ch3_label_path):
            return None, None, None, None
        else:
            ch2_label_arr = np.squeeze(np.load(ch2_label_path))
            ch2_label_arr, _ = ExtractBlock(ch2_label_arr, self.shape)

            ch3_label_arr = np.squeeze(np.load(ch3_label_path))
            ch3_label_arr, _ = ExtractBlock(ch3_label_arr, self.shape)

            ch2_arr = np.squeeze(np.load(ch2_path))
            ch2_arr, _ = ExtractBlock(ch2_arr, self.shape)

            ch3_arr = np.squeeze(np.load(ch3_path))
            ch3_arr, _ = ExtractBlock(ch3_arr, self.shape)

            return ch2_arr, ch3_arr, ch2_label_arr, ch3_label_arr

    def run(self, inputs, weights_list=None):
        cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
        cv_pred_list = []
        for index, data in enumerate(inputs):
            if isinstance(data, np.ndarray):
                inputs[index] = torch.unsqueeze(torch.from_numpy(np.concatenate([data[9: 12], data[-2:]], axis=0)), dim=0)
        inputs = MoveTensorsToDevice(inputs, device)
        for cv_index, cv_folder in enumerate(cv_folder_list):
            model = resnet50(input_channels=5, num_classes=1).to(device)
            if weights_list is None:
                one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if
                                         one.is_file()]
                one_fold_weights_list = sorted(one_fold_weights_list, key=lambda x: os.path.getctime(str(x)))
                weights_path = one_fold_weights_list[-1]
            else:
                weights_path = weights_list[cv_index]

            weights_path = os.path.join(model_path, weights_path)
            model.to(device)
            model.load_state_dict(torch.load(str(weights_path)))
            model.eval()

            with torch.no_grad():
                pred = model(inputs[0]*inputs[2], inputs[1]*inputs[3])
                pred = torch.squeeze(torch.sigmoid(pred)).cpu().detach().numpy()
            cv_pred_list.append(pred)
            del model, weights_path
        cv_pred = np.array(cv_pred_list)
        mean_pred = np.mean(cv_pred, axis=0)

        return mean_pred


if __name__ == '__main__':
    from MeDIT.Visualization import FlattenImages
    Seg = InferenceByCase((30, 150, 150))

    model_folder = r'/home/zhangyihong/Documents/RenJi/Model2D'
    model_path = os.path.join(model_folder, 'ResNet_1026_5slice_cv_2cl_mixseg')
    weights_list = None
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    data_root = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData'

    train_list = pd.read_csv(os.path.join(data_root, 'non_alltrain_name.csv'), index_col='CaseName').index.tolist()
    test_list = pd.read_csv(os.path.join(data_root, 'non_test_name.csv'), index_col='CaseName').index.tolist()

    data_list = test_list
    case_list, pred_list = [], []
    for index, case in enumerate(sorted(data_list)):
        print('********************** {} / {} | predicting {} **********************'.format(index+1, len(data_list), case))
        ch2_arr, ch3_arr, ch2_label_arr, ch3_label_arr = Seg.NPY2NPY(case, data_root)
        if isinstance(ch2_arr, np.ndarray):
            preds = Seg.run([ch2_arr, ch3_arr, ch2_label_arr, ch3_label_arr])
            pred_list.append(preds)
            case_list.append(case)

    new_df = pd.DataFrame({"CaseName": case_list, "Pred": pred_list})
    new_df.to_csv(os.path.join(model_path, 'pred.csv'), index=False)
