from __future__ import print_function

import os
import torch
import pandas as pd

import matplotlib.pyplot as plt

from BasicTool.MeDIT.Others import IterateCase

from RenJi.GradCam.grad_cam import GradCAM


def demo_my(model, input_list, input_class):
    model.eval()
    target_layers = ["layer4"]
    target_class = torch.argmax(input_class)

    gcam = GradCAM(model=model)
    probs = gcam.forward(input_list)

    ids_ = torch.tensor([[target_class]] * 1).long().to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-GradCam @{}".format(target_layer))

        # Grad-GradCam
        regions = gcam.generate(target_layer=target_layer, target_shape=(200, 200))

        print("\t#{} ({:.2f})".format(target_class, torch.argmax(probs)))

        gradcam = regions[0, 0, ...].cpu().numpy()

    return torch.argmax(probs), gradcam


def demo(model_root, data_root, device, n_classes):
    import numpy as np
    from torch.utils.data import DataLoader

    import CnnTools.T4T.Utility.Data
    from BasicTool.MeDIT.Normalize import Normalize01
    from BasicTool.MeDIT.Visualization import FusionImage
    from RenJi.Network3D.ResNet3D import GenerateModel

    one_fold_weights_list = [one for one in IterateCase(model_root, only_folder=False, verbose=0) if one.is_file()]
    one_fold_weights_list = sorted(one_fold_weights_list, key=lambda x: os.path.getctime(str(x)))
    model_weight_path = os.path.join(model_root, one_fold_weights_list[-1])
    model = GenerateModel(50, n_input_channels=1, n_classes=n_classes)
    model.load_state_dict(torch.load(model_weight_path))
    model.to(device)

    input_shape = (200, 200)
    case_list = pd.read_csv(os.path.join(data_root, 'test_name.csv'), index_col='CaseName')
    case_list = case_list.index.tolist()

    data = CnnTools.T4T.Utility.Data.DataManager(sub_list=case_list)
    data.AddOne(CnnTools.T4T.Utility.Data.Image2D(data_root + '/Npy2D', shape=input_shape))
    data.AddOne(CnnTools.T4T.Utility.Data.Label(data_root + '/label.csv'), is_input=False)
    data_loader = DataLoader(data, batch_size=1, shuffle=False)
    for i, (inputs, outputs) in enumerate(data_loader):
        if i > 66:
            inputs = inputs[:, np.newaxis, ...]
            label = torch.zeros(n_classes)
            label[int(outputs)] = 1
            inputs = CnnTools.T4T.Utility.Data.MoveTensorsToDevice(inputs, device)
            # outputs = CnnTools.T4T.Utility.Data.MoveTensorsToDevice(outputs, device).long()

            prob, gradcam = demo_my(model, inputs, label)

            merged_image = FusionImage(Normalize01(inputs.cpu().numpy().squeeze()[0]),
                                       Normalize01(gradcam.squeeze()), is_show=False)

            plt.title('Case: {}, label: {}, pred: {}'.format(i, int(outputs), prob))
            plt.imshow(merged_image)
            plt.gca().set_axis_off()
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.show()
            plt.savefig(os.path.join(r'/home/zhangyihong/Documents/RenJi/Model/ResNet3D_0812/image', '{}.jpg'.format(i)),
                        aspect='normal', bbox_inches='tight', pad_inches=0.0)

            plt.close()



if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/RenJi/Model'
    data_root = r'/home/zhangyihong/Documents/RenJi'
    output_dir = r'/home/zhangyihong/Documents/RenJi/Model'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # for case in range(354):
    demo(os.path.join(model_root, 'ResNet3D_0812'), data_root, device, n_classes=4)



