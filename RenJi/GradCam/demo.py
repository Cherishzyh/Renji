from __future__ import print_function

import os
import torch
import pandas as pd

import matplotlib.pyplot as plt

from MeDIT.Others import IterateCase

from RenJi.GradCam.grad_cam import GradCAM


def demo_my(model, input_list, input_class):
    model.eval()
    target_layers = ["resnet_a.layer4", "resnet_b.layer4"]
    # target_layers = ["layer4"]
    target_class = torch.argmax(input_class)

    gcam = GradCAM(model=model)
    probs = gcam.forward(input_list)

    ids_ = torch.tensor([[target_class]] * 1).long().to(device)
    gcam.backward(ids=ids_)

    gradcam_list = []
    for target_layer in target_layers:
        # Grad-GradCam
        regions = gcam.generate(target_layer=target_layer, target_shape=(150, 150))
        gradcam_list.append(regions[0, 0, ...].cpu().numpy())

    return probs, gradcam_list


def demo(model_root, model_name, data_root, device, n_classes):
    import numpy as np
    from torch.utils.data import DataLoader

    import T4T.Utility.Data
    from MeDIT.Normalize import Normalize01
    from MeDIT.Visualization import FusionImage
    from RenJi.Network2D.MergeResNet import resnet50
    # from RenJi.Network2D.ResNet2D import resnet50

    model_root = os.path.join(model_root, 'CV_1')
    one_fold_weights_list = [one for one in IterateCase(model_root, only_folder=False, verbose=0) if one.is_file()]
    one_fold_weights_list = sorted(one_fold_weights_list, key=lambda x: os.path.getctime(str(x)))
    model_weight_path = os.path.join(model_root, one_fold_weights_list[-1])
    model = resnet50(input_channels=5, num_classes=1)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)

    input_shape = (150, 150)
    case_list = pd.read_csv(os.path.join(data_root, '{}_name.csv'.format('test')), index_col='CaseName')
    case_list = case_list.index.tolist()

    data = T4T.Utility.Data.DataManager(sub_list=case_list)
    data.AddOne(T4T.Utility.Data.Image2D(data_root + '/2CHPred_5slice', shape=input_shape))
    data.AddOne(T4T.Utility.Data.Image2D(data_root + '/3CHPred_5slice', shape=input_shape))
    # data.AddOne(T4T.Utility.Data.Image2D(data_root + '/2CHROIPred_5slice', shape=input_shape, is_roi=True))
    # data.AddOne(T4T.Utility.Data.Image2D(data_root + '/3CHROIPred_5slice', shape=input_shape, is_roi=True))
    data.AddOne(T4T.Utility.Data.Label(data_root + '/label_2cl.csv'), is_input=False)
    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    if not os.path.exists(r'/home/zhangyihong/Documents/RenJi/Model2D/{}/image'.format(model_name)):
        os.mkdir(r'/home/zhangyihong/Documents/RenJi/Model2D/{}/image'.format(model_name))

    for i, (inputs, outputs) in enumerate(data_loader):
        inputs = T4T.Utility.Data.MoveTensorsToDevice(inputs, device)
        # inputs = T4T.Utility.Data.MoveTensorsToDevice([inputs[0]*inputs[2], inputs[1]*inputs[3]], device)
        # inputs = T4T.Utility.Data.MoveTensorsToDevice([inputs[0] * inputs[1]], device)

        probs, gradcam = demo_my(model, inputs, outputs)
        print("Generating Grad-GradCam: \t#{} ({:.6f})".format(int(outputs), float(probs)))

        # plt.title('Case: {}, label: {}, pred: {}'.format(i, int(outputs), float(probs)))
        # plt.imshow(merged_image_2ch)
        # plt.axis('off')
        # plt.savefig(
        #     os.path.join(r'/home/zhangyihong/Documents/RenJi/Model2D/{}/image'.format(model_name), '{}.jpg'.format(i)),
        #     aspect='normal', bbox_inches='tight', pad_inches=0.0)
        # plt.close()
        merged_image_2ch = FusionImage(Normalize01(inputs[0].cpu().numpy().squeeze()[0]),
                                       Normalize01(gradcam[0]), is_show=False)
        merged_image_3ch = FusionImage(Normalize01(inputs[1].cpu().numpy().squeeze()[0]),
                                       Normalize01(gradcam[1]), is_show=False)

        plt.title('Case: {}, label: {}, pred: {}'.format(i, int(outputs), probs))
        plt.subplot(121)
        plt.imshow(merged_image_2ch)
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(merged_image_3ch)
        plt.axis('off')
        plt.savefig(os.path.join(r'/home/zhangyihong/Documents/RenJi/Model2D/{}/image'.format(model_name), '{}.jpg'.format(i)),
                    aspect='normal', bbox_inches='tight', pad_inches=0.0)
        plt.close()



if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/RenJi/Model2D'
    data_root = r'/home/zhangyihong/Documents/RenJi/Data/CenterCropData'
    output_dir = r'/home/zhangyihong/Documents/RenJi/Model2D'
    model_name = 'ResNet_1210_SliceBySeg'

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    demo(os.path.join(model_root, model_name), model_name, data_root, device, n_classes=1)




