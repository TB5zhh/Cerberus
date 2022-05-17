import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import datasets, transforms
# import Cerberus.data_transforms as transforms
import Cerberus.data_transforms as transforms
import math

import numpy as np
import json
from PIL import Image
import threading

from Cerberus.model.models import DPTSegmentationModel
# from Cerberus.models_rgbdmodel.models import DPTSegmentationModel
import argparse

import cv2
import os
from os.path import exists, join, split

NYU40_PALETTE = np.asarray([
    [0, 0, 0], 
    [0, 0, 80], 
    [0, 0, 160], 
    [0, 0, 240], 
    [0, 80, 0], 
    [0, 80, 80], 
    [0, 80, 160], 
    [0, 80, 240], 
    [0, 160, 0], 
    [0, 160, 80], 
    [0, 160, 160], 
    [0, 160, 240], 
    [0, 240, 0], 
    [0, 240, 80], 
    [0, 240, 160], 
    [0, 240, 240], 
    [80, 0, 0], 
    [80, 0, 80], 
    [80, 0, 160], 
    [80, 0, 240], 
    [80, 80, 0], 
    [80, 80, 80], 
    [80, 80, 160], 
    [80, 80, 240], 
    [80, 160, 0], 
    [80, 160, 80], 
    [80, 160, 160], 
    [80, 160, 240], 
    [80, 240, 0], 
    [80, 240, 80], 
    [80, 240, 160], 
    [80, 240, 240], 
    [160, 0, 0], 
    [160, 0, 80], 
    [160, 0, 160], 
    [160, 0, 240], 
    [160, 80, 0], 
    [160, 80, 80],
    [160, 80, 160], 
    [160, 80, 240]], dtype=np.uint8)
class Cerberus_Seg():
    def __init__(self, args, rank):
        self.args = args
        self.rank = rank

    def get_predict(self, image):
        # define model
        args = self.args
        single_model = DPTSegmentationModel(41, backbone="vitb_rn50_384")
        args.info_path = "/DATA2/train/info.json"
        args.ckp_path = "/mnt/yuhang/Cerberus/model_best_bs8.pth.tar"
        
        # pretrained model
        checkpoint = torch.load(args.ckp_path)
        for name, param in checkpoint['state_dict'].items():
            single_model.state_dict()[name].copy_(param)
        #device = torch.device("cpu")
        # model = single_model.to(device)
        # print(next(model.parameters()).device)
        device_id = math.ceil((self.rank + 1 - args.num_processes_on_first_gpu) / args.num_processes_per_gpu)
        self.device = "cuda:" + str(device_id) 
        # print(self.device) 
        model = single_model.to(self.device)

        
        # preprocess data
        # In this part, input img is a numpy array. 
        # We use transform and multi-scale to preprocess input image.
        w, h = 640, 480
        info = json.load(open(args.info_path, 'r'))
        normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
        data_transform = transforms.Compose([transforms.ToTensorMultiHead(), normalize])
        scales = [0.9, 1, 1.25]
        image = Image.fromarray(image)
        ms_images = [data_transform(image.resize((round(int(w * s)/32) * 32 , round(int(h * s)/32) * 32), Image.BICUBIC))[0].unsqueeze(0) for s in scales]  

        cudnn.benchmark = True
        # optionally resume from a checkpoint
        mAP = self.test_ms(ms_images, model, args)
        return mAP
        # print(mAP)


    def test_ms(self, ms_images, model, args):
        model.eval()
        outputs = []
        w, h = 640, 480
        with torch.no_grad():
            for image in ms_images:
                image_var = Variable(image, requires_grad=False)
                image_var = image_var.to(self.device)
                final, _ = model(image_var)
                final_array = list()
                for entity in final:
                    final_array.append(entity.data)
                outputs.append(final_array)
            
            final = list()
            for label_idx in range(len(outputs[0])):
                tmp_tensor_list = list()
                for out in outputs:
                    tmp_tensor_list.append(self.resize_4d_tensor(out[label_idx], w, h)) 
                final.append(sum(tmp_tensor_list))
            pred = list()
            for label_entity in final:
                pred.append(label_entity.argmax(axis=1))

            return pred[0]
            # save_colorful_images(pred, ["sample.png"], './', NYU40_PALETTE)
            # print(pred)
            # print(pred[0].shape)

    def resize_4d_tensor(self, tensor, width, height):
        tensor_cpu = tensor.cpu().numpy()
        if tensor.size(2) == height and tensor.size(3) == width:
            return tensor_cpu
        out_size = (tensor.size(0), tensor.size(1), height, width)
        out = np.empty(out_size, dtype=np.float32)

        def resize_one(i, j):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

        def resize_channel(j):
            for i in range(tensor.size(0)):
                out[i, j] = np.array(
                    Image.fromarray(tensor_cpu[i, j]).resize(
                        (width, height), Image.BILINEAR))

        workers = [threading.Thread(target=resize_channel, args=(j,))
                for j in range(tensor.size(1))]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        return out

    # def save_colorful_images(predictions, filenames, output_dir, palettes):
    #     for ind in range(len(filenames)):
    #         im = Image.fromarray(palettes[predictions[ind].squeeze()])
    #         fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
    #         label_save = os.path.join(output_dir, filenames[ind][:-4] + '.npy')
    #         out_dir = split(fn)[0]
    #         if not exists(out_dir):
    #             os.makedirs(out_dir)
    #         np.save(label_save, predictions[ind].squeeze())
    #         im.save(fn)


# def parse_args():
#     # Training settings
#     parser = argparse.ArgumentParser(description='')
#     args = parser.parse_args()


#     return args


# def main():
#     args = parse_args()
#     image = Image.open("/DATA2/train/check_rgb/0000.png")
#     image = np.array(image)
#     predictor(args, image)


# if __name__ == '__main__':
#     main()
