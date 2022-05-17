import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import datasets, transforms
import data_transforms as transforms
from path import Path

import numpy as np
import json
from PIL import Image
import threading

from model.models import DPTSegmentationModel
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
    [160, 80, 240], 
    [240, 80, 240],
    [255, 255, 255]], dtype=np.uint8)

def save_colorful_images(ms_images, predictions, gts, filenames, output_dir, palettes):
   for ind in range(len(filenames)):
       image = np.concatenate([ms_images[ind].squeeze(), palettes[predictions[ind].squeeze()], palettes[np.clip(gts[ind], 0, 41).squeeze()]], axis=1)
       im = Image.fromarray(image)
       fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
       label_save = os.path.join(output_dir, filenames[ind][:-4] + '.npy')
       out_dir = split(fn)[0]
       if not exists(out_dir):
           os.makedirs(out_dir)
       np.save(label_save, predictions[ind].squeeze())
       im.save(fn)

def pridictor(args, image, order):
    # define model
    single_model = DPTSegmentationModel(41, backbone="vitb_rn50_384")
    args.info_path = "/DATA2/train/info.json"
    args.ckp_path = "/mnt/yuhang/Cerberus/model_best_bs8_new.pth.tar"
    
    # pretrained model
    checkpoint = torch.load(args.ckp_path)
    for name, param in checkpoint['state_dict'].items():
        single_model.state_dict()[name].copy_(param)
    model = single_model.cuda()
    
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
    pred = test_ms(ms_images, model, args)
    print(f"Order{order} Done")
    return pred


def test_ms(ms_images, model, args):
    model.eval()
    outputs = []
    w, h = 640, 480
    with torch.no_grad():
        for image in ms_images:
            image_var = Variable(image, requires_grad=False)
            image_var = image_var.cuda()
            final, _ = model(image_var)
            final_array = list()
            for entity in final:
                final_array.append(entity.data)
            outputs.append(final_array)
        
        final = list()
        for label_idx in range(len(outputs[0])):
            tmp_tensor_list = list()
            for out in outputs:
                tmp_tensor_list.append(resize_4d_tensor(out[label_idx], w, h)) 
            final.append(sum(tmp_tensor_list))
        pred = list()
        for label_entity in final:
            pred.append(label_entity.argmax(axis=1))

        return pred


def resize_4d_tensor(tensor, width, height):
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


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()


    return args


def main():
    args = parse_args()
    img_dir = Path("/DATA2/train/check_rgb/")
    gt_dir = Path("/DATA2/train/check_label/")
    img_list = sorted(img_dir.files("*.png"))
    gt_list = sorted(gt_dir.files("*.png"))
    for i in range(len(img_list)):
        image = img_list[i]
        gt = gt_list[i]
        image = np.array(Image.open(image))
        gt = np.array(cv2.imread(gt, cv2.IMREAD_GRAYSCALE))//5
        pred = pridictor(args, image, i)
        save_colorful_images([image], pred, [gt], [str(i) + ".png"], './test_value', NYU40_PALETTE)



if __name__ == '__main__':
    main()
