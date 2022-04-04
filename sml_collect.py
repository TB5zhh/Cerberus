import argparse
import json
import os
import sys
import time

import scipy
from deepv3 import BoundarySuppressionWithSmoothing

import numpy as np
import torch
from IPython import embed
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_transforms as transforms
from args import parse_args
from dataset.emulated import EmulatedDataset
from datasets.emulated_process import convert_mask_to_rgb
from misc import INFO
from segment_open import AverageMeter, DRNSeg, fast_hist, per_class_iu
from PIL import Image
from project import project

def collect_sml(dataloader, model, num_classes):

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    stat = [[] for _ in range(num_classes)]
    for _, (image, _, _) in enumerate(tqdm(dataloader)):
        data_time.update(time.time() - end)
        with torch.no_grad():
            final = model(image)[3]
            logits, indices = torch.max(final, 1)
            for i in range(num_classes):
                stat[i].append(logits[indices == i].flatten())

        batch_time.update(time.time() - end)
        end = time.time()

    for i in range(num_classes):
        stat[i] = torch.hstack(stat[i])
        stat[i] = (stat[i].mean().cpu().item(), stat[i].std().cpu().item())
    return stat

from scipy.io import savemat
def judge_sml(dataloader, model_v, model_x, num_classes, stat_v, stat_x, threshold=0., out_dir='.'):
    idx = 0
    model_v.eval()
    model_x.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((256, 256))
    os.makedirs(out_dir, exist_ok=True)
    b = BoundarySuppressionWithSmoothing(boundary_suppression=True)
    for _, (image_v, target_v, image_x, target_x, depth_v, depth_x, pose_v, pose_x) in enumerate(tqdm(dataloader)):
        data_time.update(time.time() - end)
        with torch.no_grad():
            final_v = model_v(image_v)[3]
            # savemat(f'{out_dir}/{idx}.mat', {'logits': final.cpu().data.numpy()})
            logits_v, indices_v = torch.max(final_v, 1)
            for i in range(num_classes):
                logits_v[indices_v == i] -= stat_v[i][0]
                logits_v[indices_v == i] /= stat_v[i][1]
            logits_v = b(logits_v, indices_v)

            for i in range(num_classes):
                if (indices_v == i).sum() == 0:
                    continue
                if i == 12:
                    m1 = ((logits_v - logits_v[indices_v == i].min()) / (logits_v[indices_v == i].max() - logits_v[indices_v == i].min())) < 0.9
                elif i == 0:
                    m1 = ((logits_v - logits_v[indices_v == i].min()) / (logits_v[indices_v == i].max() - logits_v[indices_v == i].min())) < 0.2
                else:
                    m1 = ((logits_v - logits_v[indices_v == i].min()) / (logits_v[indices_v == i].max() - logits_v[indices_v == i].min())) < threshold
                m2 = indices_v == i
                m = torch.logical_and(m1, m2)
                indices_v[m] = 255
            
            final_x = model_x(image_x)[3]
            # savemat(f'{out_dir}/{idx}.mat', {'logits': final.cpu().data.numpy()})
            logits_x, indices_x = torch.max(final_x, 1)
            for i in range(num_classes):
                logits_x[indices_x == i] -= stat_x[i][0]
                logits_x[indices_x == i] /= stat_x[i][1]
            logits_x = b(logits_x, indices_x)

            for i in range(num_classes):
                if (indices_x == i).sum() == 0:
                    continue
                if i == 12:
                    m1 = ((logits_x - logits_x[indices_x == i].min()) / (logits_x[indices_x == i].max() - logits_x[indices_x == i].min())) < 0.9
                elif i == 0:
                    m1 = ((logits_x - logits_x[indices_x == i].min()) / (logits_x[indices_x == i].max() - logits_x[indices_x == i].min())) < 0.2
                else:
                    m1 = ((logits_x - logits_x[indices_x == i].min()) / (logits_x[indices_x == i].max() - logits_x[indices_x == i].min())) < threshold
                m2 = indices_x == i
                m = torch.logical_and(m1, m2)
                indices_x[m] = 255
            

            # hist += fast_hist(np.asarray(indices_v.flatten().cpu()), np.asarray(target_v.flatten().cpu()), 256)  # FIXME
            
            os.makedirs(out_dir, exist_ok=True)
            # embed()
            
            for batch in range(len(indices_v)):
                embed()
                Image.fromarray(
                    np.hstack((
                        np.vstack((
                            convert_mask_to_rgb(np.asarray(target_v[batch].cpu())),
                            convert_mask_to_rgb(np.asarray(indices_v[batch].cpu())),
                        )),
                        np.vstack((
                            project(depth_x[batch], torch.as_tensor(convert_mask_to_rgb(np.asarray(target_x[batch].cpu()))).cuda(), pose_x[batch], pose_v[batch]).cpu(),
                            project(depth_x[batch], torch.as_tensor(convert_mask_to_rgb(np.asarray(indices_x[batch].cpu()))).cuda(), pose_x[batch], pose_v[batch]).cpu(),
                        )),
                        np.vstack((
                            convert_mask_to_rgb(project(depth_x[batch], target_x[batch].reshape((*target_x[batch].shape, 1)), pose_x[batch], pose_v[batch]).cpu().squeeze()),
                            convert_mask_to_rgb(project(depth_x[batch], indices_x[batch].reshape((*indices_x[batch].shape, 1)), pose_x[batch], pose_v[batch]).cpu().squeeze()),
                        )),
                        np.vstack((
                            convert_mask_to_rgb(np.asarray(target_x[batch].cpu())),
                            convert_mask_to_rgb(np.asarray(indices_x[batch].cpu())),
                        )),
                        # project(depth_x, None, pose_x, pose_v)
                    )).astype(np.uint8)
                    ).save(f'{out_dir}/{idx}.png')
                idx += 1
        batch_time.update(time.time() - end)
        end = time.time()
    print(per_class_iu(hist) * 100)


def main(args):
    # with open(f'sml_result_{args.type}.json') as f:
    # stat = json.load(f)
    if args.action == 'use':
        # Model
        model_v = DRNSeg(args.arch, args.classes, pretrained_model=None, pretrained=False)
        model_v = DataParallel(model_v).cuda()
        model_x = DRNSeg(args.arch, args.classes, pretrained_model=None, pretrained=False)
        model_x = DataParallel(model_x).cuda()

        if args.resume_v:
            if os.path.isfile(args.resume_v):
                checkpoint = torch.load(args.resume_v)
                start_epoch = checkpoint['epoch']
                model_v.load_state_dict(checkpoint['state_dict'])
        if args.resume_x:
            if os.path.isfile(args.resume_x):
                checkpoint = torch.load(args.resume_x)
                start_epoch = checkpoint['epoch']
                model_x.load_state_dict(checkpoint['state_dict'])

        # Dataset and Dataloader
        Dataset = EmulatedDataset if args.dataset_type == 'emulated' else None
        assert Dataset is not None

        dataset = Dataset(data_dir=args.data_dir,
                        transforms=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=INFO['mean'], std=INFO['std']),
                        ]),
                        type=args.type,
                        series=range(200) if 'series' not in vars(args).keys() else args.series,
                        cut=args.cut)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        stat_v = torch.load(f'sml_result_vehicle.obj')
        stat_x = torch.load(f'sml_result_road.obj')
        judge_sml(dataloader, model_v, model_x, num_classes=args.classes, stat_v=stat_v, stat_x=stat_x, out_dir=args.vis_dir, threshold=args.threshold)
    elif args.action == 'collect':
        # Model
        model = DRNSeg(args.arch, args.classes, pretrained_model=None, pretrained=False)
        model = DataParallel(model).cuda()

        if args.resume:
            if os.path.isfile(args.resume):
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])

        # Dataset and Dataloader
        Dataset = EmulatedDataset if args.dataset_type == 'emulated' else None
        assert Dataset is not None

        dataset = Dataset(data_dir=args.data_dir,
                        transforms=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=INFO['mean'], std=INFO['std']),
                        ]),
                        type=args.type,
                        series=range(200) if 'series' not in vars(args).keys() else args.series,
                        cut=args.cut)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        result = collect_sml(dataloader, model, args.classes)
        torch.save(result, f'sml_result_{args.type}.obj')


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        args = argparse.Namespace(**json.load(f))
    # args = parse_args()
    main(args)
