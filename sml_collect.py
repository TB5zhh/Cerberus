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
show = lambda x: Image.fromarray(np.asarray(x.squeeze())).show()
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
    with open('now-final-4.csv', 'w') as f:
        pass
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
                logits_x[indices_x == i] /= stat_x[i][
                    1]
            logits_x = b(logits_x, indices_x)

            for i in range(num_classes):
                if (indices_x == i).sum() == 0:
                    continue
                if i == 12:
                    m1 = ((logits_x - logits_x[indices_x == i].min()) / (logits_x[indices_x == i].max() - logits_x[indices_x == i].min())) < 0.95
                elif i == 0:
                    m1 = ((logits_x - logits_x[indices_x == i].min()) / (logits_x[indices_x == i].max() - logits_x[indices_x == i].min())) < 0
                else:
                    m1 = ((logits_x - logits_x[indices_x == i].min()) / (logits_x[indices_x == i].max() - logits_x[indices_x == i].min())) < threshold
                m2 = indices_x == i
                m = torch.logical_and(m1, m2)
                indices_x[m] = 255
            # embed()
            # indices_x = indices_x[:640]
            indices_x = indices_x[:, :640]

            indices_v = indices_v[:, :640]

            target_x = target_x[:, :640]

            target_v = target_v[:, :640]


            # hist += fast_hist(np.asarray(indices_v.flatten().cpu()), np.asarray(target_v.flatten().cpu()), 256)  # FIXME
            
            os.makedirs(out_dir, exist_ok=True)
            # embed()
            map_vertical = torch.abs(-(torch.as_tensor(range(0, 720)).reshape((-1, 1)).repeat([1, 1280]) - 720 // 2).cuda())
            depth_mask_vertical = 1.5 * 640 / map_vertical 
            map_horizontal = torch.abs((torch.as_tensor(range(0, 1280)).reshape((1, -1)).repeat([720, 1]) - 1280 // 2).cuda())
            depth_mask_horizontal = 2 * 640 / map_horizontal
            depth_mask = torch.min(depth_mask_horizontal, depth_mask_vertical)
            depth_mask = depth_mask[:640]
            depth_x = depth_x[:, :640]
            depth_v = depth_v[:, :640]
            for batch in range(len(indices_v)):
                indices_project = project(depth_x[batch], indices_x[batch].unsqueeze(2), pose_x[batch], pose_v[batch], return_mask=True)
                indices_project_color = convert_mask_to_rgb(indices_project[0].squeeze().cpu())
                indices_project_color[indices_project[1].cpu().squeeze() == False] = [0,0,0]
                target_project = project(depth_x[batch], target_x[batch].unsqueeze(2), pose_x[batch], pose_v[batch], return_mask=True)
                target_project_color = convert_mask_to_rgb(target_project[0].squeeze().cpu())
                target_project_color[target_project[1].cpu().squeeze() == False] = [0,0,0]
                same_indices = indices_v[batch] == indices_project[0].squeeze()
                same_indices[indices_project[1].squeeze() == False] = False
                different_indices = indices_v[batch] != indices_project[0].squeeze()
                different_indices[indices_project[1].squeeze() == False] = False
                same_target = target_v[batch] == target_project[0].squeeze()
                same_target[target_project[1].squeeze() == False] = False
                different_target = target_v[batch] != target_project[0].squeeze()
                different_target[target_project[1].squeeze() == False] = False

                same_anomaly_indices = torch.logical_and(same_indices, (indices_project[0] == 255).squeeze())
                same_anomaly_target = torch.logical_and(same_target, (target_project[0] == 255).squeeze())

                depth_select = torch.logical_and(depth_v[batch] <= depth_mask.cpu(), depth_v[batch] <= 200)
                # embed()
                gt = (target_v[batch] == 255)
                v_predict = (indices_v[batch] == 255).cpu()
                vx_predict = torch.as_tensor(np.where(depth_select.cpu(), different_indices.cpu(), 0).astype(bool))

                iou_v = torch.logical_and(gt, v_predict).sum() / torch.logical_or(gt, v_predict).sum()
                iou_vx = torch.logical_and(gt, vx_predict).sum() / torch.logical_or(gt, v_predict).sum()

                with open('now-final-4.csv', 'a') as f:
                    print(f"{iou_v:.4f},{iou_vx:.4f}", file=f)
                im = Image.fromarray(
                    np.hstack((
                        np.vstack((
                            convert_mask_to_rgb(np.asarray(target_v[batch].cpu())),
                            convert_mask_to_rgb(np.asarray(indices_v[batch].cpu())),
                        )),
                        np.vstack((
                            np.stack([same_anomaly_target.cpu() * 255 for i in range(3)], axis=2),
                            np.stack([same_anomaly_indices.cpu() * 255 for i in range(3)], axis=2),
                        )),
                        np.vstack((
                            np.stack([np.where(depth_select.cpu(), different_target.cpu() * 255, 0) for i in range(3)], axis=2),
                            np.stack([np.where(depth_select.cpu(), different_indices.cpu() * 255, 0) for i in range(3)], axis=2),
                        )),
                        np.vstack((
                            np.stack([different_target.cpu() * 255 for i in range(3)], axis=2),
                            np.stack([different_indices.cpu() * 255 for i in range(3)], axis=2),
                        )),

                        np.vstack((
                            project(depth_x[batch], torch.as_tensor(convert_mask_to_rgb(np.asarray(target_x[batch].cpu()))).cuda(), pose_x[batch], pose_v[batch]).cpu(),
                            project(depth_x[batch], torch.as_tensor(convert_mask_to_rgb(np.asarray(indices_x[batch].cpu()))).cuda(), pose_x[batch], pose_v[batch]).cpu(),
                        )),
                        np.vstack((
                            convert_mask_to_rgb(np.asarray(target_x[batch].cpu())),
                            convert_mask_to_rgb(np.asarray(indices_x[batch].cpu())),
                        )),
                        # project(depth_x, None, pose_x, pose_v)
                    )).astype(np.uint8)
                )
                im.save(f'{out_dir}/{idx:03d}.png')
                idx += 1
                # if idx >= 30:
                #     embed()
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
