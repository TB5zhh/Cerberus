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
def judge_sml(dataloader, model, num_classes, stat, threshold=0., out_dir='.'):
    idx = 0
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((256, 256))

    b = BoundarySuppressionWithSmoothing(boundary_suppression=True)
    for _, (image, target, _) in enumerate(tqdm(dataloader)):
        data_time.update(time.time() - end)
        with torch.no_grad():
            final = model(image)[3]
            savemat(f'{out_dir}/{idx}.mat', {'logits': final.cpu().data.numpy()})
            logits, indices = torch.max(final, 1)
            for i in range(num_classes):
                logits[indices == i] -= stat[i][0]
                logits[indices == i] /= stat[i][1]
            logits = b(logits, indices)
            # hindices = indices.unsqueeze(1)
            # hones = torch.ones_like(hindices)
            # for idx, ker_size in enumerate(range(8, 0, -1)):
            #     sum = torch.nn.functional(hones, kernels[idx], padding=ker_size)
            #     vsum = torch.nn.functional(hindices, kernel[idx], padding=ker_size)
            # mask = None
            # embed()
            for i in range(num_classes):
                if (indices == i).sum() == 0:
                    continue
                if i == 12:
                    m1 = ((logits - logits[indices == i].min()) / (logits[indices == i].max() - logits[indices == i].min())) < 0.93
                elif i == 0:
                    m1 = ((logits - logits[indices == i].min()) / (logits[indices == i].max() - logits[indices == i].min())) < 0.2
                else:
                    m1 = ((logits - logits[indices == i].min()) / (logits[indices == i].max() - logits[indices == i].min())) < threshold
                m2 = indices == i
                m = torch.logical_and(m1, m2)
                indices[m] = 255
            # embed()
            hist += fast_hist(np.asarray(indices.flatten().cpu()), np.asarray(target.flatten().cpu()), 256)  # FIXME

            os.makedirs(out_dir, exist_ok=True)
            for batch in range(len(indices)):
                Image.fromarray(
                    np.vstack((
                        # np.transpose(np.asarray(image[batch].cpu().squeeze()), (1, 2, 0)),
                        convert_mask_to_rgb(np.asarray(target[batch].cpu())),
                        convert_mask_to_rgb(np.asarray(indices[batch].cpu())),
                    ))
                    ).save(f'{out_dir}/{idx}.png')
                idx += 1
        batch_time.update(time.time() - end)
        end = time.time()
    print(per_class_iu(hist) * 100)


def main(args):
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

    # with open(f'sml_result_{args.type}.json') as f:
    # stat = json.load(f)
    if args.action == 'use':
        stat = torch.load(f'sml_result_{args.type}.obj')
        judge_sml(dataloader, model, num_classes=args.classes, stat=stat, out_dir=args.vis_dir, threshold=args.threshold)
    elif args.action == 'collect':
        result = collect_sml(dataloader, model, args.classes)
        torch.save(result, f'sml_result_{args.type}.obj')


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        args = argparse.Namespace(**json.load(f))
    # args = parse_args()
    main(args)
