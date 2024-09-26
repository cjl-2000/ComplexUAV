# -*- coding: utf-8 -*-
import os
import sys

import argparse

import torch
from torch.utils.data import DataLoader

from model.dataset.denseuav import DenseUAVEval
from model.dataset.sues import SUESVal
from model.dataset.uav_visloc import VisLocEval
from model.dataset.university import U1652DatasetEval, get_transforms
from model.evaluate.eval import evaluate, evaluate_all

from model.model import TimmModel

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--batch-size-eval', default=56, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--model', default='convnext_b', type=str,
                    help='ViTS-384 | ViTB-384 | convnext_b | resnet50 | convnext_t | convnext_s')
parser.add_argument('--gpu', default=3, type=int,
                    help='GPU id to use.')
parser.add_argument('--datasets', default=['sues', "university", "denseuav", "visloc"],
                    help='sues| university | denseuav | visloc')
parser.add_argument('--height', default='150', type=int,
                    help='150, 200, 250,300 only sues')
parser.add_argument('--use_safa', default=False,
                    help='是否使用SAFA')
parser.add_argument('--normalize_features', default=True,
                    help='是否使用采样策略')
parser.add_argument('--share', default=True,
                    help='特征提取骨干网络是否共享权重')
parser.add_argument('--checkpoint',
                    default="/home/hello/cjl/ComplexUAV-main/result/info/convnext_b/visloc/base_sample/model_best_33_0.8016.pth.tar",
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--eval_gallery_n', default=-1, type=int,
                    help='int for all or int')

parser.add_argument('--img_size', type=int, nargs='+', default=[384, 384], help="the size of ground images")
parser.add_argument('--mean', type=int, nargs='+', default=[0.485, 0.456, 0.406], help="the mean of normalized images")
parser.add_argument('--std', type=int, nargs='+', default=[0.229, 0.224, 0.225], help="the std of normalized images")
parser.add_argument('--verbose', default=True,
                    help='是否使用采样策略')
args = parser.parse_args()

model = TimmModel(args)
model.cuda(args.gpu)
loc = 'cuda:{}'.format(args.gpu)
checkpoint = torch.load(args.checkpoint, map_location=loc)
# 移除 'module.' 前缀
state_dict = {k.partition('module.')[2] if k.startswith('module.') else k: v for k, v in checkpoint['model'].items()}
model.load_state_dict(state_dict)
print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))

val_transforms, _, _ = get_transforms(args.img_size, mean=args.mean, std=args.std)
for dataset in args.datasets:
    if dataset == 'university':

        args.query_folder_test = '/home/hello/data/University-Release/test/query_drone'
        args.gallery_folder_test = '/home/hello/data/University-Release/test/gallery_satellite'
        # Reference Satellite Images
        gallery_dataset_test = U1652DatasetEval(data_folder=args.gallery_folder_test,
                                                mode="gallery",
                                                transforms=val_transforms,
                                                )
        # Query Ground Images Test
        query_dataset_test = U1652DatasetEval(data_folder=args.query_folder_test,
                                              mode="query",
                                              transforms=val_transforms,
                                              sample_ids=gallery_dataset_test.get_sample_ids(),
                                              gallery_n=args.eval_gallery_n,
                                              )
        print("")
    elif dataset == 'sues':

        args.query_folder_test = os.path.join("/home/hello/data/SUES/Datasets/Testing", str(args.height), "query_drone")
        args.gallery_folder_test = os.path.join("/home/hello/data/SUES/Datasets/Testing", str(args.height),
                                                "gallery_satellite")
        # Reference Satellite Images
        gallery_dataset_test = SUESVal(data_folder=args.gallery_folder_test,
                                       mode="gallery",
                                       transforms=val_transforms,
                                       )
        # Query Ground Images Test
        query_dataset_test = SUESVal(data_folder=args.query_folder_test,
                                     mode="query",
                                     transforms=val_transforms,
                                     sample_ids=gallery_dataset_test.get_sample_ids(),
                                     gallery_n=args.eval_gallery_n,
                                     )

    elif dataset == 'denseuav':

        args.query_folder_test = os.path.join("/home/hello/data/DenseUAV/test/query_drone")
        args.gallery_folder_test = os.path.join("/home/hello/data/DenseUAV/test/gallery_satellite")
        # Reference Satellite Images
        gallery_dataset_test = DenseUAVEval(data_folder=args.gallery_folder_test,
                                            mode="gallery",
                                            transforms=val_transforms,
                                            )
        # Query Ground Images Test
        query_dataset_test = DenseUAVEval(data_folder=args.query_folder_test,
                                          mode="query",
                                          transforms=val_transforms,
                                          sample_ids=gallery_dataset_test.get_sample_ids(),
                                          gallery_n=args.eval_gallery_n,
                                          )
    elif dataset == 'visloc':

        args.query_folder_test = os.path.join("/home/hello/data/UAV-VisLoc/test/query_drone")
        args.gallery_folder_test = os.path.join("/home/hello/data/UAV-VisLoc/test/gallery_satellite")
        # Reference Satellite Images
        gallery_dataset_test = VisLocEval(data_folder=args.gallery_folder_test,
                                          mode="gallery",
                                          transforms=val_transforms,
                                          )
        # Query Ground Images Test
        query_dataset_test = VisLocEval(data_folder=args.query_folder_test,
                                        mode="query",
                                        transforms=val_transforms,
                                        sample_ids=gallery_dataset_test.get_sample_ids(),
                                        gallery_n=args.eval_gallery_n,
                                        )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=args.batch_size_eval,
                                       num_workers=args.workers,
                                       shuffle=False,
                                       pin_memory=True)

    gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                         batch_size=args.batch_size_eval,
                                         num_workers=args.workers,
                                         shuffle=False,
                                         pin_memory=True)

    args.save_path = os.path.join(os.path.dirname(args.checkpoint))

    evaluate_all(config=args,
                 model=model,
                 query_loader=query_dataloader_test,
                 gallery_loader=gallery_dataloader_test,
                 ranks=[1, 5, 10],
                 dataset=dataset,
                 cleanup=True)
