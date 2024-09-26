# -*- coding: utf-8 -*-
import sys
import time

import torch

import argparse
from thop import profile, clever_format

from sample4geo.model import TimmModel
from sample4geo.model_cls import TimmClassModel

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model', default='vmamba_b', type=str,
                    help='ViTS-384 | ViTB-384 | convnext_b | resnet50 | convnext_t | convnext_s | vmamba_b | vmamba_s')

parser.add_argument('--use_safa', default=False,
                    help='是否使用SAFA')
parser.add_argument('--use_class', default=False,
                    help='是否使用分类思想进行训练')
parser.add_argument('--share', default=True,
                    help='特征提取骨干网络是否共享权重')
parser.add_argument('--class_num', default=4136, type=int,
                    help='类别数量')
parser.add_argument('--img_size', type=int, nargs='+', default=[384, 384], help="the size of ground images")
parser.add_argument('--calc_nums', default=1000, type=int, help='width')
args = parser.parse_args()
if args.use_class:
    model = TimmClassModel(args)
else:
    model = TimmModel(args)
model.cuda().eval()
inputs_drone = torch.randn((1, 3, args.img_size[0], args.img_size[1])).cuda()
inputs_satellite = torch.randn((1, 3, args.img_size[0], args.img_size[1])).cuda()

# 预热
for _ in range(100):
    model(inputs_drone, inputs_satellite)

since = time.time()
for _ in range(args.calc_nums):
    model(inputs_drone, inputs_satellite)

print("{} inference_time = {}ms".format(args.model, ((time.time() - since) *1000 / args.calc_nums)))
