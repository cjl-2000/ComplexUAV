# -*- coding: utf-8 -*-
import sys

import torch

import argparse
from thop import profile, clever_format

from sample4geo.model import TimmModel
from sample4geo.model_cls import TimmClassModel


def calc_flops_params(model,
                      input_size_drone,
                      input_size_satellite,
                      ):
    inputs_drone = torch.randn(input_size_drone).cuda()
    inputs_satellite = torch.randn(input_size_satellite).cuda()
    total_ops, total_params = profile(
        model, (inputs_drone, inputs_satellite,), verbose=False)
    print("FLOPS = " + str(total_ops/1000**3)+"G")
    print("Params = " + str(total_params / 1000 ** 2) + "M")
    macs, params = clever_format([total_ops, total_params], "%.3f")
    return macs, params


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model', default='vmamba_s', type=str,
                    help='ViTS-384 | ViTB-384 | convnext_b | resnet50 | convnext_t | convnext_s | vmamba_b | vmamba_s')

parser.add_argument('--use_safa', default=False,
                    help='是否使用SAFA')
parser.add_argument('--use_class', default=True,
                    help='是否使用分类思想进行训练')
parser.add_argument('--share', default=True,
                    help='特征提取骨干网络是否共享权重')
parser.add_argument('--class_num', default=4136, type=int,
                    help='类别数量')
parser.add_argument('--img_size', type=int, nargs='+', default=[384, 384], help="the size of ground images")
parser.add_argument('--test_h', default=384, type=int, help='height')
parser.add_argument('--test_w', default=384, type=int, help='width')
args = parser.parse_args()
if args.use_class:
    model = TimmClassModel(args)
else:
    model = TimmModel(args)
model.cuda().eval()


# thop计算MACs
macs, params = calc_flops_params(
    model, (1, 3, args.img_size[0], args.img_size[1]), (1, 3, args.img_size[0], args.img_size[1]))
print("model MACs={}, Params={}".format(macs, params))
