import argparse
import numpy as np
import cv2
import torch

import torchvision.models as models
import torchvision.transforms as transforms

import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from sample4geo.model import TimmModel
from sample4geo.model_cls import TimmClassModel


def reshape_transform(tensor):
    result = tensor.permute(0, 3, 1, 2)
    return result


parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model', default='convnext_b', type=str,
                    help='ViTS-384 | ViTB-384 | convnext_b | resnet50 | convnext_t | convnext_s')

parser.add_argument('--use_safa', default=True,
                    help='是否使用SAFA')
parser.add_argument('--use_class', default=False,
                    help='是否使用分类思想进行训练')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--share', default=True,
                    help='特征提取骨干网络是否共享权重')
parser.add_argument('--class_num', default=120, type=int,
                    help='类别数量')
parser.add_argument('--img_size', type=int, nargs='+', default=[384, 384], help="the size of ground images")
parser.add_argument('--checkpoint',
                    default="/home/hello/cjl/Sample4Geo-main/result/info/convnext_b/visloc/base_safa_sample/model_best_48_0.8861.pth.tar",
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
args = parser.parse_args()
if args.use_class:
    model = TimmClassModel(args)
else:
    model = TimmModel(args)
model.cuda(args.gpu).eval()
loc = 'cuda:{}'.format(args.gpu)
checkpoint = torch.load(args.checkpoint, map_location=loc)
# 移除 'module.' 前缀
state_dict = {k.partition('module.')[2] if k.startswith('module.') else k: v for k, v in checkpoint['model'].items()}
model.load_state_dict(state_dict)
traget_layers = [model.model.features.stages[-1][-1]]
# traget_layers = [model.model.stages[-1][-1]]
#  2.读取图片，将图片转为RGB
# origin_img = cv2.imread("/home/hello/data/UAV-VisLoc/train/drone/01_0409.JPG")[:,:,::-1]
origin_img = cv2.imread("/home/hello/data/UAV-VisLoc/train/drone/06_0289.JPG")[:, :, ::-1]
origin_img = cv2.resize(origin_img, args.img_size)
net_input = preprocess_image(origin_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
net_input = net_input.cuda()

#  5.实例化cam
cam = pytorch_grad_cam.GradCAMPlusPlus(model=model, target_layers=traget_layers)

grayscale_cam = cam(net_input)
grayscale_cam = grayscale_cam[0, :]
origin_img = np.float32(origin_img) / 255
visualization_img = show_cam_on_image(origin_img, grayscale_cam)
cv2.imwrite('convnext_safa_heatmap_06_0289.jpg', visualization_img)
cv2.waitKey(0)
