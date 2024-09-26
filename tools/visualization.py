import argparse
import json
import os

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sample4geo.dataset.visloc import VisLocEval, get_transforms
from sample4geo.model import TimmModel
from sample4geo.model_cls import TimmClassModel
from sample4geo.trainer import predict


def eval_query(qf, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)

    score = score.squeeze().cpu().numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    top_10_index = gl[index[:10]]
    return top_10_index


def evaluate(config,
             model,
             query_loader,
             gallery_loader, ):
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)

    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()
    top_10_list = []

    for i in tqdm(range(len(ids_query))):
        top_10_index = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)
        top_10_list.append(top_10_index)
    return top_10_list, list(ql)


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
parser.add_argument('--verbose', default=True,
                    help='进度可视化')
parser.add_argument('--normalize_features', default=True,
                    help='特征归一化')
parser.add_argument('--img_size', type=int, nargs='+', default=[384, 384], help="the size of ground images")
parser.add_argument('--mean', type=int, nargs='+', default=[0.485, 0.456, 0.406], help="the mean of normalized images")
parser.add_argument('--std', type=int, nargs='+', default=[0.229, 0.224, 0.225], help="the std of normalized images")
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

args.query_folder_test = os.path.join("/home/hello/data/UAV-VisLoc/test/query_drone")
args.gallery_folder_test = os.path.join("/home/hello/data/UAV-VisLoc/test/gallery_satellite")
val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(args.img_size, mean=args.mean,
                                                                              std=args.std)
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
                                gallery_n=-1,
                                )

query_dataloader_test = DataLoader(query_dataset_test,
                                   batch_size=128,
                                   num_workers=16,
                                   shuffle=False,
                                   pin_memory=True)

gallery_dataloader_test = DataLoader(gallery_dataset_test,
                                     batch_size=128,
                                     num_workers=16,
                                     shuffle=False,
                                     pin_memory=True)

top_10_indexs, query_index = evaluate(config=args,
                                     model=model,
                                     query_loader=query_dataloader_test,
                                     gallery_loader=gallery_dataloader_test)

visual = dict()
train_ids_list = list()

# for shuffle pool
for i in range(len(top_10_indexs)):
    visual[str(query_index[i])] = top_10_indexs[i].tolist()

# Write the `visual` dictionary to a txt file
with open('visual.txt', 'w') as file:
    json.dump(visual, file, indent=4)

print("Dictionary 'visual' has been written to 'visual.txt'")
print(visual)
