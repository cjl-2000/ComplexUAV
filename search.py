import os

import argparse

from loguru import _logger
from torch.cuda.amp import autocast
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from model.dataset.visloc import VisLocTrain
from model.model import TimmModel

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--batch-size-eval', default=2, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', default=0.0001, type=float,
                    help='initial learning rate')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint', default=r"I:\ComplexUAV-main\result\model_best_9_0.9309.pth.tar", type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', default='./result/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://localhost:12358', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=False, action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--dataset', default='visloc', type=str,
                    help='sues,university,denseuav,visloc')
parser.add_argument('--model', default='convnext_b', type=str,
                    help='ViTS-384 | convnext_b | resnet50 | EfficientNet-B3 | convnext_t | convnext_s')
parser.add_argument('--verbose', default=True,
                    help='是否使用采样策略')
parser.add_argument('--normalize_features', default=True,
                    help='是否使用采样策略')
parser.add_argument('--prob_flip', default=0.5, type=float,
                    help='是否使用采样策略')
parser.add_argument('--use_safa', default=True,
                    help='是否使用SAFA')
parser.add_argument('--eval_gallery_n', default=-1, type=int,
                    help='int for all or int')
parser.add_argument('--clip_grad', default=None, type=float)
parser.add_argument('--label_smoothing', default=0.1, type=float)
parser.add_argument('--img_size', type=int, nargs='+', default=[384, 384], help="the size of ground images")
parser.add_argument('--mean', type=int, nargs='+', default=[0.485, 0.456, 0.406], help="the mean of normalized images")
parser.add_argument('--std', type=int, nargs='+', default=[0.229, 0.224, 0.225], help="the std of normalized images")
parser.add_argument('--eval-freq', default=1, type=int, help="the frequency of evaluation")
parser.add_argument('-e', '--evaluate', dest='evaluate', default=False, action='store_true',
                    help='evaluate model on validation set')


def predict(model, dataloader):
    bar = tqdm(dataloader, total=len(dataloader))

    img_features_list = []

    ids_list = []
    with torch.no_grad():
        for img, ids in bar:
            ids_list.append(ids)

            with autocast():
                img = img.cuda().float()
                img_feature = model(img)

            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).cuda()

    return img_features, ids_list


def evaluate(
        model,
        query_loader,
        gallery_loader, ):
    print("Extract Features:")
    img_features_query, ids_query = predict(model, query_loader)
    img_features_gallery, ids_gallery = predict(model, gallery_loader)

    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()

    for i in tqdm(range(len(ids_query))):
        good_index = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)

    return good_index


def eval_query(qf, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)

    score = score.squeeze().cpu().numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    top_10_index = index[:10]  # 获取前10个索引
    top_10_labels = gl[top_10_index]  # 根据索引获取对应的标签

    return top_10_labels


class Datasets(Dataset):

    def __init__(self, data_folder,
                 crop=False,

                 ):
        super().__init__()

        self.images = [os.path.join(data_folder, img) for img in os.listdir(data_folder)]

        self.sample_ids = [int(os.path.basename(img).split('.')[0]) for img in self.images]

    def __getitem__(self, index):
        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (384, 384))

        label = int(sample_id)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image_processed = img / 255.0
        image_processed = (image_processed - mean) / std
        image_processed = torch.from_numpy(image_processed).permute(2, 0, 1)
        return image_processed, label

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    args = parser.parse_args()
    query_image_path = r"G:\km2-online-data\test_image"
    gallery_image_path = r"G:\km2-online-data\map"
    model = TimmModel(args)
    model.cuda().eval()
    # optionally checkpoint from a checkpoint
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            # 假设checkpoint['model']包含有 'module.' 前缀的权重
            original_state_dict = checkpoint['model']

            # 创建新字典，去除 'module.' 前缀
            modified_state_dict = {k[len("module."):]: v for k, v in original_state_dict.items() if
                                   k.startswith('module.')}

            # 现在可以加载修改后的状态字典到模型中了
            model.load_state_dict(modified_state_dict)
            # model.load_state_dict(checkpoint['model'])

            # Train
        train_dataset = VisLocTrain(query_folder=args.query_folder_train,
                                    gallery_folder=args.gallery_folder_train,
                                    transforms_query=train_sat_transforms,
                                    transforms_gallery=train_drone_transforms,
                                    prob_flip=args.prob_flip,
                                    shuffle_batch_size=args.batch_size,
                                    )
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

    query_dataloader = DataLoader(query_datasets,
                                  batch_size=1,
                                  num_workers=0,
                                  shuffle=False,
                                  pin_memory=True)
    gallery_dataloader = DataLoader(gallery_datasets,
                                    batch_size=64,
                                    num_workers=0,
                                    shuffle=False,
                                    pin_memory=True)

    evaluate(model, query_dataloader, gallery_dataloader)

    print(1)
