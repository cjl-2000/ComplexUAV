import builtins
import os
import shutil
import time
import copy
import warnings

import yaml
import logging

from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from transformers import get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_constant_schedule_with_warmup

from model.dataset.denseuav import DenseUAVTrain, DenseUAVEval
from model.dataset.uav_visloc import VisLocTrain, VisLocEval
from model.evaluate.eval_cls import evaluate
from model.loss.cls import ClassLoss
from model.loss.infonce import InfoNCELoss
from model.model import TimmModel
from model.model_cls import TimmClassModel
from model.trainer import train
from model.trainer_cls import train_cls
from model.utils import adjust_learning_rate, save_checkpoint, set_up_system, adjust_learning_rate_cls

_logger = logging.getLogger('train')
import argparse
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from timm.utils import setup_default_logging
from datetime import datetime
from model.dataset.sues import SUESTrain, SUESVal
from model.dataset.university import U1652DatasetEval, U1652DatasetTrain, get_transforms

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=24, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--batch-size-eval', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', default=0.0003, type=float,
                    help='initial learning rate,adawm-cnn:0.001 | adawm-vit:0.00003 ')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint',
                    default=None,
                    type=str, metavar='PATH',
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
parser.add_argument('--multiprocessing-distributed', default=None,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--dataset', default='visloc', type=str,
                    help='sues,university,denseuav,visloc')
parser.add_argument('--model', default='convnext_b', type=str,
                    help='ViTS-384 | convnext_t | convnext_s | convnext_b | ViTB-384 | resnet50')
parser.add_argument('--height', default='150', type=int,
                    help='150, 200, 250,300 only sues')
parser.add_argument('--custom_sampling', default=False,
                    help='是否使用采样策略')
parser.add_argument('--verbose', default=True,
                    help='是否使用采样策略')
parser.add_argument('--normalize_features', default=True,
                    help='是否使用采样策略')
parser.add_argument('--prob_flip', default=0.5, type=float,
                    help='是否使用采样策略')
parser.add_argument('--use_safa', default=False,
                    help='是否使用SAFA')
parser.add_argument('--use_class', default=True,
                    help='是否使用分类思想进行训练')
parser.add_argument('--share', default=True,
                    help='特征提取骨干网络是否共享权重')
parser.add_argument('--class_num', default=4136, type=int,
                    help='类别数量,2256-denseuav | 120-sues | 701-university | 4136-visloc')
parser.add_argument('--cls_loss', default="FocalLoss",
                    help='分类损失')
parser.add_argument('--feature_loss', default=None,
                    help='对比损失')
parser.add_argument('--kl_loss', default=None,
                    help='KL损失')
parser.add_argument('--eval_gallery_n', default=-1, type=int,
                    help='int for all or int')

parser.add_argument('--clip_grad', default=100, type=float)
parser.add_argument('--label_smoothing', default=0.1, type=float)
parser.add_argument('--img_size', type=int, nargs='+', default=[384, 384], help="the size of ground images")
parser.add_argument('--mean', type=int, nargs='+', default=[0.485, 0.456, 0.406], help="the mean of normalized images")
parser.add_argument('--std', type=int, nargs='+', default=[0.229, 0.224, 0.225], help="the std of normalized images")
parser.add_argument('--eval-freq', default=1, type=int, help="the frequency of evaluation")
parser.add_argument('-e', '--evaluate', dest='evaluate', default=False, action='store_true',
                    help='evaluate model on validation set')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # 主进程地址为本地主机
    os.environ['MASTER_PORT'] = '12358'  # 用于通信的端口，所有进程保持一致，确保可以正确进行通信和数据交换
    # 初始化进程组,使用了 NCCL 后端进行通信
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main():
    args = parser.parse_args()
    if args.dataset == 'university':
        args.query_folder_train = '/home/hello/data/University-Release/train/drone'
        args.gallery_folder_train = '/home/hello/data/University-Release/train/satellite'
        args.query_folder_test = '/home/hello/data/University-Release/test/query_drone'
        args.gallery_folder_test = '/home/hello/data/University-Release/test/gallery_satellite'
    elif args.dataset == 'sues':
        args.query_folder_train = os.path.join("/home/hello/data/SUES/Datasets/Training", str(args.height), "drone")
        args.gallery_folder_train = os.path.join("/home/hello/data/SUES/Datasets/Training", str(args.height),
                                                 "satellite")
        args.query_folder_test = os.path.join("/home/hello/data/SUES/Datasets/Testing", str(args.height), "query_drone")
        args.gallery_folder_test = os.path.join("/home/hello/data/SUES/Datasets/Testing", str(args.height),
                                                "gallery_satellite")
    elif args.dataset == 'denseuav':
        args.query_folder_train = os.path.join("/home/hello/data/DenseUAV/train/drone")
        args.gallery_folder_train = os.path.join("/home/hello/data/DenseUAV/train/satellite")
        args.query_folder_test = os.path.join("/home/hello/data/DenseUAV/test/query_drone")
        args.gallery_folder_test = os.path.join("/home/hello/data/DenseUAV/test/gallery_satellite")
    elif args.dataset == 'visloc':
        args.query_folder_train = os.path.join("/home/hello/data/UAV-VisLoc/train/drone")
        args.gallery_folder_train = os.path.join("/home/hello/data/UAV-VisLoc/train/satellite")
        args.query_folder_test = os.path.join("/home/hello/data/UAV-VisLoc/test/query_drone")
        args.gallery_folder_test = os.path.join("/home/hello/data/UAV-VisLoc/test/gallery_satellite")
    print(args)
    args_dict = vars(args)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    else:
        timestamp = time.time()
        local_time = time.localtime(timestamp)
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S", local_time)
        if args.dataset == "sues":
            args.save_path = os.path.join(args.save_path, "cls", args.model, "{}_{}".format(args.dataset, args.height),
                                          time_str)
        else:
            args.save_path = os.path.join(args.save_path, "cls", args.model, args.dataset, time_str)
        os.makedirs(args.save_path)

    with open(os.path.join(args.save_path, "args.yaml"), 'w') as file:
        yaml.dump(args_dict, file)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ['WORLD_SIZE'])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    is_best = False
    best_acc = 0.

    setup_default_logging(log_path=f'{args.save_path}/train.log')

    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        os.environ['MASTER_ADDR'] = 'localhost'  # 主进程地址为本地主机
        os.environ['MASTER_PORT'] = '12358'  # 用于通信的端口，所有进程保持一致，确保可以正确进行通信和数据交换
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    set_up_system(args)

    # create model
    end = time.time()
    if args.gpu == 0:
        _logger.info("=> creating model")

    if not args.multiprocessing_distributed or (dist.is_initialized() and args.gpu == 0):
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

    model = TimmClassModel(args)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    elif args.gpu is not None:
        model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    # loss_function = InfoNCELoss(args).cuda(args.gpu)
    loss_function = ClassLoss(args).cuda(args.gpu)
    parameters.extend(loss_function.parameters())
    ignored_params = list(map(id, model.module.head.parameters() if args.multiprocessing_distributed else model.head.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': model.module.head.parameters() if args.multiprocessing_distributed else model.head.parameters(), 'lr': 0.5 * args.lr}])

    # optimizer = optim.SGD([
    #     {'params': base_params, 'lr': 0.1 * args.lr},
    #     {'params': model.module.head.parameters(), 'lr': args.lr}
    # ], weight_decay=1e-4, momentum=0.9, nesterov=True)
    # optionally checkpoint from a checkpoint
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            if args.gpu == 0:
                _logger.info("=> loading checkpoint '{}'".format(args.checkpoint))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']

            model.load_state_dict(checkpoint['model'])

            optimizer.load_state_dict(checkpoint['optimizer'])

            if args.gpu == 0:
                _logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
        else:
            if args.gpu == 0:
                _logger.info("=> no checkpoint found at '{}'".format(args.checkpoint))

    if args.gpu == 0:
        _logger.info(f"=> creating model cost '{time.time() - end}'")
    end = time.time()

    if args.gpu == 0:
        _logger.info("=> creating dataset")
        # Transforms
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(args.img_size, mean=args.mean,
                                                                                  std=args.std)
    if args.dataset.lower() == "university":
        # Train
        train_dataset = U1652DatasetTrain(query_folder=args.query_folder_train,
                                          gallery_folder=args.gallery_folder_train,
                                          transforms_query=train_sat_transforms,
                                          transforms_gallery=train_drone_transforms,
                                          prob_flip=args.prob_flip,
                                          shuffle_batch_size=args.batch_size,
                                          )

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
    elif args.dataset.lower() == "sues":
        # Train
        train_dataset = SUESTrain(query_folder=args.query_folder_train,
                                  gallery_folder=args.gallery_folder_train,
                                  transforms_query=train_sat_transforms,
                                  transforms_gallery=train_drone_transforms,
                                  prob_flip=args.prob_flip,
                                  shuffle_batch_size=args.batch_size,
                                  )
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

    elif args.dataset.lower() == "denseuav":
        # Train
        train_dataset = DenseUAVTrain(query_folder=args.query_folder_train,
                                      gallery_folder=args.gallery_folder_train,
                                      transforms_query=train_sat_transforms,
                                      transforms_gallery=train_drone_transforms,
                                      prob_flip=args.prob_flip,
                                      shuffle_batch_size=args.batch_size,
                                      )
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

    elif args.dataset.lower() == "visloc":
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


    else:
        print('not implemented!')
        raise Exception

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

    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler=None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    if args.evaluate:
        if not args.multiprocessing_distributed or args.gpu == 0:
            evaluate(config=args,
                     model=model,
                     query_loader=query_dataloader_test,
                     gallery_loader=gallery_dataloader_test,
                     ranks=[1, 5, 10],
                     step_size=1000,
                     cleanup=True)
        return
    # -----------------------------------------------------------------------------#
    # Shuffle                                                                     #
    # -----------------------------------------------------------------------------#
    if args.custom_sampling and args.dataset.lower() != "visloc":
        train_loader.dataset.shuffle()

    if args.gpu == 0:
        _logger.info(f"=> creating dataset cost {time.time() - end}")
    # -----------------------------------------------------------------------------#
    # Train                                                                       #
    # -----------------------------------------------------------------------------#
    for epoch in range(args.start_epoch, args.epochs):
        if args.gpu == 0:
            _logger.info('start epoch:{}, date:{}'.format(epoch, datetime.now()))
        if args.distributed:
            train_sampler.set_epoch(epoch)

        lr = adjust_learning_rate_cls(optimizer, epoch, lr=args.lr, total_epoch=args.epochs)

        train_loss = train_cls(args,
                               model,
                               dataloader=train_loader,
                               loss_function=loss_function,
                               optimizer=optimizer)
        print("Epoch: {}, Train Loss = {:.3f}, Lr_backbone = {:.6f},Lr_head = {:.6f}".format(epoch,
                                                                   train_loss,
                                                                   optimizer.param_groups[0]['lr'],
                                                                optimizer.param_groups[1]['lr']))
        _logger.info(f"The learning rate of backbone {epoch} is {optimizer.param_groups[0]['lr']}")
        _logger.info(f"The learning rate of head {epoch} is {optimizer.param_groups[1]['lr']}")
        _logger.info(f"The Train Loss is {train_loss}")
        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
                r1_test = evaluate(config=args,
                                   model=model,
                                   query_loader=query_dataloader_test,
                                   gallery_loader=gallery_dataloader_test,
                                   ranks=[1, 5, 10],
                                   step_size=1000,
                                   cleanup=True)
                _logger.info(f"=========================Recall==========================\n {r1_test}")
                # remember best acc@1 and save checkpoint
                is_best = r1_test > best_acc
                best_acc = max(r1_test, best_acc)

            if not args.multiprocessing_distributed or (
                    args.multiprocessing_distributed and args.gpu % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best, filename=f'checkpoint.pth.tar', save_path=args.save_path)


class ProgressMeter(object):
    def __init__(self, args, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.args = args

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.args.gpu == 0:
            _logger.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
