import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F


def get_preds(outputs, outputs2):
    if isinstance(outputs, list):
        preds = []
        preds2 = []
        for out, out2 in zip(outputs, outputs2):
            preds.append(torch.max(out.data, 1)[1])
            preds2.append(torch.max(out2.data, 1)[1])
    else:
        _, preds = torch.max(outputs.data, 1)
        _, preds2 = torch.max(outputs2.data, 1)
    return preds, preds2


def train_cls(args, model, dataloader, loss_function, optimizer, scheduler=None):
    # set model train mode
    model.train()

    losses = AverageMeter()

    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)

    step = 1

    if args.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    # for loop over one epoch
    for query, reference, label in bar:

        # data (batches) to device
        if args.gpu is not None:
            query = query.cuda(args.gpu, non_blocking=True)
            reference = reference.cuda(args.gpu, non_blocking=True)
            label = label.cuda(args.gpu, non_blocking=True)
        now_batch_size = query.shape[0]
        # 模型前向传播
        with autocast():
            outputs1, outputs2 = model(query, reference)

        # 计算损失
        loss, cls_loss, f_triplet_loss, kl_loss = loss_function(
            outputs1, outputs2, label, label)

        losses.update(loss.item())

        # Calculate gradient using backward pass
        loss.backward()

        # Gradient clipping
        if args.clip_grad:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)

            # Update model parameters (weights)
        optimizer.step()
        # Zero gradients for next step
        optimizer.zero_grad()

        if args.verbose:
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "cls_loss": "{:.4f}".format(cls_loss.item()),
                       "f_triplet_loss": "{:.4f}".format(f_triplet_loss.item()),
                       "kl_loss": "{:.4f}".format(kl_loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr_backbone": "{:.6f}".format(optimizer.param_groups[0]['lr']),
                       "lr_head": "{:.6f}".format(optimizer.param_groups[1]['lr'])}

            bar.set_postfix(ordered_dict=monitor)

        step += 1

    if args.verbose:
        bar.close()

    return losses.avg


def predict(train_config, model, dataloader):
    model.eval()

    # wait before starting progress bar
    time.sleep(0.1)

    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    img_features_list = []

    ids_list = []
    with torch.no_grad():

        for img, ids in bar:

            ids_list.append(ids)

            with autocast():
                img = img.to(train_config.gpu)
                cls, img_feature = model(img)

                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.gpu)

    if train_config.verbose:
        bar.close()

    return img_features, ids_list
