import copy
import os.path

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import gc
import time
import torch
from tqdm import tqdm
from sample4geo.utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F

def fliplr(img,args):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).to(args.gpu) # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def evaluate_lpn(config,
                 model,
                 query_loader,
                 gallery_loader,
                 ranks=[1, 5, 10],
                 step_size=1000,
                 cleanup=True):
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader,data="query")
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader,data="gallery")

    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()

    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ids_query))):
        ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    AP = ap / len(ids_query) * 100

    CMC = CMC.float()
    CMC = CMC / len(ids_query)  # average CMC

    # top 1%
    top1 = round(len(ids_gallery) * 0.01)

    string = []

    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1] * 100))

    string.append('Recall@top1: {:.4f}'.format(CMC[top1] * 100))
    string.append('AP: {:.4f}'.format(AP))

    print(' - '.join(string))
    # 定义结果字符串
    result_string = ' - '.join(string)

    # 指定要保存的文件名a
    file_name = os.path.join(config.save_path, "{}_evaluation_results.txt".format(config.dataset))

    # 打开文件并将结果写入
    with open(file_name, "a") as file:
        file.write(result_string)
        file.write("\n")  # 在已有内容后添加空行

    print("结果已保存在 {} 文件中".format(file_name))
    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        # torch.cuda.empty_cache()

    return CMC[0]




def eval_query(qf, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)

    score = score.squeeze().cpu().numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc


def predict(train_config, model_base, dataloader, data=None,block=4):
    # 创建原始模型的深拷贝
    model = copy.deepcopy(model_base)
    for i in range(block):
        cls_name = 'classifier' + str(i)
        c = getattr(model, cls_name)
        c.classifier = nn.Sequential()
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
            features = torch.FloatTensor()
            ids_list.append(ids)
            img = img.to(train_config.gpu)
            n, c, h, w = img.size()
            ff = torch.FloatTensor(n, 512, block).zero_().to(train_config.gpu)

            # why for in range(2)：
            # 1. for flip img
            # 2. for normal img

            for i in range(2):
                if i == 1:
                    img = fliplr(img,train_config)

                outputs = None
                if data == "query":
                    outputs, _ = model(img, None)
                elif data == "gallery":
                    _, outputs = model(None, img)

                ff += outputs
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(block)
            # ”2范数“ 也称为Euclid范数（欧几里得范数，常用计算向量长度），
            # 即：向量元素绝对值的平方和再开方，表示x到零点的欧式距离

            ff = ff.div(fnorm.expand_as(ff))
            # 把fnorm扩展成ff一样的形状，提高维度，
            # div除法（逐元素相除）
            # print("1", ff.shape)
            ff = ff.view(ff.size(0), -1)
            # print("2", ff.shape)

            features = torch.cat((features, ff.data.cpu()), 0)  # 在维度0上拼接
            if train_config.normalize_features:
                features = F.normalize(features, dim=-1)
            # with autocast():
            #     img = img.to(train_config.gpu)
            #     if data == "query":
            #         img_feature, _ = model(img, None)
            #     elif data == "gallery":
            #         _, img_feature = model(None, img)
            #     # normalize is calculated in fp32
            #     if train_config.normalize_features:
            #         img_feature = F.normalize(img_feature, dim=-1)

            # save features in fp32 for sim calculation
            img_features_list.append(features.to(torch.float32))

        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0)
        ids_list = torch.cat(ids_list, dim=0).to(train_config.gpu)

    if train_config.verbose:
        bar.close()

    return img_features, ids_list
