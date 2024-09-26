import os.path

import torch
import numpy as np
from tqdm import tqdm
import gc

from model.evaluate.university import eval_query
from model.trainer import predict


def evaluate(config,
             model,
             query_loader,
             gallery_loader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)

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


def evaluate_all(config,
                 model,
                 query_loader,
                 gallery_loader,
                 ranks=[1, 5, 10],
                 dataset=None,
                 cleanup=True):
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)

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
    file_name = os.path.join(config.save_path, "all_datasets_evaluation_results.txt")

    # 打开文件并将结果写入
    with open(file_name, "a") as file:
        file.write("{}: ".format(dataset))
        file.write(result_string)
        file.write("\n")  # 在已有内容后添加空行

    print("结果已保存在 {} 文件中".format(file_name))
    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        # torch.cuda.empty_cache()

    return CMC[0]


def calculate_sim(config,
                  model,
                  query_loader,
                  gallery_loader,
                  ranks=[1, 5, 10],
                  step_size=1000,
                  cleanup=True):
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)

    near_dict = calculate_nearest(query_features=img_features_query,
                                  reference_features=img_features_gallery,
                                  query_labels=ids_query,
                                  reference_labels=ids_gallery,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)
    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        # torch.cuda.empty_cache()

    return near_dict


def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64,
                      step_size=1000):
    Q = len(query_features)

    steps = Q // step_size + 1

    similarity = []

    for i in range(steps):
        start = step_size * i

        end = start + step_size

        sim_tmp = query_features[start:end] @ reference_features.T

        similarity.append(sim_tmp.cpu())

    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range + 1, dim=1)

    topk_references = []

    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i, :]])

    topk_references = torch.stack(topk_references, dim=0)

    # mask for ids without gt hits
    mask = topk_references != query_labels.unsqueeze(1)

    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()

    # dict that only stores ids where similiarity higher than the lowes gt hit score
    nearest_dict = dict()

    for i in range(len(topk_references)):
        nearest = topk_references[i][mask[i]][:neighbour_range]

        nearest_dict[query_labels[i].item()] = list(nearest)

    return nearest_dict
