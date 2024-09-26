import os
import random
import time
from collections import defaultdict

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import copy

from tqdm import tqdm


class VisLocTrain(Dataset):

    def __init__(self,
                 query_folder,
                 gallery_folder,
                 transforms_query=None,
                 transforms_gallery=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        self.pairs = []

        # 获取查询和画廊图像路径列表
        self.query_img = [os.path.join(query_folder, img) for img in os.listdir(query_folder)]
        self.gallery_img = [os.path.join(gallery_folder, img) for img in os.listdir(gallery_folder)]
        self.ids = [os.path.basename(img).split('.')[0] for img in self.gallery_img]
        self.ids.sort()
        self.map_dict = {i: self.ids[i] for i in range(len(self.ids))}
        self.reverse_map = {v: k for k, v in self.map_dict.items()}
        # 建立画廊图像前缀到图像路径的映射
        gallery_dict = defaultdict(list)
        for gallery_img in self.gallery_img:
            prefix = os.path.basename(gallery_img).split('.')[0]  # 获取前缀
            gallery_dict[prefix].append(gallery_img)  # 可支持多个相同前缀的图像

        # 根据前缀找到对应的图像对并添加到 self.pairs 中
        for query_img in self.query_img:
            prefix = os.path.basename(query_img).split('.')[0]  # 获取查询图像的前缀
            if prefix in gallery_dict:  # 使用字典进行快速查找
                self.pairs.append((prefix, query_img, gallery_dict[prefix][0]))  # 添加图像对到 pairs 列表中
        self.transforms_query = transforms_query
        self.transforms_gallery = transforms_gallery
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        self.train_ids =list()
        # for shuffle pool
        for pair in self.pairs:
            idx = int(pair[0])
            self.train_ids.append(idx)

        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):

        idx, query_img_path, gallery_img_path = self.samples[index]
        label = self.reverse_map.get(idx)
        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        gallery_img = cv2.imread(gallery_img_path)
        gallery_img = cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB)

        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            gallery_img = cv2.flip(gallery_img, 1)

            # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']

        if self.transforms_gallery is not None:
            gallery_img = self.transforms_gallery(image=gallery_img)['image']

        return query_img, gallery_img, label

    def __len__(self):
        return len(self.samples)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):

        '''
            custom shuffle function for unique class_id sampling in batch
            '''

        print("\nShuffle Dataset:")

        idx_pool = copy.deepcopy(self.train_ids)

        neighbour_split = neighbour_select // 2

        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)

        # Shuffle pairs order
        random.shuffle(idx_pool)

        # Lookup if already used in epoch
        idx_epoch = set()
        idx_batch = set()

        # buckets
        batches = []
        current_batch = []

        # counter
        break_counter = 0

        # progressbar
        pbar = tqdm()

        while True:

            pbar.update()

            if len(idx_pool) > 0:
                idx = idx_pool.pop(0)

                if idx not in idx_batch and idx not in idx_epoch and len(current_batch) < self.shuffle_batch_size:

                    idx_batch.add(idx)
                    current_batch.append(idx)
                    idx_epoch.add(idx)
                    break_counter = 0

                    if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:

                        near_similarity = similarity_pool[idx][:neighbour_range]

                        near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])

                        far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])

                        random.shuffle(far_neighbours)

                        far_neighbours = far_neighbours[:neighbour_split]

                        near_similarity_select = near_neighbours + far_neighbours

                        for idx_near in near_similarity_select:

                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:
                                break

                            # check if idx not already in batch or epoch
                            if idx_near not in idx_batch and idx_near not in idx_epoch and idx_near:
                                idx_batch.add(idx_near)
                                current_batch.append(idx_near)
                                idx_epoch.add(idx_near)
                                similarity_pool[idx].remove(idx_near)
                                break_counter = 0

                else:
                    # if idx fits not in batch and is not already used in epoch -> back to pool
                    if idx not in idx_batch and idx not in idx_epoch:
                        idx_pool.append(idx)

                    break_counter += 1

                if break_counter >= 1024:
                    break

            else:
                break

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.train_ids), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.train_ids) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0], self.samples[-1]))

class VisLocEval(Dataset):

    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 gallery_n=-1):
        super().__init__()

        self.pairs = []
        self.images = [os.path.join(data_folder, img) for img in os.listdir(data_folder)]
        self.sample_ids = [int(os.path.basename(img).split('.')[0]) for img in self.images]

        self.transforms = transforms

        self.given_sample_ids = sample_ids
        self.mode = mode

        self.gallery_n = gallery_n

    def __getitem__(self, index):

        img_path = self.images[index]
        sample_id = self.sample_ids[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if self.mode == "sat":

        #    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #    img180 = cv2.rotate(img90, cv2.ROTATE_90_CLOCKWISE)
        #    img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE)

        #    img_0_90 = np.concatenate([img, img90], axis=1)
        #    img_180_270 = np.concatenate([img180, img270], axis=1)

        #    img = np.concatenate([img_0_90, img_180_270], axis=0)

        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']

        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1

        return img, label

    def __len__(self):
        return len(self.images)

    def get_sample_ids(self):
        return set(self.sample_ids)


def get_transforms(img_size,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    val_transforms = A.Compose([A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                A.Normalize(mean, std),
                                ToTensorV2(),
                                ])

    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15,
                                                    always_apply=False, p=0.5),
                                      A.OneOf([
                                          A.AdvancedBlur(p=1.0),
                                          A.Sharpen(p=1.0),
                                      ], p=0.3),
                                      A.OneOf([
                                          A.GridDropout(ratio=0.4, p=1.0),
                                          A.CoarseDropout(max_holes=25,
                                                          max_height=int(0.2 * img_size[0]),
                                                          max_width=int(0.2 * img_size[0]),
                                                          min_holes=10,
                                                          min_height=int(0.1 * img_size[0]),
                                                          min_width=int(0.1 * img_size[0]),
                                                          p=1.0),
                                      ], p=0.3),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])

    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15,
                                                      always_apply=False, p=0.5),
                                        A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                        ], p=0.3),
                                        A.OneOf([
                                            A.GridDropout(ratio=0.4, p=1.0),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2 * img_size[0]),
                                                            max_width=int(0.2 * img_size[0]),
                                                            min_holes=10,
                                                            min_height=int(0.1 * img_size[0]),
                                                            min_width=int(0.1 * img_size[0]),
                                                            p=1.0),
                                        ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])

    return val_transforms, train_sat_transforms, train_drone_transforms


if __name__ == "__main__":
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms((384, 384), mean=mean, std=std)
    query_folder_train = r"I:\datasets\UAV-VisLoc\train\drone"
    gallery_folder_train = r'I:\datasets\UAV-VisLoc\train\satellite'
    query_folder_test = r'I:\datasets\UAV-VisLoc\test\query_drone'
    gallery_folder_test = r'I:\datasets\UAV-VisLoc\test\gallery_satellite'
    # Train
    train_dataset = VisLocTrain(query_folder=query_folder_train,
                                gallery_folder=gallery_folder_train,
                                transforms_query=train_sat_transforms,
                                transforms_gallery=train_drone_transforms,
                                prob_flip=0.5,
                                shuffle_batch_size=2,
                                )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=2,
                                  num_workers=0,
                                  shuffle=False,
                                  pin_memory=True)

    # Reference Satellite Images
    query_dataset_test = VisLocEval(data_folder=query_folder_test,
                                    mode="query",
                                    transforms=val_transforms,
                                    )

    query_dataloader_test = DataLoader(query_dataset_test,
                                       batch_size=2,
                                       num_workers=0,
                                       shuffle=False,
                                       pin_memory=True)
