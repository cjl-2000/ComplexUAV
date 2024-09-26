import math
import random

import cv2
import numpy as np
from PIL import Image
from torchvision import datasets, transforms


class CenterCrop(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self):
        pass

    def __call__(self, img):
        img_ = np.array(img).copy()
        h, w, c = img_.shape
        min_edge = min((h, w))
        if min_edge == h:
            edge_lenth = int((w - min_edge) / 2)
            new_image = img_[:, edge_lenth:w - edge_lenth, :]
        else:
            edge_lenth = int((h - min_edge) / 2)
            new_image = img_[edge_lenth:h - edge_lenth, :, :]
        assert new_image.shape[0] == new_image.shape[1], "the shape is not correct"
        cv2.imshow("query", cv2.resize(new_image, (512, 512)))
        image = Image.fromarray(new_image.astype('uint8')).convert('RGB')
        return image


class RotateAndCrop(object):
    def __init__(self, rate, output_size=(512, 512), rotate_range=360):
        self.rate = rate
        self.output_size = output_size
        self.rotate_range = rotate_range

    def __call__(self, img):
        img_ = np.array(img).copy()

        def getPosByAngle(img, angle):
            h, w, c = img.shape
            y_center = h // 2
            x_center = w // 2
            r = h // 2
            angle_lt = angle - 45
            angle_rt = angle + 45
            angle_lb = angle + 135
            angle_rb = angle + 225
            angleList = [angle_lt, angle_rt, angle_lb, angle_rb]
            pointsList = []
            for angle in angleList:
                x1 = x_center + r * math.cos(angle * math.pi / 180)
                y1 = y_center + r * math.sin(angle * math.pi / 180)
                pointsList.append([x1, y1])
            pointsOri = np.float32(pointsList)
            pointsListAfter = np.float32(
                [[0, 0], [0, self.output_size[0]], [self.output_size[0], self.output_size[1]],
                 [self.output_size[1], 0]])
            M = cv2.getPerspectiveTransform(pointsOri, pointsListAfter)
            res = cv2.warpPerspective(
                img, M, (self.output_size[0], self.output_size[1]))
            return res

        if np.random.random() > self.rate:
            image = img
        else:
            angle = int(np.random.random() * self.rotate_range)
            image = getPosByAngle(img_, angle)
        return image


class RandomCrop(object):
    def __init__(self, rate=0.2):
        self.rate = rate

    def __call__(self, img):
        img_ = np.array(img).copy()
        h, w, c = img_.shape
        random_width = int(np.random.random() * self.rate * w)
        random_height = int(np.random.random() * self.rate * h)
        x_l = random_width
        x_r = w - random_width
        y_l = random_height
        y_r = h - random_height
        new_image = img_[y_l:y_r, x_l:x_r, :]
        image = Image.fromarray(new_image.astype('uint8')).convert('RGB')
        return image


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.3, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        img_ = np.array(img).copy()

        mean = np.mean(np.mean(img_, 0), 1)

        for _ in range(100):
            area = img_.shape[0] * img_.shape[1]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_.shape[1] and h < img_.shape[0]:
                x1 = random.randint(0, img_.shape[1] - h)
                y1 = random.randint(0, img_.shape[0] - w)

                img_[x1:x1 + h, y1:y1 + w, 0] = mean[0]
                img_[x1:x1 + h, y1:y1 + w, 1] = mean[1]
                img_[x1:x1 + h, y1:y1 + w, 2] = mean[2]
                img = Image.fromarray(img_.astype('uint8')).convert('RGB')
                return img

        return img
