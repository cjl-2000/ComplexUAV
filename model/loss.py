import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn


class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = image_features1 / torch.norm(image_features1, dim=1, keepdim=True)
        image_features2 = image_features2 / torch.norm(image_features2, dim=1, keepdim=True)
        logits_per_image1 = logit_scale * torch.matmul(image_features1, image_features2.T)
        labels = torch.arange(image_features1.size(0), device=self.device)

        loss_i = self.loss_function(logits_per_image1, labels)
        loss_t = self.loss_function(logits_per_image1.T, labels)
        # loss_i = F.cross_entropy(logits_per_image1, labels)
        # loss_t = F.cross_entropy(logits_per_image1.T, labels)
        loss = loss_i + loss_t / 2
        return loss
