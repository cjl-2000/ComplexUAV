import timm

import torch.nn as nn

import torch.nn.functional as F

from model.ConvNext import convnext_base, convnext_tiny, convnext_small
from model.Head.head import SingleBranch, SingleBranchSwin, SingleBranchCNN
from model.Head.utils import ClassBlock
from model.SAFA import SAFA


class TimmClassModel(nn.Module):

    def __init__(self, args):

        super(TimmClassModel, self).__init__()
        self.img_size = args.img_size
        self.share = args.share

        self.kwargs = {'num_classes': None, 'drop_path_rate': 0.2}
        self.model, output_channel = self.init_backbone(args.model, args.use_safa)
        if not self.share:
            self.model_2, _ = self.init_backbone(args.model, args.use_safa)

        args.input_dim = output_channel
        self.head = self.init_head(args)

    def init_backbone(self, backbone, safa=True):
        if backbone == "resnet50":
            if safa:
                backbone_model = SAFA(timm.create_model('resnet50', pretrained=True),
                                      in_channel=(self.img_size[0] // 32) * (self.img_size[1] // 32))
            else:
                backbone_model = timm.create_model('resnet50', pretrained=True, num_classes=0)
                output_channel = 2048

        elif backbone == "ViTS-384":

            backbone_model = timm.models.create_model("vit_base_patch16_384", pretrained=True, num_classes=0)
            output_channel = 768

        elif backbone == "ViTB-384":

            backbone_model = timm.create_model("vit_base_patch16_384", pretrained=True, num_classes=0)
            output_channel = 768

        elif backbone == "convnext_b":
            if safa:
                backbone_model = SAFA(convnext_base(**self.kwargs),
                                      in_channel=(self.img_size[0] // 32) * (self.img_size[1] // 32))
            else:
                backbone_model = convnext_base(**self.kwargs)
                output_channel = 1024

        elif backbone == "convnext_t":
            if safa:
                backbone_model = SAFA(convnext_tiny(**self.kwargs),
                                      in_channel=(self.img_size[0] // 32) * (self.img_size[1] // 32))
            else:
                backbone_model = convnext_tiny(**self.kwargs)
                output_channel = 768
        elif backbone == "convnext_s":
            if safa:
                backbone_model = SAFA(convnext_small(**self.kwargs),
                                      in_channel=(self.img_size[0] // 32) * (self.img_size[1] // 32))
            else:
                backbone_model = convnext_small(**self.kwargs)
                output_channel = 768

        else:
            raise NameError("{} not in the backbone list!!!".format(backbone))
        return backbone_model, output_channel

    def init_head(self, args):
        head = ClassBlock(args.input_dim, args.class_num)
        return head

    def forward(self, img1=None, img2=None):
        if self.share:
            if img2 is not None:

                image_features1 = self.model(img1)
                image_features1 = self.head(image_features1)
                image_features2 = self.model(img2)
                image_features2 = self.head(image_features2)

                return image_features1, image_features2

            else:
                image_features = self.model(img1)
                image_features = self.head(image_features)
                return image_features
        else:
            if img1 is None:
                image_features = self.model_2(img2)
                image_features = self.head(image_features)
                return F.normalize(image_features, dim=-1)
            if img2 is None:
                image_features = self.model(img1)
                image_features = self.head(image_features)
                return image_features

            if img1 and img2:
                image_features1 = self.model(img1)
                image_features1 = self.head(image_features1)
                image_features2 = self.model_2(img2)
                image_features2 = self.head(image_features2)

                return image_features1, image_features2
