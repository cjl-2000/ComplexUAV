
import timm

import torch.nn as nn

import torch.nn.functional as F


from model.ConvNext import convnext_base, convnext_tiny, convnext_small
from model.SAFA import SAFA


# class TimmModel(nn.Module):
#
#     def __init__(self,
#                  model_name,
#                  pretrained=True,
#                  img_size=384):
#
#         super(TimmModel, self).__init__()
#
#         self.img_size = img_size
#         kwargs = {'num_classes': None, 'drop_path_rate': 0.2}
#         if "convnext" in model_name:
#             # automatically change interpolate pos-encoding to img_size
#             # self.model = convnext_base(**kwargs)
#             self.model = SAFA(convnext_base(**kwargs), in_channel=(self.img_size // 32) * (self.img_size // 32))
#         else:
#             self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
#         self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#     def get_config(self, ):
#         data_config = timm.data.resolve_model_data_config(self.model)
#         return data_config
#
#     def set_grad_checkpointing(self, enable=True):
#         self.model.set_grad_checkpointing(enable)
#
#     def forward(self, img1, img2=None):
#
#         if img2 is not None:
#
#             image_features1 = self.model(img1)
#             image_features2 = self.model(img2)
#
#             return F.normalize(image_features1, dim=-1), F.normalize(image_features2, dim=-1)
#
#         else:
#             image_features = self.model(img1)
#
#             return  F.normalize(image_features, dim=-1)
#


class TimmModel(nn.Module):

    def __init__(self, args):

        super(TimmModel, self).__init__()
        self.img_size = args.img_size
        self.share = args.share
        self.kwargs = {'num_classes': None, 'drop_path_rate': 0.2}
        self.model = self.init_backbone(args.model, args.use_safa)

        if not self.share:
            self.model_2, _ = self.init_backbone(args.model, args.use_safa)

    def init_backbone(self, backbone, safa=True):
        if backbone == "resnet50":
            if safa:
                backbone_model = SAFA(timm.create_model('resnet50', pretrained=True),
                                      in_channel=(self.img_size[0] // 32) * (self.img_size[1] // 32))
            else:
                backbone_model = timm.create_model('resnet50', pretrained=True)

        elif backbone == "ViTS-384":

            backbone_model = timm.create_model("vit_small_patch16_384", pretrained=True, num_classes=0)

        elif backbone == "seresnet50":

            backbone_model = timm.create_model("seresnet50", pretrained=True, num_classes=0)

        elif backbone == "ViTB-384":

            backbone_model = timm.create_model("vit_base_patch16_384", pretrained=True, num_classes=0)

        elif backbone == "convnext_b":
            if safa:
                backbone_model = SAFA(convnext_base(**self.kwargs),
                                      in_channel=(self.img_size[0] // 32) * (self.img_size[1] // 32))
            else:
                backbone_model = convnext_base(**self.kwargs)

        elif backbone == "convnext_t":
            if safa:
                backbone_model = SAFA(convnext_tiny(**self.kwargs),
                                      in_channel=(self.img_size[0] // 32) * (self.img_size[1] // 32))
            else:
                backbone_model = convnext_tiny(**self.kwargs)

        elif backbone == "convnext_s":
            if safa:
                backbone_model = SAFA(convnext_small(**self.kwargs),
                                      in_channel=(self.img_size[0] // 32) * (self.img_size[1] // 32))
            else:
                backbone_model = convnext_small(**self.kwargs)


        else:
            raise NameError("{} not in the backbone list!!!".format(backbone))
        return backbone_model

    def forward(self, img1, img2=None):
        if self.share:
            if img2 is not None:

                image_features1 = self.model(img1)
                image_features2 = self.model(img2)
                return F.normalize(image_features1, dim=-1), F.normalize(image_features2, dim=-1)

            else:
                image_features = self.model(img1)
                return F.normalize(image_features, dim=-1)
        else:
            if img1 is None:
                image_features = self.model_2(img2)
                return F.normalize(image_features, dim=-1)
            if img2 is None:
                image_features = self.model(img1)
                return F.normalize(image_features, dim=-1)

            if img1 and img2:
                image_features1 = self.model(img1)
                image_features2 = self.model_2(img2)
                return F.normalize(image_features1, dim=-1), F.normalize(image_features2, dim=-1)
