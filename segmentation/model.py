import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from segmentation.resnet import resnet_50


class ResNetHypercolumn5PSP(nn.Module):
    def __init__(self, model, num_classes, upsampling_mode='bilinear', final_size=None, scales=[1], fusion='sum',
                 multi_outputs=False, dim=256, dropout_p=0.0, dropout2d=False, pyramid_sizes=[1, 2, 3, 6],
                 pyramid_pooling='avg', groups=1):
        super(ResNetHypercolumn5PSP, self).__init__()

        self.upsampling_mode = upsampling_mode
        self.num_classes = num_classes
        self.final_size = final_size
        self.scales = scales
        self.fusion = fusion
        self.multi_outputs = multi_outputs
        self.dim = dim
        self.dropout_p = dropout_p
        self.dropout2d = dropout2d
        self.pyramid_sizes = pyramid_sizes
        self.pyramid_pooling = pyramid_pooling
        self.groups = groups

        self.scale_augmentation = nn.Sequential()
        if scales != [1]:
            self.scale_augmentation.add_module('random_scale', RandomScale(self.scales))

        self.layer1 = model.layer1
        self.maxpool = model.maxpool
        self.block1 = model.block1
        self.block2 = model.block2
        self.block3 = model.block3
        self.block4 = model.block4

        num_features = self.layer1.conv1.out_channels
        self.classifier0 = self.make_classifier(num_features)
        num_features = self.block1[1].conv1.in_channels
        self.classifier1 = self.make_classifier(num_features)
        num_features = self.block2[1].conv1.in_channels
        self.classifier2 = self.make_classifier(num_features)
        num_features = self.block3[1].conv1.in_channels
        self.classifier3 = self.make_classifier(num_features)
        num_features = self.block4[1].conv1.in_channels
        self.classifier4 = self.make_classifier(num_features)

        self.upsample = nn.Upsample(mode=self.upsampling_mode)
        self.final_upsample = nn.Upsample(mode=self.upsampling_mode, size=self.final_size)

        # image normalization
        self.image_normalization_mean = model.image_normalization_mean
        self.image_normalization_std = model.image_normalization_std

    def forward(self, x):
        if self.final_size is not None:
            self.final_upsample.size = self.final_size
        else:
            x_size = x.size()
            self.final_upsample.size = (x_size[2], x_size[3])
        x = self.scale_augmentation(x)
        x0 = self.layer1(x)
        x0_pool = self.maxpool(x0)
        x1 = self.block1(x0_pool)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x0_size = x0.size()
        self.upsample.size = (x0_size[2], x0_size[3])

        x0s = self.classifier0(x0)
        x1s = self.upsample(self.classifier1(x1))
        x2s = self.upsample(self.classifier2(x2))
        x3s = self.upsample(self.classifier3(x3))
        x4s = self.upsample(self.classifier4(x4))

        if self.fusion == 'sum':
            x_dense = x0s + x1s + x2s + x3s + x4s
        elif self.fusion == 'avg':
            x_dense = (x0s + x1s + x2s + x3s + x4s) / 5
        elif self.fusion == 'max':
            x_dense = torch.max(torch.max(torch.max(torch.max(x0s, x1s), x2s), x3s), x4s)

        x_dense = self.final_upsample(x_dense)

        if self.multi_outputs:
            return x_dense, self.final_upsample(x0s), self.final_upsample(x1s), self.final_upsample(
                x2s), self.final_upsample(x3s), self.final_upsample(x4s)
        return x_dense

    def make_classifier(self, num_features):
        classifier = nn.Sequential()
        classifier.add_module('pyramid_pooling',
                              PyramidPoolingModule(num_features, upsampling_mode='bilinear',
                                                   pyramid_sizes=self.pyramid_sizes,
                                                   pooling=self.pyramid_pooling))
        num_inputs = num_features + int(num_features / len(self.pyramid_sizes)) * len(self.pyramid_sizes)
        classifier.add_module('conv1',
                              nn.Conv2d(num_inputs, self.dim, kernel_size=3, stride=1, padding=2, dilation=2,
                                        bias=False))
        classifier.add_module('bn1', nn.BatchNorm2d(self.dim))
        classifier.add_module('relu1', nn.ReLU())
        if self.dropout_p > 0:
            if self.dropout2d:
                classifier.add_module('dropout', nn.Dropout2d(self.dropout_p))
            else:
                classifier.add_module('dropout', nn.Dropout(self.dropout_p))
        classifier.add_module('conv2', nn.Conv2d(self.dim, self.num_classes, kernel_size=1, stride=1))
        return classifier

    def set_final_size(self, final_size):
        self.final_size = final_size
        self.final_upsample.size = self.final_size

    def dense_prediction(self, x):
        return self.forward(x)

    def get_config_optim(self, lr, lrp):
        return [{'params': self.layer1.parameters(), 'lr': lr * lrp},
                {'params': self.block1.parameters(), 'lr': lr * lrp},
                {'params': self.block2.parameters(), 'lr': lr * lrp},
                {'params': self.block3.parameters(), 'lr': lr * lrp},
                {'params': self.block4.parameters(), 'lr': lr * lrp},
                {'params': self.classifier0.parameters()},
                {'params': self.classifier1.parameters()},
                {'params': self.classifier2.parameters()},
                {'params': self.classifier3.parameters()},
                {'params': self.classifier4.parameters()}]


class RandomScale(nn.Module):
    def __init__(self, scales=[0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3], upsampling_mode='bilinear'):
        super(RandomScale, self).__init__()

        self.scales = scales
        self.upsampling_mode = upsampling_mode
        self.upsample = nn.Upsample(mode=self.upsampling_mode)

    def forward(self, input):
        x_size = input.size()
        if self.training:
            scale = random.choice(self.scales)
            self.upsample.size = (int(scale * x_size[2]), int(scale * x_size[3]))
        else:
            self.upsample.size = (x_size[2], x_size[3])
        output = self.upsample(input)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(mode={mode}, scales={scales})'.format(mode=self.upsampling_mode,
                                                                                 scales=self.scales)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pyramid_sizes=[1, 2, 3, 6], upsampling_mode='bilinear', pooling='avg'):
        super(PyramidPoolingModule, self).__init__()

        self.pyramid_sizes = pyramid_sizes
        self.upsampling_mode = upsampling_mode
        self.pooling = pooling

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, size) for size in pyramid_sizes])

    def _make_stage(self, in_channels, size):
        if self.pooling == 'avg':
            pool = nn.AdaptiveAvgPool2d(output_size=(size, size))
        else:
            pool = None
            print('incorrect pooling:', self.pooling)
        dim_output = int(in_channels / len(self.pyramid_sizes))
        stage = nn.Sequential()
        stage.add_module('pool', pool)
        stage.add_module('conv', nn.Conv2d(in_channels, dim_output, kernel_size=1, bias=False))
        # stage.add_module('relu', nn.ReLU())
        return stage

    def forward(self, input):
        h, w = input.size(2), input.size(3)
        output = [F.upsample(input=stage(input), size=(h, w), mode=self.upsampling_mode) for stage in self.stages] + [
            input]
        output = torch.cat(output, 1)
        return output


def resnet_50_hc5_psp(num_classes, pretrained='imagenet', upsampling_mode='bilinear', final_size=None, scales=[1],
                      fusion='sum', multi_outputs=False, dim=256, dropout_p=0.0, dropout2d=False,
                      pyramid_sizes=[1, 2, 3, 6], pyramid_pooling='avg'):
    model = resnet_50(pretrained=pretrained)
    return ResNetHypercolumn5PSP(model, num_classes, upsampling_mode=upsampling_mode, final_size=final_size,
                                 scales=scales, fusion=fusion, multi_outputs=multi_outputs, dim=dim,
                                 dropout_p=dropout_p, dropout2d=dropout2d, pyramid_sizes=pyramid_sizes,
                                 pyramid_pooling=pyramid_pooling)