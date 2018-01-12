import torch.nn as nn
from torchvision import transforms, models


class ResNet(nn.Module):
    def __init__(self, model, num_classes=1000):
        super(ResNet, self).__init__()

        self.num_classes = num_classes

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv1', model.conv1)
        self.layer1.add_module('bn1', model.bn1)
        self.layer1.add_module('relu1', model.relu)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block1 = model.layer1
        self.block2 = model.layer2
        self.block3 = model.layer3
        self.block4 = model.layer4

        self.avgpool = model.avgpool

        self.classifier = model.fc

        if self.num_classes != 1000:
            self.classifier = nn.Linear(self.classifier.in_features, self.num_classes)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [{'params': self.layer1.parameters(), 'lr': lr * lrp},
                {'params': self.block1.parameters(), 'lr': lr * lrp},
                {'params': self.block2.parameters(), 'lr': lr * lrp},
                {'params': self.block3.parameters(), 'lr': lr * lrp},
                {'params': self.block4.parameters(), 'lr': lr * lrp},
                {'params': self.classifier.parameters()}]

    def get_transform(self, split):
        normalize = transforms.Normalize(mean=self.image_normalization_mean, std=self.image_normalization_std)
        if split == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        return transform


def resnet_50(num_classes=1000, pretrained='imagenet'):
    if pretrained == 'imagenet':
        model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50(pretrained=False)
    model = ResNet(model, num_classes)
    return model
