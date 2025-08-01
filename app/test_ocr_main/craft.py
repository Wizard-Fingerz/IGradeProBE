# craft.py

import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict
import torch.nn.functional as F


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.stage5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )

    def forward(self, x):
        sources = []
        x = self.stage1(x)
        sources.append(x)
        x = self.stage2(x)
        sources.append(x)
        x = self.stage3(x)
        sources.append(x)
        x = self.stage4(x)
        sources.append(x)
        x = self.stage5(x)
        sources.append(x)
        return sources


class CRAFT(nn.Module):
    def __init__(self):
        super(CRAFT, self).__init__()
        self.basenet = VGG16FeatureExtractor()

        self.upconv1 = nn.Conv2d(1024, 512, 1, 1)
        self.upconv2 = nn.Conv2d(512, 256, 1, 1)
        self.upconv3 = nn.Conv2d(256, 128, 1, 1)
        self.upconv4 = nn.Conv2d(128, 64, 1, 1)

        self.conv_cls = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(16, 2, 1, 1)
        )

        init_weights(self.modules())

    def forward(self, x):
        sources = self.basenet(x)

        y = sources[-1]
        y = F.interpolate(y, size=sources[-2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[-2]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[-3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[-3]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[-4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[-4]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[-5].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[-5]], dim=1)
        y = self.upconv4(y)

        y = self.conv_cls(y)
        return y.permute(0, 2, 3, 1)  # [B, H, W, 2]
