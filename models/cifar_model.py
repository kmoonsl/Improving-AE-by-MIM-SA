import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.sa1 = sa_layer(64)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.sa2 = sa_layer(128)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.sa3 = sa_layer(256)

        self.conv4 = nn.Conv2d(256, 16, 4, 1, 0)

    def forward(self, x):
        x = F.leaky_relu(self.sa1(self. conv1_bn(self.conv1(x))), 0.2)
        x = F.leaky_relu(self.sa2(self.conv2_bn(self.conv2(x))), 0.2)
        x = F.leaky_relu(self.sa3(self.conv3_bn(self.conv3(x))), 0.2)
        x = self.conv4(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv1_1 = nn.ConvTranspose2d(20, 256, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(256)
        self.sa1 = sa_layer(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(128)
        self.sa2 = sa_layer(128)

        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(64)
        self.sa3 = sa_layer(64)

        self.deconv4 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

    def forward(self, x):
        x = F.relu(self.sa1(self.deconv1_1_bn(self.deconv1_1(x))))
        x = F.relu(self.sa2(self.deconv2_bn(self.deconv2(x))))
        x = F.relu(self.sa3(self.deconv3_bn(self.deconv3(x))))
        img = torch.tanh(self.deconv4(x)) * 0.5 + 0.5

        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, inplace=True)

        return x

class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(256, 1, 4, 1, 0)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))

        return output

class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(256, 128, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 4, 1)
        self.conv_mu = nn.Conv2d(128, 4, 1)
        self.conv_var = nn.Conv2d(128, 4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        input = F.softmax(disc_logits,0)

        return disc_logits, mu, var

class ZDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(16, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)


    def forward(self, x):
        x = F.leaky_relu((self.linear1(x)), 0.2)
        x = F.dropout(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=32):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0,x_1 = x.chunk(2,dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out