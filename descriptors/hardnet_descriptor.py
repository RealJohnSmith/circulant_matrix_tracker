import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pylab
import results


def rgb2gray(rgb_image):
    "Based on http://stackoverflow.com/questions/12201577"
    # [0.299, 0.587, 0.144] normalized gives [0.29, 0.57, 0.14]
    return pylab.dot(rgb_image[:, :, :3], [0.29, 0.57, 0.14])


class hardnet:
    usegpu = False
    model = None
    upscale = False
    model_path = 'descriptors/pretrained/hardnetBr6.pth'


def initialize(usegpu):
    hardnet.usegpu = usegpu

    stride = 2
    hardnet.upscale = False

    model = DenseHardNet(stride)
    checkpoint = torch.load(hardnet.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if usegpu:
        hardnet.model = model.cuda()
    else:
        hardnet.model = model.cpu()

    results.log_meta("descriptor[ {} ].stride".format(get_name()), stride)
    results.log_meta("descriptor[ {} ].upscale".format(get_name()), hardnet.upscale)
    results.log_meta("descriptor[ {} ].model_path".format(get_name()), hardnet.model_path)

    return 128


def describe(image):
    gray = rgb2gray(image)
    var_image = torch.autograd.Variable(torch.from_numpy(gray.astype(np.float32)), volatile=True)
    var_image_reshape = var_image.view(1, 1, gray.shape[0], gray.shape[1])
    if hardnet.usegpu:
        var_image_reshape = var_image_reshape.cuda()

    desc = hardnet.model(var_image_reshape, hardnet.upscale).data.cpu().numpy()
    return desc[0]


def get_name():
    return "Hardnet"


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        hardnet.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1, keepdim = True) + hardnet.eps)
        x= x / norm.expand_as(x)
        return x


class LocalNorm2d(nn.Module):
    def __init__(self, kernel_size = 32):
        super(LocalNorm2d, self).__init__()
        hardnet.ks = kernel_size
        hardnet.pool = nn.AvgPool2d(kernel_size = hardnet.ks, stride = 1,  padding = 0)
        hardnet.eps = 1e-10
        return
    def forward(self,x):
        pd = int(hardnet.ks/2)
        mean = hardnet.pool(F.pad(x, (pd,pd,pd,pd), 'reflect'))
        return torch.clamp((x - mean) / (torch.sqrt(torch.abs(hardnet.pool(F.pad(x*x,  (pd,pd,pd,pd), 'reflect')) - mean*mean )) + hardnet.eps), min = -6.0, max = 6.0)


class DenseHardNet(nn.Module):
    """HardNet model definition"""
    def __init__(self, _stride = 2):
        super(DenseHardNet, self).__init__()
        hardnet.input_norm = LocalNorm2d(17)
        hardnet.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=_stride, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=_stride,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
            L2Norm()
        )
        return

    def forward(self, input, upscale = False):
        if input.size(1) > 1:
            feats = hardnet.features(hardnet.input_norm(input.mean(dim = 1, keepdim = True)))
        else:
            feats = hardnet.features(hardnet.input_norm(input))
        if upscale:
            return F.upsample(feats, (input.size(2), input.size(3)),mode='bilinear')
        return feats

