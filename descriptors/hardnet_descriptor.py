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
    stride = 2
    model_path = 'descriptors/pretrained/all_datasets_HardNet++.pth'


def initialize(usegpu):
    hardnet.usegpu = usegpu

    hardnet.upscale = False

    model = DenseHardNet(hardnet.stride)
    checkpoint = torch.load(hardnet.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if usegpu:
        hardnet.model = model.cuda()
    else:
        hardnet.model = model.cpu()

    results.log_meta("descriptor[ {} ].stride".format(get_name()), hardnet.stride)
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


def update_roi(old_roi, moved_by):
    roi = old_roi
    roi[0] = round(moved_by[1] * hardnet.stride) + roi[0]
    roi[1] = round(moved_by[0] * hardnet.stride) + roi[1]
    return roi


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1, keepdim = True) + self.eps)
        x= x / norm.expand_as(x)
        return x

class LocalNorm2d(nn.Module):
    def __init__(self, kernel_size = 32):
        super(LocalNorm2d, self).__init__()
        self.ks = kernel_size
        self.pool = nn.AvgPool2d(kernel_size = self.ks, stride = 1,  padding = 0)
        self.eps = 1e-10
        return
    def forward(self,x):
        pd = int(self.ks/2)
        mean = self.pool(F.pad(x, (pd,pd,pd,pd), 'reflect'))
        return torch.clamp((x - mean) / (torch.sqrt(torch.abs(self.pool(F.pad(x*x,  (pd,pd,pd,pd), 'reflect')) - mean*mean )) + self.eps), min = -6.0, max = 6.0)

class DenseHardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self, _stride = 2):
        super(DenseHardNet, self).__init__()
        #self.input_norm = nn.InstanceNorm2d(1,affine = False)#(77)
        self.features = nn.Sequential(
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
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 128, kernel_size=8, stride = 1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            L2Norm()
        )
        return

    def forward(self, input, upscale = True):
        if input.size(1) > 1:
            feats = self.features(input)
        else:
            feats = self.features(input.mean(dim = 1, keepdim = True))
        return F.upsample(feats, (input.size(2),
                                  input.size(3)),mode='bilinear')