#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is a python reimplementation of the open source tracker in
http://www2.isr.uc.pt/~henriques/circulant/index.html

Found http://wiki.scipy.org/NumPy_for_Matlab_Users very useful

Based on the work of João F. Henriques, 2012
http://www.isr.uc.pt/~henriques

Rodrigo Benenson, MPI-Inf 2013
http://rodrigob.github.io
"""

from __future__ import print_function

import os
import os.path
import sys
import glob
import time
from optparse import OptionParser
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.misc
import matplotlib
usegpu=False

#matplotlib.use("Agg")
#usegpu=True

import pylab
import numpy as np
import gc

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
        self.input_norm = LocalNorm2d(17)
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
            feats = self.features(self.input_norm(input.mean(dim = 1, keepdim = True)))
        else:
            feats = self.features(self.input_norm(input))
        if upscale:
            return F.upsample(feats, (input.size(2), input.size(3)),mode='bilinear')
        return feats

class CirculantMatrixTracker:

    def __init__(self, object_example):
        """
        object_example is an image showing the object to track
        """

        return

    def find(self, image):
        """
        Will return the x/y coordinates where the object was found,
        and the score
        """

        return

    def update_template(self, new_example, forget_factor=1):
        """
        Update the tracking template,
        new_example is expected to match the size of
        the example provided to the constructor
        """

        return


def load_video_info_bbabenko(video_path):
    """
    Loads all the relevant information for the video in the given path:
    the list of image files (cell array of strings), initial position
    (1x2), target size (1x2), whether to resize the video to half
    (boolean), and the ground truth information for precision calculations
    (Nx2, for N frames). The ordering of coordinates is always [y, x].
    The path to the video is returned, since it may change if the images
    are located in a sub-folder (as is the default for MILTrack's videos).
    """

    # load ground truth from text file (MILTrack's format)
    text_files = glob.glob(os.path.join(video_path, "*_gt.txt"))
    assert text_files, \
        "No initial position and ground truth (*_gt.txt) to load."

    first_file_path = text_files[0]
    #f = open(first_file_path, "r")
    #ground_truth = textscan(f, '%f,%f,%f,%f') # [x, y, width, height]
    #ground_truth = cat(2, ground_truth{:})
    ground_truth = pylab.loadtxt(first_file_path, delimiter=",")
    #f.close()

    # set initial position and size
    first_ground_truth = ground_truth[0, :]
    # target_sz contains height, width
    target_sz = pylab.array([first_ground_truth[3], first_ground_truth[2]])
    # pos contains y, x center
    pos = [first_ground_truth[1], first_ground_truth[0]] \
          + pylab.floor(target_sz / 2)

    #try:
    if True:
        # interpolate missing annotations
        # 4 out of each 5 frames is filled with zeros
        for i in range(4):  # x, y, width, height
            xp = range(0, ground_truth.shape[0], 5)
            fp = ground_truth[xp, i]
            x = range(ground_truth.shape[0])
            ground_truth[:, i] = pylab.interp(x, xp, fp)
        # store positions instead of boxes
        ground_truth = ground_truth[:, [1, 0]] + ground_truth[:, [3, 2]] / 2
    #except Exception as e:
    else:
        print("Failed to gather ground truth data")
        #print("Error", e)
        # ok, wrong format or we just don't have ground truth data.
        ground_truth = []

    # list all frames. first, try MILTrack's format, where the initial and
    # final frame numbers are stored in a text file. if it doesn't work,
    # try to load all png/jpg files in the folder.

    text_files = glob.glob(os.path.join(video_path, "*_frames.txt"))[0]
    if text_files:
        #f = open(first_file_path, "r")
        #frames = textscan(f, '%f,%f')
        frames = pylab.loadtxt(text_files, delimiter=",", dtype=int)
        #f.close()


        # see if they are in the 'imgs' subfolder or not
        test1_path_to_img = os.path.join(video_path,
                                         "imgs/img%05i.png" % frames[0])
        test2_path_to_img = os.path.join(video_path,
                                         "img%05i.png" % frames[0])
        if os.path.exists(test1_path_to_img):
            video_path = os.path.join(video_path, "imgs/")
        elif os.path.exists(test2_path_to_img):
            video_path = video_path  # no need for change
        else:
            raise Exception("Failed to find the png images")

        # list the files
        img_files = [video_path + "/img%05i.png" % i
                     for i in range(frames[0], frames[1] + 1)]

        #img_files = num2str((frames{1} : frames{2})', 'img%05i.png')
        #img_files = cellstr(img_files);
    else:
        # no text file, just list all images
        img_files = glob.glob(os.path.join(video_path, "*.png"))
        if len(img_files) == 0:
            img_files = glob.glob(os.path.join(video_path, "*.jpg"))

        assert len(img_files), "Failed to find png or jpg images"

        img_files.sort()

    # if the target is too large, use a lower resolution
    # no need for so much detail
    if pylab.sqrt(pylab.prod(target_sz)) >= 100:
        pos = pylab.floor(pos / 2)
        target_sz = pylab.floor(target_sz / 2)
        resize_image = True
    else:
        resize_image = False


    ret = [img_files, pos, target_sz, resize_image, ground_truth, video_path]
    return ret



def load_video_info_vot(video_path):
    """
    Loads all the relevant information for the video in the given path:
    the list of image files (cell array of strings), initial position
    (1x2), target size (1x2), whether to resize the video to half
    (boolean), and the ground truth information for precision calculations
    (Nx2, for N frames). The ordering of coordinates is always [y, x].
    The path to the video is returned, since it may change if the images
    are located in a sub-folder (as is the default for MILTrack's videos).
    """

    # load ground truth from text file (MILTrack's format)
    gt_file_path = os.path.join(video_path, "groundtruth.txt")

    with open(gt_file_path) as f:
        ground_truth = [map(float, x.strip().split(",")) for x in f.readlines()]
        ground_truth = map(lambda coords: [coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1]], map(lambda coords: [min(coords[0], coords[2], coords[4], coords[6]), min(coords[1], coords[3], coords[5], coords[7]), max(coords[0], coords[2], coords[4], coords[6]), max(coords[1], coords[3], coords[5], coords[7])], ground_truth))
        first_ground_truth = ground_truth[0]
        ground_truth = map(lambda coords: [coords[1] + coords[3] / 2, coords[0] + coords[2] / 2], ground_truth)

    # set initial position and size
    target_sz = pylab.array([first_ground_truth[3], first_ground_truth[2]])
    # pos contains y, x center
    pos = [first_ground_truth[1], first_ground_truth[0]] \
        + pylab.floor(target_sz / 2)

    img_files = glob.glob(os.path.join(video_path, "*.png"))
    if len(img_files) == 0:
        img_files = glob.glob(os.path.join(video_path, "*.jpg"))

    assert len(img_files), "Failed to find png or jpg images"

    img_files.sort()

    # if the target is too large, use a lower resolution
    # no need for so much detail
    if pylab.sqrt(pylab.prod(target_sz)) >= 100 and False:
        pos = pylab.floor(pos / 2)
        target_sz = pylab.floor(target_sz / 2)
        ground_truth = map(lambda coords: [coords[0] / 2, coords[1] / 2], ground_truth)
        resize_image = True
    else:
        resize_image = False

    outdir = video_path.replace("data/sets", "data/logs")
    if not os.path.exists(video_path.replace("data/sets", "data/logs")):
          os.makedirs(outdir)
    ret = [img_files, pos, target_sz, resize_image, ground_truth, video_path, outdir]
    return ret


def rgb2gray(rgb_image):
    "Based on http://stackoverflow.com/questions/12201577"
    # [0.299, 0.587, 0.144] normalized gives [0.29, 0.57, 0.14]
    return pylab.dot(rgb_image[:, :, :3], [0.29, 0.57, 0.14])


def get_subwindow(im, pos, sz, model, upscale):
    """
    Obtain sub-window from image, with replication-padding.
    Returns sub-window of image IM centered at POS ([y, x] coordinates),
    with size SZ ([height, width]). If any pixels are outside of the image,
    they will replicate the values at the borders.

    The subwindow is also normalized to range -0.5 .. 0.5, and the given
    cosine window COS_WINDOW is applied
    (though this part could be omitted to make the function more general).
    """

    if pylab.isscalar(sz):  # square sub-window
        sz = [sz, sz]

    ys = pylab.floor(pos[0]) \
        + pylab.arange(sz[0], dtype=int) - pylab.floor(sz[0]/2)
    xs = pylab.floor(pos[1]) \
        + pylab.arange(sz[1], dtype=int) - pylab.floor(sz[1]/2)

    ys = ys.astype(int)
    xs = xs.astype(int)

    # check for out-of-bounds coordinates,
    # and set them to the values at the borders
    ys[ys < 0] = 0
    ys[ys >= im.shape[2]] = im.shape[2] - 1

    xs[xs < 0] = 0
    xs[xs >= im.shape[3]] = im.shape[3] - 1
    #zs = range(im.shape[2])

    # extract image
    #out = im[pylab.ix_(ys, xs, zs)]
    out = im[pylab.ix_(range(im.shape[0]), range(im.shape[1]), ys, xs)] - 0.5
    desc = model(out, upscale).data.cpu().numpy()

    global cos_window, cos_window_defined

    if not cos_window_defined:
        cos_window = pylab.outer(pylab.hanning(desc.shape[2]), pylab.hanning(desc.shape[3]))
        cos_window_defined = True

    desc[:, :] = pylab.multiply(desc[:, :], cos_window)

    return desc

    #pre-process window --
    # normalize to range -0.5 .. 0.5
    # pixels are already in range 0 to 1
    #out = out.astype(pylab.float64) - 0.5

    # apply cosine window
    #out = pylab.multiply(cos_window, out)






def dense_gauss_kernel(sigma, x, y=None):
    """
    Gaussian Kernel with dense sampling.
    Evaluates a gaussian kernel with bandwidth SIGMA for all displacements
    between input images X and Y, which must both be MxN. They must also
    be periodic (ie., pre-processed with a cosine window). The result is
    an MxN map of responses.

    If X and Y are the same, ommit the third parameter to re-use some
    values, which is faster.
    """

    xf = pylab.fft2(x)  # x in Fourier domain
    x_flat = x.flatten()
    xx = pylab.dot(x_flat.transpose(), x_flat)  # squared norm of x

    if y is not None:
        # general case, x and y are different
        yf = pylab.fft2(y)
        y_flat = y.flatten()
        yy = pylab.dot(y_flat.transpose(), y_flat)
    else:
        # auto-correlation of x, avoid repeating a few operations
        yf = xf
        yy = xx

    # cross-correlation term in Fourier domain
    xyf = pylab.multiply(xf, pylab.conj(yf))

    # to spatial domain
    xyf_ifft = pylab.ifft2(xyf)
    #xy_complex = circshift(xyf_ifft, floor(x.shape/2))
    row_shift, col_shift = pylab.floor(pylab.array(x.shape)/2).astype(int)
    xy_complex = pylab.roll(xyf_ifft, row_shift, axis=0)
    xy_complex = pylab.roll(xy_complex, col_shift, axis=1)
    xy = pylab.real(xy_complex)

    # calculate gaussian response for all positions
    scaling = -1 / (sigma**2)
    xx_yy = xx + yy
    xx_yy_2xy = xx_yy - 2 * xy
    k = pylab.exp(scaling * pylab.maximum(0, xx_yy_2xy / x.size))

    #print("dense_gauss_kernel x.shape =", x.shape)
    #print("dense_gauss_kernel k.shape =", k.shape)

    return k


def show_precision(positions, ground_truth, video_path, title):
    """
    Calculates precision for a series of distance thresholds (percentage of
    frames where the distance to the ground truth is within the threshold).
    The results are shown in a new figure.

    Accepts positions and ground truth as Nx2 matrices (for N frames), and
    a title string.
    """

    print("Evaluating tracking results.")

    pylab.ioff()  # interactive mode off

    max_threshold = 50  # used for graphs in the paper

    # calculate distances to ground truth over all frames
    delta = positions - ground_truth
    distances = pylab.sqrt((delta[:, 0]**2) + (delta[:, 1]**2))

    #distances[pylab.isnan(distances)] = []

    # compute precisions
    precisions = pylab.zeros((max_threshold, 1), dtype=float)
    for p in range(max_threshold):
        precisions[p] = pylab.sum(distances <= p, dtype=float) / len(distances)

    if False:
        pylab.figure()
        pylab.plot(distances)
        pylab.title("Distances")
        pylab.xlabel("Frame number")
        pylab.ylabel("Distance")

    # plot the precisions
    pylab.figure()  # 'Number', 'off', 'Name',
    pylab.title("Precisions - " + title)
    pylab.plot(precisions, "k-", linewidth=2)
    pylab.xlabel("Threshold")
    pylab.ylabel("Precision")

    pylab.show()
    return


def plot_tracking(frame, pos, target_sz, im, ground_truth, cached, outpath):

    global \
        tracking_figure, tracking_figure_title, tracking_figure_axes, \
        tracking_rectangle, gt_point, \
        z_figure_axes, response_figure_axes

    timeout = 1e-6
    #timeout = 0.05  # uncomment to run slower
    if frame == 0:
        #pylab.ion()  # interactive mode on
        tracking_figure = pylab.figure()
        gs = pylab.GridSpec(1, 3, width_ratios=[3, 1, 1])

        tracking_figure_axes = tracking_figure.add_subplot(gs[0])
        tracking_figure_axes.set_title("Tracked object (and ground truth)")

        z_figure_axes = tracking_figure.add_subplot(gs[1])
        z_figure_axes.set_title("Template")

        response_figure_axes = tracking_figure.add_subplot(gs[2])
        response_figure_axes.set_title("Response")

        tracking_rectangle = pylab.Rectangle((0, 0), 0, 0)
        tracking_rectangle.set_color((1, 1, 0, 0.5))
        tracking_figure_axes.add_patch(tracking_rectangle)

        gt_point = pylab.Circle((0, 0), radius=5)
        gt_point.set_color((0, 0, 1, 0.5))
        tracking_figure_axes.add_patch(gt_point)

        tracking_figure_title = tracking_figure.suptitle("")

        pylab.show(block=False)

    elif tracking_figure is None:
        return  # we simply go faster by skipping the drawing
    elif not pylab.fignum_exists(tracking_figure.number):
        #print("Drawing window closed, end of game. "
        #      "Have a nice day !")
        #sys.exit()
        print("From now on drawing will be omitted, "
              "so that computation goes faster")
        tracking_figure = None
        return

    global z, response
    tracking_figure_axes.imshow(im, cmap=pylab.cm.gray)


    rect_y, rect_x = tuple(pos - target_sz/2.0)
    rect_height, rect_width = target_sz
    tracking_rectangle.set_xy((rect_x, rect_y))
    tracking_rectangle.set_width(rect_width)
    tracking_rectangle.set_height(rect_height)

    if len(ground_truth) > 0:
        gt = ground_truth[frame]
        gt_y, gt_x = gt
        gt_point.center = (gt_x, gt_y)

    if z[0][0] is not None:
        z_figure_axes.imshow(z[0][0], cmap=pylab.cm.hot)

    if response[0][0] is not None:
        response_figure_axes.imshow(response[0][0], cmap=pylab.cm.hot)

    tracking_rectangle.set_color((0 if not cached else 1, 0.5 if not cached else 0, 0, 0.7 if not cached else 0.2))


    tracking_figure_title.set_text("Frame %i (out of %i)" % (frame + 1, len(ground_truth)))

    #tracking_figure.canvas.draw()  # update
    pylab.draw()

    if frame != 0:
        pylab.savefig(outpath + '/image%05i.jpg' % frame, bbox_inches='tight')

    pylab.waitforbuttonpress(timeout=timeout)

    return


def track(input_video_path, usegpu):

    stride = 2
    upscale = False

    model_weights = '../hardnet/pretrained/pretrained_all_datasets/HardNet++.pth'
    model = DenseHardNet(stride)
    checkpoint = torch.load(model_weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if usegpu:
        model = model.cuda()
    else:
        model = model.cpu()


    """
    notation: variables ending with f are in the frequency domain.
    """

    # parameters according to the paper --
    padding = 1.0  # extra area surrounding the target
    #spatial bandwidth (proportional to target)
    output_sigma_factor = 1 / float(16)
    sigma = 0.2  # gaussian kernel bandwidth
    lambda_value = 1e-2  # regularization
    # linear interpolation factor for adaptation
    interpolation_factor = 0.075

    info = load_video_info_vot(input_video_path)
    img_files, pos, target_sz, \
        should_resize_image, ground_truth, video_path, outpath = info

    # window size, taking padding into account
    sz = pylab.floor(target_sz * (1 + padding))

    # desired output (gaussian shaped), bandwidth proportional to target size
    output_sigma = pylab.sqrt(pylab.prod(target_sz)) * output_sigma_factor


    avgCount = 0

    total_time = 0  # to calculate FPS
    positions = pylab.zeros((len(img_files), 2))  # to calculate precision

    global z, response, cos_window, cos_window_defined
    z = [[None for i in range(128)] for c in range(3)]
    alphaf = [[None for i in range(128)] for c in range(3)]
    response = [[None for i in range(128)] for c in range(3)]
    cos_window = None
    cos_window_defined = False


    for frame, image_path in enumerate(img_files):

        if (frame % 10) == 0:
            print("Processing frame", frame)

        # load image
        im = pylab.imread(image_path)

        im = im / 255.0

        #if len(im.shape) == 3 and im.shape[2] > 1:
        # im = rgb2gray(im)

        if should_resize_image:
            im = scipy.misc.imresize(im, 0.5)

        start_time = time.time()

        var_image = torch.autograd.Variable(torch.from_numpy(im.astype(np.float32)), volatile=True)
        var_image_reshape = var_image.view(3, 1, var_image.size(0), var_image.size(1))
        if (usegpu):
            var_image_reshape = var_image_reshape.cuda()

        is_first_frame = frame == 0


        subwindow = get_subwindow(var_image_reshape, pos, sz, model, upscale)

        print(subwindow.shape)

        if is_first_frame:
            grid_y = pylab.arange(subwindow.shape[2]) - pylab.floor(subwindow.shape[2]/2)
            grid_x = pylab.arange(subwindow.shape[3]) - pylab.floor(subwindow.shape[3]/2)

            rs, cs = pylab.meshgrid(grid_x, grid_y)
            y = pylab.exp(-0.5 / output_sigma**2 * (rs**2 + cs**2))
            yf = pylab.fft2(y)



        if not is_first_frame:

            row, col = 0, 0
            avgCount = 0

            for c in range(0, subwindow.shape[0]):
                for i in range(0, subwindow.shape[1]):
                    xchannel = subwindow[c, i, :, :]

                    # calculate response of the classifier at all locations
                    k = dense_gauss_kernel(sigma, xchannel, z[c] [i])
                    kf = pylab.fft2(k)
                    alphaf_kf = pylab.multiply(alphaf[c][i], kf)
                    response[c][i] = pylab.real(pylab.ifft2(alphaf_kf))  # Eq. 9

                    # target location is at the maximum response
                    r = response[c][i]
                    argmax = r.argmax()

                    if r.item(argmax) != 0:
                        tmp = pylab.unravel_index(argmax, r.shape)
                        row += tmp[1]
                        col += tmp[0]
                        avgCount += 1


            if avgCount > 0:
                pos = pos - pylab.floor(sz/2) + [col * sz[0] / avgCount / subwindow.shape[2], row * sz[1] / avgCount / subwindow.shape[3]]

        # end "if not first frame"

        # get subwindow at current estimated target position,
        # to train classifer
        subwindow = get_subwindow(var_image_reshape, pos, sz, model, upscale)

        for c in range(0, subwindow.shape[0]):
            for i in range(0, subwindow.shape[1]):

                xchannel = subwindow[c, i, :, :]

                # Kernel Regularized Least-Squares,
                # calculate alphas (in Fourier domain)
                k = dense_gauss_kernel(sigma, xchannel)
                new_alphaf = pylab.divide(yf, (pylab.fft2(k) + lambda_value))  # Eq. 7
                new_z = xchannel

                if is_first_frame:
                    #first frame, train with a single image
                    alphaf[c][i] = new_alphaf
                    z[c][i] = xchannel
                else:
                    # subsequent frames, interpolate model
                    f = interpolation_factor
                    alphaf[c][i] = (1 - f) * alphaf[c][i] + f * new_alphaf
                    z[c][i] = (1 - f) * z[c][i] + f * new_z
                # end "first frame or not"

        # save position and calculate FPS
        positions[frame, :] = pos
        total_time += time.time() - start_time


        # visualization
        plot_tracking(frame, pos, target_sz, im, ground_truth, avgCount == 0, outpath)
    # end of "for each image in video"

    print("Frames-per-second:",  len(img_files) / total_time)

    title = os.path.basename(os.path.normpath(input_video_path))

    if len(ground_truth) > 0:
        # show the precisions plot
        show_precision(positions, ground_truth, video_path, title)

    return

def parse_arguments():

    parser = OptionParser()
    parser.description = \
        "This program will track objects " \
        "on videos in the MILTrack paper format. " \
        "See http://goo.gl/pSTo9r"

    parser.add_option("-i", "--input", dest="video_path",
                      metavar="PATH", type="string", default=None,
                      help="path to a folder o a MILTrack video")


    (options, args) = parser.parse_args()
    #print (options, args)

    if not options.video_path:
        parser.error("'input' option is required to run this program")
    if not os.path.exists(options.video_path):
            parser.error("Could not find the input file %s"
                         % options.video_path)

    return options


def main():
    options = parse_arguments()

    track(options.video_path, usegpu)

    print("End of game, have a nice day!")
    return


if __name__ == "__main__":

    main()

# end of file
