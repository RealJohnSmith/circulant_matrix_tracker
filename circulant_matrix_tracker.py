#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path
from optparse import OptionParser
import results
import pylab
import loader
import time

from descriptors import raw_gray_descriptor, hardnet_descriptor


# parameters according to the paper --
class kcf_params:
    padding = 1.0  # extra area surrounding the target
    # spatial bandwidth (proportional to target)
    output_sigma_factor = 1 / float(16)
    sigma = 0.2  # gaussian kernel bandwidth
    lambda_value = 1e-2  # regularization
    # linear interpolation factor for adaptation
    interpolation_factor = 0.075



def get_subwindow(image, box):
    xs = pylab.floor(box[0]) \
         + pylab.arange(box[4], dtype=int) - pylab.floor(box[4] / 2)
    ys = pylab.floor(box[1]) \
         + pylab.arange(box[5], dtype=int) - pylab.floor(box[5] / 2)

    xs = xs.astype(int)
    ys = ys.astype(int)

    xs[xs < 0] = 0
    xs[xs >= image.shape[1]] = image.shape[1] - 1

    ys[ys < 0] = 0
    ys[ys >= image.shape[0]] = image.shape[0] - 1

    return image[pylab.ix_(ys, xs, range(image.shape[2]))]


def apply_cos_window(channels):
    global cos_window

    if cos_window is None:
        cos_window = pylab.outer(pylab.hanning(channels.shape[1]), pylab.hanning(channels.shape[2]))

    return pylab.multiply(channels[:] - 0.5, cos_window)


def dense_gauss_kernel(sigma, x, y=None):
    xf = pylab.fft2(x)  # x in Fourier domain
    x_flat = x.flatten()
    xx = pylab.dot(x_flat.transpose(), x_flat)  # squared norm of x

    if y is not None:
        yf = pylab.fft2(y)
        y_flat = y.flatten()
        yy = pylab.dot(y_flat.transpose(), y_flat)
    else:
        yf = xf
        yy = xx

    xyf = pylab.multiply(xf, pylab.conj(yf))

    xyf_ifft = pylab.ifft2(xyf)
    row_shift, col_shift = pylab.floor(pylab.array(x.shape) / 2).astype(int)
    xy_complex = pylab.roll(xyf_ifft, row_shift, axis=0)
    xy_complex = pylab.roll(xy_complex, col_shift, axis=1)
    xy = pylab.real(xy_complex)

    scaling = -1 / (sigma ** 2)
    xx_yy = xx + yy
    xx_yy_2xy = xx_yy - 2 * xy
    k = pylab.exp(scaling * pylab.maximum(0, xx_yy_2xy / x.size))

    return k


def track(descriptor):

    global options
    desc_channel_count = descriptor.initialize(options.use_gpu)

    roi = loader.track_bounding_box_from_first_frame()
    roi = [roi[0] + roi[2] / 2, roi[1] + roi[3] / 2, roi[2], roi[3], roi[2] * (1 + kcf_params.padding), roi[3] * (1 + kcf_params.padding)]

    output_sigma = pylab.sqrt(pylab.prod([roi[3], roi[2]])) * kcf_params.output_sigma_factor

    avg_count = 0


    global cos_window
    cos_window = None
    template = [None for i in range(desc_channel_count)]
    alpha_f = [None for i in range(desc_channel_count)]
    response = [None for i in range(desc_channel_count)]
    yf = None

    track_time = 0
    full_track_time = time.time()
    while loader.has_next_frame():
        im = loader.next_frame()

        if (loader.frame_number() % 10) == 0:
            print("Processing frame {}".format(loader.frame_number()))

        start_time = time.time()

        is_first_frame = loader.frame_number() == 0

        cropped = get_subwindow(im, roi)
        channels = descriptor.describe(cropped)
        subwindow = apply_cos_window(channels)

        if is_first_frame:
            grid_y = pylab.arange(subwindow.shape[1]) - pylab.floor(subwindow.shape[1] / 2)
            grid_x = pylab.arange(subwindow.shape[2]) - pylab.floor(subwindow.shape[2] / 2)

            rs, cs = pylab.meshgrid(grid_x, grid_y)
            y = pylab.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))
            yf = pylab.fft2(y)
        else:
            avg_count, avg_x, avg_y = 0, 0, 0

            for i in range(0, subwindow.shape[0]):
                channel = subwindow[i, :, :]

                # calculate response of the classifier at all locations
                k = dense_gauss_kernel(kcf_params.sigma, channel, template[i])
                kf = pylab.fft2(k)
                alphaf_kf = pylab.multiply(alpha_f[i], kf)
                response[i] = pylab.real(pylab.ifft2(alphaf_kf))  # Eq. 9

                argmax = response[i].argmax()

                if response[i].item(argmax) != 0:
                    tmp = pylab.unravel_index(argmax, response[i].shape)
                    avg_x += tmp[1]
                    avg_y += tmp[0]
                    avg_count += 1

            if avg_count > 0:
                moved_by = [float(avg_y) / avg_count - float(channel.shape[0]) / 2,
                           float(avg_x) / avg_count - float(channel.shape[1]) / 2]
                roi[0] = round(moved_by[1] * roi[4] / channel.shape[1] + roi[0])
                roi[1] = round(moved_by[0] * roi[5] / channel.shape[0] + roi[1])

        cropped = get_subwindow(im, roi)
        channels = descriptor.describe(cropped)
        subwindow = apply_cos_window(channels)

        for i in range(0, subwindow.shape[0]):

            channel = subwindow[i, :, :]

            k = dense_gauss_kernel(kcf_params.sigma, channel)
            new_alpha_f = pylab.divide(yf, (pylab.fft2(k) + kcf_params.lambda_value))  # Eq. 7
            new_template = channel

            if is_first_frame:
                alpha_f[i] = new_alpha_f
                template[i] = new_template
            else:
                f = kcf_params.interpolation_factor
                alpha_f[i] = (1 - f) * alpha_f[i] + f * new_alpha_f
                template[i] = (1 - f) * template[i] + f * new_template

        track_time += time.time() - start_time

        results.log_tracked(im, roi, avg_count == 0, template[0], response[0])
    # end of "for each image in video"

    results.log_meta("speed.frames_tracked", loader.frame_number())
    results.log_meta("speed.track_no_io_time", str(track_time) + "s")
    results.log_meta("speed.track_no_io_fps", loader.frame_number() / track_time)
    results.log_meta("speed.track_no_init_time", str(time.time() - full_track_time) + "s")

    results.show_precision()

    return


def parse_arguments():
    parser = OptionParser()
    parser.description = "This program will track objects in image sequences"

    parser.add_option("-i", "--input", dest="input_path",
                      metavar="PATH", type="string", default=None,
                      help="path to a folder with dataset")
    parser.add_option("-o", "--output", dest="output_path",
                      metavar="PATH", type="string", default=None,
                      help="path to a folder to which output images should be stored. If none is supplied, default will be created")
    parser.add_option("--note", dest="note",
                      type="string", default=None,
                      help="optional note that will get passed to output data")
    parser.add_option("-g", "--use-gpu", dest="use_gpu",
                      action="store_true",
                      help="try to run on gpu, where applies")
    parser.add_option("-d", "--descriptor", dest="descriptor",
                      action="store", type="string", default="raw",
                      help="Set descriptor to run with")

    (options, args) = parser.parse_args()

    if not options.input_path:
        parser.error("'input' option is required to run this program")
    if not os.path.exists(options.input_path):
        parser.error("Could not find the input data set in %s" % options.video_path)

    return options


def main():
    global options
    run_time = time.time()
    options = parse_arguments()

    loader.load(options.input_path, options.output_path)

    if options.descriptor.lower() == "raw" or\
            options.descriptor.lower() == "gray" or\
            options.descriptor.lower() == "grey":
        descriptor = raw_gray_descriptor
    elif options.descriptor.lower() == "hardnet":
        descriptor = hardnet_descriptor
    else:
        raise Exception("Unknown descriptor '{}'".format(options.descriptor))

    results.log_meta("descriptor", descriptor.get_name())
    results.log_meta("dataset", options.input_path)
    if options.note is not None:
        results.log_meta("note", options.note)

    if options.use_gpu:
        results.log_meta("use_gpu", "true")
    else:
        results.log_meta("use_gpu", "false")

    results.log_meta("tracker.padding", kcf_params.padding)
    results.log_meta("tracker.interpolation_factor", kcf_params.interpolation_factor)
    results.log_meta("tracker.lambda", kcf_params.lambda_value)
    results.log_meta("tracker.sigma", kcf_params.sigma)
    results.log_meta("tracker.output_sigma_factor", kcf_params.output_sigma_factor)

    track(descriptor)

    run_time -= time.time()
    run_time *= -1

    results.log_meta("speed.total_run_time", str(run_time) + "s")

    print("Finished in {}s".format(run_time))
    return


if __name__ == "__main__":
    main()
