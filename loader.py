import os
import os.path
import glob
import pylab
import datetime
import scipy.misc


class loader:
    def __init__(self):
        pass

    initialized = False
    log_dir = None
    img_paths = None
    gt_bounding_boxes = None
    rescale = 1
    frame_number = -1
    normalize_image = True


def next_frame():
    if not loader.initialized:
        raise Exception("No dataset was loaded")

    loader.frame_number += 1

    im = pylab.imread(loader.img_paths[loader.frame_number])
    if loader.normalize_image:
        im = im / 255.0
    if loader.rescale != 1:
        im = scipy.misc.imresize(im, loader.rescale)

    return im


def has_next_frame():
    if not loader.initialized:
        return False
    return len(loader.img_paths) > loader.frame_number


def get_gt_bounding_box():
    if not loader.initialized:
        raise Exception("No dataset was loaded")
    return loader.gt_bounding_boxes[loader.frame_number]


def get_log_dir():
    if not loader.initialized:
        raise Exception("No dataset was loaded")

    return loader.log_dir


def track_bounding_box_from_first_frame():
    if not loader.initialized:
        raise Exception("No dataset was loaded")
    return loader.gt_bounding_boxes[0]


def load_bbabenko(video_path, output_path, rescale=None):
    if loader.initialized:
        raise Exception("Data set already loaded: {}".format(loader.initialized))

    text_files = glob.glob(os.path.join(video_path, "*_gt.txt"))
    if len(text_files) == 0:
        return False

    first_file_path = text_files[0]
    loader.gt_bounding_boxes = pylab.loadtxt(first_file_path, delimiter=",")

    first_ground_truth = loader.gt_bounding_boxes[0, :]

    for i in range(4):  # x, y, width, height
        xp = range(0, loader.gt_bounding_boxes.shape[0], 5)
        fp = loader.gt_bounding_boxes[xp, i]
        x = range(loader.gt_bounding_boxes.shape[0])
        loader.gt_bounding_boxes[:, i] = pylab.interp(x, xp, fp)

    text_files = glob.glob(os.path.join(video_path, "*_frames.txt"))[0]
    if text_files:
        frames = pylab.loadtxt(text_files, delimiter=",", dtype=int)

        test1_path_to_img = os.path.join(video_path,
                                         "imgs/img%05i.png" % frames[0])
        test2_path_to_img = os.path.join(video_path,
                                         "img%05i.png" % frames[0])

        if os.path.exists(test1_path_to_img):
            loader.img_paths = [video_path + "/imgs/img%05i.png" % i for i in range(frames[0], frames[1] + 1)]
        elif os.path.exists(test2_path_to_img):
            loader.img_paths = [video_path + "/imgs/img%05i.png" % i for i in range(frames[0], frames[1] + 1)]
        else:
            return False

    else:
        loader.img_paths = glob.glob(os.path.join(video_path, "*.png"))
        if len(loader.img_paths) == 0:
            loader.img_paths = glob.glob(os.path.join(video_path, "*.jpg"))

        if len(loader.img_paths) == 0:
            return False

        loader.img_paths.sort()

    if rescale is None:
        if pylab.sqrt(first_ground_truth[3] * first_ground_truth[2]) >= 100:
            loader.rescale = 0.5
        else:
            loader.rescale = 1
    else:
        loader.rescale = rescale

    loader.gt_bounding_boxes *= loader.rescale

    set_log_dir(video_path, output_path)

    loader.initialized = 'BBabenko'
    loader.normalize_image = False
    return True


def load_vot(video_path, output_path, rescale=None):
    if loader.initialized:
        raise Exception("Data set already loaded: {}".format(loader.initialized))

    gt_file_path = os.path.join(video_path, "groundtruth.txt")

    if not os.path.realpath(gt_file_path):
        return False

    with open(gt_file_path) as f:
        loader.gt_bounding_boxes = [map(float, x.strip().split(",")) for x in f.readlines()]
        loader.gt_bounding_boxes = map(
            lambda coords: [coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1]],
            map(lambda coords:
                [min(coords[0], coords[2], coords[4], coords[6]),
                 min(coords[1], coords[3], coords[5], coords[7]),
                 max(coords[0], coords[2], coords[4], coords[6]),
                 max(coords[1], coords[3], coords[5], coords[7])], loader.gt_bounding_boxes))

        first_ground_truth = loader.gt_bounding_boxes[0]

    loader.img_paths = glob.glob(os.path.join(video_path, "*.png"))
    if len(loader.img_paths) == 0:
        loader.img_paths = glob.glob(os.path.join(video_path, "*.jpg"))

    if len(loader.img_paths) == 0:
        return False

    loader.img_paths.sort()

    if rescale is None:
        if pylab.sqrt(first_ground_truth[3] * first_ground_truth[2]) >= 100 and False:
            loader.gt_bounding_boxes = map(lambda coords: [coords[0] / 2, coords[1] / 2], loader.gt_bounding_boxes)
            loader.rescale = 0.5
        else:
            loader.rescale = 1
    else:
        loader.rescale = rescale

    loader.gt_bounding_boxes *= loader.rescale

    set_log_dir(video_path, output_path)

    loader.initialized = 'VOT'
    loader.normalize_image = True
    return True


def set_log_dir(video_path, output_path):
    if output_path is not None:
        loader.log_dir = output_path
    else:
        if video_path.find("data/sets"):
            loader.log_dir = os.path.join(video_path.replace("data/sets", "data/logs"), datetime.datetime.now().strftime('%G-%b-%d-%H:%M'))
        else:
            loader.log_dir = os.path.join(video_path, datetime.datetime.now().strftime('%G-%b-%d-%H:%M'))
    if not os.path.exists(loader.log_dir):
        os.makedirs(loader.log_dir)

    if loader.log_dir.find("data/logs"):
        if os.path.exists(loader.log_dir[:loader.log_dir.find("data/logs")+10] + "last"):
            os.remove(loader.log_dir[:loader.log_dir.find("data/logs")+10] + "last")

        os.symlink(loader.log_dir[loader.log_dir.find("data/logs")+10:], loader.log_dir[:loader.log_dir.find("data/logs")+10] + "last")

    if loader.log_dir.endswith("\\"):
        loader.log_dir = loader.log_dir[:-1] + "/"
    elif not loader.log_dir.endswith("/"):
        loader.log_dir = loader.log_dir + "/"
