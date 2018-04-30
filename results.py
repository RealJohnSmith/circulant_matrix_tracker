import matplotlib
matplotlib.use("Agg")  # Headless operation
import pylab
import loader


class results:
    def __init__(self):
        pass

    initialized = False
    frame_number = 0
    omitted_at_frame_number = None

    tracking_figure_axes = None
    tracking_figure = None
    tracking_figure_title = None
    response_figure_axes = None
    tracking_rectangle = None
    template_axes = None
    gt_rectangle = None


def log_tracked(image, tracked_roi, cached, template_f, response_f):
    timeout = 1e-6
    tracking_figure_axes = results.tracking_figure_axes
    tracking_figure = results.tracking_figure
    tracking_figure_title = results.tracking_figure_title
    response_figure_axes = results.response_figure_axes
    tracking_rectangle = results.tracking_rectangle
    gt_rectangle = results.gt_rectangle
    template_axes = results.template_axes

    if not results.initialized:
        # pylab.ion()  # interactive mode on
        tracking_figure = results.tracking_figure = pylab.figure()

        gs = pylab.GridSpec(1, 3, width_ratios=[3, 1, 1])

        tracking_figure_axes = results.tracking_figure_axes = tracking_figure.add_subplot(gs[0])
        tracking_figure_axes.set_title("Tracked object (and ground truth)")

        template_axes = results.template_axes = tracking_figure.add_subplot(gs[1])
        template_axes.set_title("Template")

        response_figure_axes = results.response_figure_axes = tracking_figure.add_subplot(gs[2])
        response_figure_axes.set_title("Response")

        tracking_rectangle = results.tracking_rectangle = pylab.Rectangle((0, 0), 0, 0)
        tracking_rectangle.set_color((1, 1, 0, 0.5))
        tracking_figure_axes.add_patch(tracking_rectangle)

        gt_rectangle = results.gt_rectangle = pylab.Rectangle((0, 0), 0, 0)
        gt_rectangle.set_color((0, 0, 1, 0.5))
        tracking_figure_axes.add_patch(gt_rectangle)

        tracking_figure_title = results.tracking_figure_title = tracking_figure.suptitle("")

        pylab.show(block=False)

    elif tracking_figure is None:
        return  # we simply go faster by skipping the drawing
    elif not pylab.fignum_exists(tracking_figure.number):
        print("From now on drawing will be omitted, "
              "so that computation goes faster")
        results.omitted_at_frame_number = results.frame_number
        results.tracking_figure = None
        return

    tracking_figure_axes.imshow(image)

    tracking_rectangle.set_bounds(tracked_roi[0] - tracked_roi[2] / 2, tracked_roi[1] - tracked_roi[3] / 2, tracked_roi[2], tracked_roi[3])

    gt = loader.get_gt_bounding_box()
    gt_rectangle.set_bounds(gt[0], gt[1], gt[2], gt[3])

    if template_f is not None:
        template_axes.imshow(template_f, cmap=pylab.cm.hot)

    if response_f is not None:
        response_figure_axes.imshow(response_f, cmap=pylab.cm.hot)

    tracking_rectangle.set_color((0 if not cached else 1, 0.5 if not cached else 0, 0, 0.7 if not cached else 0.2))

    tracking_figure_title.set_text("Frame {}".format(results.frame_number))

    pylab.draw()

    if results.initialized:
        pylab.savefig(loader.get_log_dir() + 'image%05i.jpg' % results.frame_number, bbox_inches='tight')

    pylab.waitforbuttonpress(timeout=timeout)

    results.initialized = True
    results.frame_number += 1

    return


def show_precision():
    pass  # TODO

    # """
    # Calculates precision for a series of distance thresholds (percentage of
    # frames where the distance to the ground truth is within the threshold).
    # The results are shown in a new figure.
    #
    # Accepts positions and ground truth as Nx2 matrices (for N frames), and
    # a title string.
    # """
    #
    # print("Evaluating tracking results.")
    #
    # pylab.ioff()  # interactive mode off
    #
    # max_threshold = 50  # used for graphs in the paper
    #
    # # calculate distances to ground truth over all frames
    # delta = positions - ground_truth
    # distances = pylab.sqrt((delta[:, 0]**2) + (delta[:, 1]**2))
    #
    # #distances[pylab.isnan(distances)] = []
    #
    # # compute precisions
    # precisions = pylab.zeros((max_threshold, 1), dtype=float)
    # for p in range(max_threshold):
    #     precisions[p] = pylab.sum(distances <= p, dtype=float) / len(distances)
    #
    # if False:
    #     pylab.figure()
    #     pylab.plot(distances)
    #     pylab.title("Distances")
    #     pylab.xlabel("Frame number")
    #     pylab.ylabel("Distance")
    #
    # # plot the precisions
    # pylab.figure()  # 'Number', 'off', 'Name',
    # pylab.title("Precisions - " + title)
    # pylab.plot(precisions, "k-", linewidth=2)
    # pylab.xlabel("Threshold")
    # pylab.ylabel("Precision")
    #
    # pylab.show()
    # return


def log_meta(key, value):
    with open(loader.get_log_dir() + "meta.txt", "a") as metadata_file:
        metadata_file.write("{}: {}\n".format(key, value))