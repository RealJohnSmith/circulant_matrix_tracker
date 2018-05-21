


def initialize(usegpu):
    """
    This method will be called at the start. Descriptor can do some initializing stuff
    usegpu = True means that gpu computation is allowed and should be attempted, if possible
    :return: number of channels that this descriptor outputs
    """

    raise NotImplementedError("Method not implemented")

def describe(image):
    """
    Do the descriptor thing. Image is 3D matrix - first 2 dimensions correspond to pixels of image patch
    last one represents multiple color channel per pixel. R, G, B. In this order, scaled to 0-1 range
    :return: 3D matrix. First dimension should correspond to channels, the next two to 2D image.
                Can be different size then input image
    """

    raise NotImplementedError("Method not implemented")

def get_name():
    """
    :return: String. Name of this descriptor
    """

    raise NotImplementedError("Method not implemented")


def update_roi(old_roi, moved_by):
    """

    :param old_roi: roi from previous frame. Array [x, y, width, height, widht_with_padding, height_with_padding] (TODO: Not sure, something like that)
    :param moved_by: [y, x] how the argmax detection moved. value of [0, 0] is in the middle (no movement since last frame)
    :return: new roi based on old one and moved_by. New roi should reflect how the tracked object moved
    """


    raise NotImplementedError("Method not implemented")