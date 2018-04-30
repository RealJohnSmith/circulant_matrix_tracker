
class Descriptor:

    def __init__(self):
        pass

    def initialize(self, usegpu):
        """
        This method will be called at the start. Descriptor can do some initializing stuff
        :return: number of channels that this descriptor outputs
        """

        raise NotImplementedError("Method not implemented")

    def describe(self, image):
        """
        Do the descriptor thing. Image is 3D matrix - first 2 dimensions correspond to pixels of image patch
        last one represents multiple color channel per pixel. R, G, B. In this order
        :return: 3D matrix. First dimension should correspond to channels, the next two to 2D image.
                    Can be different size then input image
        """

        raise NotImplementedError("Method not implemented")

    def get_name(self):
        """
        :return: String. Name of this descriptor
        """

        raise NotImplementedError("Method not implemented")