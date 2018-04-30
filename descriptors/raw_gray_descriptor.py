import pylab


def rgb2gray(rgb_image):
    "Based on http://stackoverflow.com/questions/12201577"
    # [0.299, 0.587, 0.144] normalized gives [0.29, 0.57, 0.14]
    return pylab.dot(rgb_image[:, :, :3], [0.29, 0.57, 0.14])



def initialize(usegpu):
    return 1

def describe(image):
    return rgb2gray(image).reshape([1, image.shape[0], image.shape[1]])

def get_name():
    return "RawGray"