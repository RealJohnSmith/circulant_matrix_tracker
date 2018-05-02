# Circulant matrix tracker expanded to other descriptors #


This code combines Circulant Matrix Tracker by João F. Henriques ([Github](https://github.com/rodrigob/circulant_matrix_tracker), [Author's page](http://www.robots.ox.ac.uk/~joao/circulant/))
and Hardnet image descriptor ([Github](https://github.com/DagnyT/hardnet), [Paper](https://arxiv.org/abs/1705.10872))  


## To use this code: ##

1. Get datasets. Comes with loader for [MILTrack](https://bbabenko.github.io/miltrack.html) and [VOT](http://votchallenge.net/vot2016/dataset.html). Alternatively you can write your own loader.
2. Extract somewhere. Recommended structure is to put data into `./data/sets/*` and create folder `./data/logs/` where output will be generated 
3. Run `./circulant_matrix_tracker.py -i path_to_dataset`

For more options run `./circulant_matrix_tracker.py -h`

## Descriptors ##

To change image descriptor to be used for tracking use `-d|--descriptor` option. Currently supported descriptors are:

* `raw` | `gray` - Describe image using raw grayscale pixels
* `hardnet` - Describe image using 128-channel pretrained HardNet++ descriptor 


### What is TODO/WIP so far: ###

* *WIP:* GPU computation. If enabled some things get offloaded to gpu, but this parts needs more optimizations work
* *TODO:* HOG Features descriptor
* *TODO:* Some metric to automatically measure and evaluate quality of tracking
* *TODO:* Rewrite everything to c++


### Dependencies ###

* Python (used 2.7, might work with other versions too)
* Numpy
* Matplotlib
* Scipy
* Torch
* PyLab


