# Hole-robust Wireframe Detection

Data and code for the WACV 2022 paper,

"Hole-robust Wireframe Detection" by Naejin Kong, Kiwoong Park and Harshith Goka


## Paper with Supplementary Materials
See [arXiv Version](https://arxiv.org/abs/2111.15064).


## Dataset Generation Scripts
See `dataset/`.


## Algorithm Code
See `algorithm/`.


## Tested Environment
 * Ubuntu 18.04
 * Python 3.7
 * Virtualenv
 * Nvidia GPU + Cuda 10.1


## Installation
1. Clone this repository.
2. Set up virtualenv.
```bash
# Create a new virtualenv
$ sudo apt-get install python3.7-dev
$ virtualenv venv --python=python3.7

# Activate virtualenv
$ source venv/bin/activate

# Install packages
(venv) $ pip install --upgrade pip
(venv) $ pip install -r requirements.txt \
-f https://download.pytorch.org/whl/torch_stable.html \
-f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
(venv) $ pip uninstall pycocotools
(venv) $ pip install pycocotools==2.0.2 --no-binary pycocotools
```

## License
Please refer to `LICENSE`.


## Citing

```
@InProceedings{Kong_2022_WACV,
    author    = {Kong, Naejin and Park, Kiwoong and Goka, Harshith},
    title     = {Hole-Robust Wireframe Detection},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {1636-1645}
}
```
