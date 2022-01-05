# Hole-robust Wireframe Detection: Algorithm

Algorithm code for the WACV 2022 paper,

"Hole-robust Wireframe Detection" by Naejin Kong, Kiwoong Park and Harshith Goka

(Our administration granted to release algorithm code with respect to an L-CNN framework only.)


## Tested Environment
 * Ubuntu 18.04
 * Python 3.7
 * Virtualenv
 * Nvidia GPU + Cuda 10.1


## Preparation
1. Make sure that all of the datasets are prepared, by following `../dataset/README.md`.
2. Clone [L-CNN](https://github.com/zhou13/lcnn), copy all files except for `README.md` under the cloned L-CNN root directory and paste them into `hole-robust-wf/algorithm/`.
```bash
$ git clone https://github.com/zhou13/lcnn.git
$ cd /path/to/cloned/lcnn/
$ rm README.md
$ cp -r . /path/to/hole-robust-wf/algorithm/
```


## How to train

### Description
Train the hole-robust wireframe detection model applied to the L-CNN framework.

### Usage
1. Run `train_hole_robust.py`.
```bash
$ python train_hole_robust.py -d 0 -i hole_robust_wf config/wireframe.yaml
```


## How to evaluate

### Description
Run the trained model on Wireframe(or York Urban) testset with or without hole.

### Usage
1. Run `process_hole_robust.py`.
```bash
$ python process_hole_robust.py config/wireframe.yaml \
<PATH-TO-CKPT-PATH> \
<PATH-TO-TESTSET> \
<PATH-TO-MASK-DIR> \
<PATH-TO-NPZ-DIR>
```
 * `<PATH-TO-CKPT-PATH>`: file path to the trained model  
   ex) `logs/000000-000000-e0748b-hole_robust_wf/checkpoint_best.pth`
 * `<PATH-TO-TESTSET>`: root directory of testset  
   ex) `data/wireframe` (Wireframe testset), `data/york` (York Urban testset)
 * `<PATH-TO-MASK-DIR>`: directory of masks for testset  
   ex) `None`(w/o mask), `data/wireframe/valid_mask/0010`(00%-10% testset), `data/wireframe/valid_mask/1030`(10%-30% testset)  
       `data/york/valid_mask/0010`(00%-10% testset), `data/york/valid_mask/1030`(10%-30% testset)
 * `<PATH-TO-NPZ-DIR>`: directory of detected results to be saved  
   ex) `logs/npz-hole_robust_wf-eval-wireframe-1030`

2. To evaluate *sAP* and *mAPJ*:
```bash
$ python eval-sAP.py <PATH-TO-NPZ-DIR>
$ python eval-mAPJ.py <PATH-TO-NPZ-DIR>
```


## How to infer

### Description
Detect wireframes on user's own images.

### Usage
1. Run `demo_hole_robust.py`.
```bash
$ python demo_hole_robust.py -d 0 config/wireframe.yaml \
<PATH-TO-CKPT-PTH> \
<PATH-TO-IMAGE-DIR> \
<PATH-TO-MASK-DIR>
```
 * `<PATH-TO-CKPT-PTH>`: file path to the trained model
 * `<PATH-TO-IMAGE-DIR>`: directory of input images
 * `<PATH-TO-MASK-DIR>`: directory of input masks corresponding to the input images


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
