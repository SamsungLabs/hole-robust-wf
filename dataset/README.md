# Hole-robust Wireframe Detection: Dataset

Dataset generation scripts for the WACV 2022 paper,

"Hole-robust Wireframe Detection" by Naejin Kong, Kiwoong Park and Harshith Goka


## Tested Environment
 * Ubuntu 18.04
 * Python 3.7
 * Virtualenv
 * Nvidia GPU + Cuda 10.1


## Preparation
1. Download pre-processed Wireframe dataset.
 * Download [wireframe.tar.xz](https://drive.google.com/drive/folders/1rXLAh5VIj8jwf8vLfuZncStihRO2chFr) and uncompress it to `../algorithm`.
2. Download pre-processed York Urban dataset.
 * Download [york.tar.xz](https://drive.google.com/drive/folders/1rXLAh5VIj8jwf8vLfuZncStihRO2chFr) and uncompress it to `../algorithm`.
3. Download Places365 testset.
 * Download [test_large.tar](http://data.csail.mit.edu/places/places365/test_large.tar) and uncompress it to `/path/to/data/test_large`.
4. Download Places365-Challenge2016 dataset.
 * Download [train_large_places365challenge.tar](http://data.csail.mit.edu/places/places365/train_large_places365challenge.tar) and uncompress it to `/path/to/data/data_large`.


## Generate Masks for Testset

### Description
Regenerates masks for Wireframe testset and York Urban testset by using precomputed information in `testset_info/`.

Please refer to `avoid_isolation.py` for our implementation of the "avoid-isolation" algorithm.

### Usage
1. Make sure that Places365 testset is prepared.
2. Run `generate_testset_mask.py`.
```bash
$ python generate_testset_mask.py \
  --dir_places365 /path/to/data/test_large \
  --dir_testset "../algorithm/data/*/valid_mask/*"
```

## Generate Masks for Trainset

### Description
Regenerates masks for Wireframe trainset by using precomputed information in `trainset_info/`.

### Usage
1. Make sure that both Wireframe dataset and Places365 testset are prepared.
2. Run `prepare_silhouette_pool.py`.
```bash
$ python prepare_silhouette_pool.py \
--dir_places365 /path/to/data/test_large \
--dir_intermediate "../algorithm/data/*/silhouettes/*"
```
3. Run `generate_trainset_mask.py`.
```bash
$ python generate_trainset_mask.py \
--dir_intermediate "../algorithm/data/*/silhouettes/*" \
--dir_trainset ../algorithm/data/wireframe/train_mask
```

## Generate Pseudo-labeled Dataset

### Description
Generates Pseudo-labeled Dataset by using 157 categories predefined in `pseudo_labeled_dataset_info/categories_places157.txt`.

We allow the user to apply any preferred wireframe detection method to achieve initial pseudo labels.

Threshold values in three criteria for final filtering are hard coded in the script.

### Usage
1. Make sure that Places365-Challenge2016 dataset is prepared.
2. Run `generate_pseudo_labeled_dataset.py`.
```bash
$ python generate_pseudo_labeled_dataset.py \
--dir_places365 /path/to/data/data_large \
--dir_predicted /path/to/data/data_large_npz \
--dir_pseudo_labeled_dataset ../algorithm/data/pseudo_labeled/train
```

### NOTE
`generate_pseudo_labeled_dataset.py` will create an interim list of image file paths `images_places157.flist`.
This list will contain the selected 3,304,547 images that are in 157 structural scene categories.

Corresponding to each image in the list, 
we assume an `.npz` file that contains raw prediction output will be available as follows:
* An image path in `images_places157.flist`: `a/airplane_cabin/00000001.jpg` under `/path/to/data/data_large/`
* Raw prediction output for this image: `a/airplane_cabin/00000001.npz` under `/path/to/data/data_large_npz/`

Users may apply any preferred wireframe detection method as long as its output data format is compatible, 
which is shared identically among most recent methods.


## License
Please refer to `LICENSE`.


## Acknowledgement
We acknowledge Roman Suvorov to inspire us how to effectively postprocess raw detected panoptic labels.


## Citing

```
@inproceedings{robustwf2022wacv,
  title={Hole-robust Wireframe Detection},
  author={Kong, Naejin and Park, Kiwoong and Goka, Harshith},
  booktitle={Proceedings of the IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2022}
}
```
