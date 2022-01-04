import os
import cv2
import glob
import time
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from math import sqrt

from utils.utils import make_directory, read_txt, write_txt, count_file


NUM_PLACES157 = 3304547     # Number of images in Places157 Challenge training set
NUM_CUTOFF_SCORE = 0.9470   # Maximum allowed number of intervals for saving as silhouette maps

LIST_CRITERIA = [74.98, 6456.57, 1.34]  # Threshold values determined by inspecting the distributions of Wireframe training dataset

DIR_INFO = "pseudo_labeled_dataset_info"            # Pre-given information for pseudo labeled dataset
DIR_PLACES157_CATEGORY = "categories_places157.txt" # List of selected 157 scene categories from Places365 categories
DIR_PLACES157_IMAGE = "images_places157.flist"      # List of images in Places157 Challenge training set


# Command line arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--dir_places365", help="Path directory of Places365 Challenge training set",
                    default="/path/to/data/data_large", type=str)
parser.add_argument("--dir_predicted", help="Path directory of predicted wireframes from Places157 Challenge training set",
                    default="/path/to/data/data_large_pred", type=str)
parser.add_argument("--dir_pseudo_labeled_dataset", help="Path directory of generated pseudo labeled dataset",
                    default="../algorithm/data/pseudo_labeled/train", type=str)
parser.add_argument('--n_jobs', help='How many processes to use',
                    default=1, type=int)
args = parser.parse_args()


def is_satisfy_criteria(lines_pseudo, image_shape):
    """
    condition_a: Number of lines in an image > 74.98
    condition_b: Total length of all lines in an image > 6456.57
    condition_c: The ratio '# of junctions / # of lines' in an image < 1.34
    """
    num_junc = 0    # number of junctions
    num_line = 0    # number of lines
    len_line = 0.0  # total length of lines in 512x512
    map_junc = np.zeros(image_shape[:2], dtype=np.bool)

    for line_pseudo in lines_pseudo:
        x1, y1 = line_pseudo[0]
        x2, y2 = line_pseudo[1]

        x1_512, y1_512 = x1/image_shape[0]*512, y1/image_shape[1]*512
        x2_512, y2_512 = x2/image_shape[0]*512, y2/image_shape[1]*512
        len_line += sqrt((x2_512-x1_512)*(x2_512-x1_512) + (y2_512-y1_512)*(y2_512-y1_512))

        map_junc[int(x1), int(y1)] = True
        map_junc[int(x2), int(y2)] = True

    num_junc = np.sum(map_junc)
    num_line = lines_pseudo.shape[0]

    if num_line == 0:
        return 0

    condition_a = num_line > LIST_CRITERIA[0]
    condition_b = len_line > LIST_CRITERIA[1]
    condition_c = (num_junc / num_line) < LIST_CRITERIA[2]

    return 1 if condition_a and condition_b and condition_c else 0


def get_pseudo_labeled_dataset(paths_image, paths_label):
    pbar = tqdm(total=len(paths_label))

    for path_image, path_label in zip(paths_image, paths_label):
        pbar.update(1)

        image = cv2.imread(path_image)
        with np.load(path_label) as npz:
            lines = npz["lines"]
            score = npz["score"]

            # Apply line cutoff score to retrieve pseudo positive lines
            lines_pseudo = np.array([ l for l, s in zip(lines, score) if s >= NUM_CUTOFF_SCORE])

            # Check whether pseudo positive lines satisfiy three criteria determined from the distributions of Wireframe training dataset
            if is_satisfy_criteria(lines_pseudo, image.shape):
                path_image_pseudo = path_image.replace(args.dir_places365, args.dir_pseudo_labeled_dataset)
                path_label_pseudo = os.path.splitext(path_image_pseudo)[0] + "_label.npz"
                make_directory(os.path.dirname(path_image_pseudo))
                shutil.copy(path_image, path_image_pseudo)
                np.savez(path_label_pseudo, **{"lines_pseudo": lines_pseudo})

def main():
    curr_time = time.time()

    # 1. Prepare a list of image file paths in Places365 Challenge training set
    print("1. Prepare a list of image file paths in Places365 Challenge training set")
    if not os.path.isfile(os.path.join(DIR_INFO, DIR_PLACES157_IMAGE)):
        categories = [x.split(" ")[0][1:] for x in read_txt(os.path.join(DIR_INFO, DIR_PLACES157_CATEGORY))]
        for category in categories:
            paths_image = sorted(glob.glob(os.path.join(args.dir_places365, category, "*.jpg")))
            write_txt(os.path.join(DIR_INFO, DIR_PLACES157_IMAGE), [x[len(args.dir_places365)+1:] for x in paths_image])      

    # 2. Predict wireframes from Places157 Challenge training set by using any existing method
    print("2. Predict wireframes from Places157 Challenge training set by using any existing method")
    ####################
    # [ blank ]
    # ...
    # ...
    ####################

    # 3. Check whether predicted wireframes from Places157 Challenge training set are prepared
    print("3. Check whether predicted wireframes from Places157 Challenge training set are prepared")
    num_count = count_file(args.dir_predicted, ext=[".npz"])
    if not num_count == NUM_PLACES157:
        print(f"> Predicted wireframes are not prepared: {num_count:,} / {NUM_PLACES157:,}")
        return

    # 4. Apply line cutoff score as well as three criteria determined from the distributions of Wireframe training dataset
    print("4. Apply line cutoff score as well as three criteria determined from the distributions of Wireframe training dataset")
    paths_image = sorted([os.path.join(args.dir_places365, x) for x in read_txt(os.path.join(DIR_INFO, DIR_PLACES157_IMAGE))])
    paths_label = sorted(glob.glob(os.path.join(args.dir_predicted, "**", "*.npz"), recursive=True))
    assert len(paths_image) == len(paths_label)
    chunk_size = len(paths_label) // args.n_jobs + (1 if len(paths_label) % args.n_jobs > 0 else 0)
    Parallel(n_jobs=args.n_jobs)(
        delayed(get_pseudo_labeled_dataset)(paths_image[start:start+chunk_size], paths_label[start:start+chunk_size])
        for start in range(0, len(paths_label), chunk_size)
    )
    num_pseudo_labeled_dataset = len(glob.glob(os.path.join(args.dir_pseudo_labeled_dataset, "**", "*.npz"), recursive=True))
    print(f"Finished: total {num_pseudo_labeled_dataset:,} <image, wireframe> pairs were generated for pseudo labeled dataset (from {NUM_PLACES157:,} examples)")

    print("Processing time: %.2f sec" % (time.time() - curr_time))


if __name__ == "__main__":
    main()
