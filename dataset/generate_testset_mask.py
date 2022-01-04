import os
import cv2
import glob
import time
import argparse
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.utils import make_directory, read_csv
from utils.real_object_silhouettes import load_predictor, real_object_silhouettes


NUM_MASKSIZE = 512  # Size of the mask map

DIR_INFO = "testset_info"   # Pre-given information for test set


# Command line arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--dir_places365", help="Path directory of Places365 test set",
                    default="/path/to/data/test_large", type=str)
parser.add_argument("--dir_testset", help="Path directory of generated test set",
                    default="../algorithm/data/*/valid_mask/*", type=str)
parser.add_argument('--n_jobs', help='How many processes to use',
                    default=1, type=int)
args = parser.parse_args()


def get_testset(list_info):
    pbar = tqdm(total=len(list_info))

    predictor = load_predictor()

    for info in list_info:
        pbar.update(1)

        # Get the given index and image path
        idx = int(os.path.splitext(info[1].split("_mask")[-1])[0])
        info[1] = os.path.join(args.dir_places365, info[1].split("_mask")[0] + ".jpg")

        # Obtain the chosen silhouette map
        silhouette_map = real_object_silhouettes(predictor, info[1])[idx]

        # Superimpose silhouette map onto a 512x512 region
        mask = np.zeros((NUM_MASKSIZE, NUM_MASKSIZE), dtype=np.uint8)
        mask[int(info[2]):int(info[2])+silhouette_map.shape[0],int(info[3]):int(info[3])+silhouette_map.shape[1]] = silhouette_map

        # Save mask
        dir_testset = args.dir_testset.replace("*", "%s") % (os.path.dirname(info[-1]), os.path.basename(info[-1]))
        make_directory(dir_testset)
        cv2.imwrite(os.path.join(dir_testset, info[0]), mask)


def main():
    curr_time = time.time()
   
    list_info = read_csv(glob.glob(os.path.join(DIR_INFO, "*.csv")))

    chunk_size = len(list_info) // args.n_jobs + (1 if len(list_info) % args.n_jobs > 0 else 0)
    Parallel(n_jobs=args.n_jobs)(
        delayed(get_testset)(list_info[start:start+chunk_size])
        for start in range(0, len(list_info), chunk_size)
    )

    print("Processing time: %.2f sec" % (time.time() - curr_time))


if __name__ == "__main__":
	main()
