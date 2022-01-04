import os
import cv2
import glob
import time
import argparse
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from PIL import Image

from utils.utils import make_directory, read_csv


NUM_MASKSIZE = 512  # Size of the mask map

DIR_INFO = "trainset_info"   # Pre-given information for test set


# Command line arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--dir_intermediate", help="Path directory of intermediate silhouttes",
                    default="../algorithm/data/*/silhouettes/*", type=str)
parser.add_argument("--dir_trainset", help="Path directory of generated train set",
                    default="../algorithm/data/*/train_mask/*", type=str)
parser.add_argument('--n_jobs', help='How many processes to use',
                    default=1, type=int)
args = parser.parse_args()


def get_testset(list_info, pos):
    pbar = tqdm(total=len(list_info), position=pos)

    for info in list_info:
        pbar.update(1)

        dir_intermediate = args.dir_intermediate.replace("*", "%s") % (os.path.dirname(info[-1]), os.path.basename(info[-1]))
        silhouette_map = np.array(Image.open(os.path.join(dir_intermediate, info[1])))

        # Superimpose silhouette map onto a 512x512 region
        mask = np.zeros((NUM_MASKSIZE, NUM_MASKSIZE), dtype=np.uint8)
        mask[int(info[2]):int(info[2])+silhouette_map.shape[0],int(info[3]):int(info[3])+silhouette_map.shape[1]] = silhouette_map

        # Save mask
        dir_trainset = args.dir_trainset.replace("*", "%s") % (os.path.dirname(info[-1]), os.path.basename(info[-1]))
        make_directory(dir_trainset)
        cv2.imwrite(os.path.join(dir_trainset, info[0]), mask)


def main():
    curr_time = time.time()

    for interval_csv in glob.glob(os.path.join(DIR_INFO, "*.csv")):
        list_info = read_csv(interval_csv, trainset_mask=True)

        chunk_size = len(list_info) // args.n_jobs + (1 if len(list_info) % args.n_jobs > 0 else 0)
        Parallel(n_jobs=args.n_jobs)(
            delayed(get_testset)(list_info[start:start+chunk_size], start//chunk_size)
            for start in range(0, len(list_info), chunk_size)
        )

    print("Processing time: %.2f sec" % (time.time() - curr_time))


if __name__ == "__main__":
	main()
