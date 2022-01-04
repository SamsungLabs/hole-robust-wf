import os
import cv2
import glob
import time
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.utils import make_directory, read_csv
from utils.real_object_silhouettes import load_predictor, real_object_silhouettes


NUM_MASKSIZE = 512  # Size of the mask map

DIR_INFO = "trainset_info"   # Pre-given information for test set


# Command line arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--dir_places365", help="Path directory of Places365 test set",
                    default="/path/to/data/test_large", type=str)
parser.add_argument("--dir_intermediate", help="Path directory of intermediate silhouttes",
                    default="../algorithm/data/*/silhouettes/*", type=str)
parser.add_argument('--n_jobs', help='How many processes to use',
                    default=1, type=int)
args = parser.parse_args()


def get_testset(list_info, dataset_name):
    silhouettes_set = {l[1] for l in list_info}
    
    pbar = tqdm(total=len(silhouettes_set))

    predictor = load_predictor()

    for sihouette_name in silhouettes_set:
        pbar.update(1)

        # Get the given index and image path
        idx = int(os.path.splitext(sihouette_name.split("_mask")[-1])[0])
        sihouette_path = os.path.join(args.dir_places365, sihouette_name.split("_mask")[0] + ".jpg")

        # Obtain the chosen silhouette map
        silhouette_map = real_object_silhouettes(predictor, sihouette_path)[idx]

        # Save mask
        dir_intermediate = args.dir_intermediate.replace("*", "%s") % (os.path.dirname(dataset_name), os.path.basename(dataset_name))
        make_directory(dir_intermediate)
        cv2.imwrite(os.path.join(dir_intermediate, sihouette_name), silhouette_map)


def main():
    curr_time = time.time()
    
    for interval_csv in glob.glob(os.path.join(DIR_INFO, "*.csv")):
        list_info = read_csv(interval_csv, trainset_mask=True)

        # For trainset mask we want to have "wireframe/02%-03%" instead of the "wireframe/0203"
        path_split = os.path.basename(os.path.splitext(interval_csv)[0]).split("-") # ['wireframe', '0203']
        path_split[1] = f"{path_split[1][:2]}%-{path_split[1][2:]}%"
        dataset_name = "/".join(path_split)

        chunk_size = len(list_info) // args.n_jobs + (1 if len(list_info) % args.n_jobs > 0 else 0)
        Parallel(n_jobs=args.n_jobs)(
            delayed(get_testset)(list_info[start:start+chunk_size], dataset_name)
            for start in range(0, len(list_info), chunk_size)
        )

    print("Processing time: %.2f sec" % (time.time() - curr_time))


if __name__ == "__main__":
    main()
