import cv2
import numpy as np
from random import randint, random, seed, shuffle


NUM_ATTEMPT = 500       # Number of max attempts for each Hole Type
NUM_TYPE1_RATIO = 0.8   # A ratio to skip trying to create Hole Type 1


def get_heatmap(npz, size):
    junc_heatmap = np.zeros(size, dtype=np.float32)
    line_heatmap = np.zeros(size, dtype=np.float32)

    for x in range(npz["lpos"].shape[0]):       
        x1, y1 = (npz["lpos"][x,0,0], npz["lpos"][x,0,1])
        x2, y2 = (npz["lpos"][x,1,0], npz["lpos"][x,1,1])
        junc_heatmap[int(size[0]/128*x1),int(size[1]/128*y1)] = 1.0
        junc_heatmap[int(size[0]/128*x2),int(size[1]/128*y2)] = 1.0
        line_heatmap = cv2.line(line_heatmap, (int(size[0]/128*y1),int(size[1]/128*x1)), (int(size[0]/128*y2),int(size[1]/128*x2)), (1.0,1.0,1.0), 1, lineType=cv2.LINE_8)

    return junc_heatmap, line_heatmap


def hole_type_1(silhouette, npz, junc_heatmap, line_heatmap):
    '''
    1. Hole Type 1
     * condition_a: The hole overlaps with line segment(s).
     * condition_b: The hole does not contain any junction.
    '''
    num_attempt = NUM_ATTEMPT

    while num_attempt:
        num_attempt -= 1

        row = randint(0, junc_heatmap.shape[0] - silhouette.shape[0])
        col = randint(0, junc_heatmap.shape[1] - silhouette.shape[1])

        mask_temp = np.zeros(junc_heatmap.shape, dtype=np.float32)
        mask_temp[row:row+silhouette.shape[0], col:col+silhouette.shape[1], ...] += silhouette

        condition_a = np.sum(line_heatmap * mask_temp) > 0
        condition_b = np.sum(junc_heatmap * mask_temp) == 0

        if condition_a and condition_b:
            return mask_temp

    return None


def hole_type_2(silhouette, npz, junc_heatmap, line_heatmap):
    '''
    2. Hole Type 2
     * condition_a: One or more junction(s) are contained in the hole.
     * condition_b: The number of line segments whose endpoints are both contained in the hole, is <= 1.
    '''
    num_attempt = NUM_ATTEMPT

    while num_attempt:
        num_attempt -= 1

        row = randint(0, junc_heatmap.shape[0] - silhouette.shape[0])
        col = randint(0, junc_heatmap.shape[1] - silhouette.shape[1])

        mask_temp = np.zeros(junc_heatmap.shape, dtype=np.float32)
        mask_temp[row:row+silhouette.shape[0], col:col+silhouette.shape[1], ...] += silhouette

        count = 0
        for x in range(npz["lpos"].shape[0]):       
            x1, y1 = (npz["lpos"][x,0,0], npz["lpos"][x,0,1])
            x2, y2 = (npz["lpos"][x,1,0], npz["lpos"][x,1,1])
            if mask_temp[int(mask_temp.shape[0]/128*x1),int(mask_temp.shape[1]/128*y1)] and mask_temp[int(mask_temp.shape[0]/128*x2),int(mask_temp.shape[1]/128*y2)]:
                count += 1

        condition_a = np.sum(junc_heatmap * mask_temp) > 0
        condition_b = count <= 1

        if condition_a and condition_b:
            return mask_temp

    return None


def hole_type_3(silhouette, npz, junc_heatmap, line_heatmap):
    '''
    3. Hole Type 3
     * condition_a: One or more junction(s) are contained in the hole.
     * condition_b: Satisfying the following criteria that can be made (sub-)optimal by repeated searching:
       - condition_b_1: Fewest number of line segments whose endpoints are both contained in the hole.
       - condition_b_2: Largest total length of line segments that (at least partially) overlap with the hole.
    '''
    num_attempt = NUM_ATTEMPT

    condition_b_1_min = 1e10
    condition_b_2_max = -1e10

    range_row = list(range(junc_heatmap.shape[0] - silhouette.shape[0] + 1))
    range_col = list(range(junc_heatmap.shape[1] - silhouette.shape[1] + 1))
    shuffle(range_row)
    shuffle(range_col)
    for row in range_row:
        for col in range_col:
            mask_temp = np.zeros(junc_heatmap.shape, dtype=np.float32)
            mask_temp[row:row+silhouette.shape[0], col:col+silhouette.shape[1], ...] += silhouette
            condition_a = np.sum(junc_heatmap * mask_temp) > 0
            if condition_a:
                mask = mask_temp
                break
        if condition_a:
            break

    while num_attempt:
        num_attempt -= 1

        row = randint(0, junc_heatmap.shape[0] - silhouette.shape[0])
        col = randint(0, junc_heatmap.shape[1] - silhouette.shape[1])

        mask_temp = np.zeros(junc_heatmap.shape, dtype=np.float32)
        mask_temp[row:row+silhouette.shape[0], col:col+silhouette.shape[1], ...] += silhouette

        count = 0
        for x in range(npz["lpos"].shape[0]):       
            x1, y1 = (npz["lpos"][x,0,0], npz["lpos"][x,0,1])
            x2, y2 = (npz["lpos"][x,1,0], npz["lpos"][x,1,1])
            if mask_temp[int(mask_temp.shape[0]/128*x1),int(mask_temp.shape[1]/128*y1)] and mask_temp[int(mask_temp.shape[0]/128*x2),int(mask_temp.shape[1]/128*y2)]:
                count += 1

        overlap = np.sum(line_heatmap * mask_temp)

        condition_a = np.sum(junc_heatmap * mask_temp) > 0
        condition_b_1 = count <= condition_b_1_min
        condition_b_2 = overlap >= condition_b_2_max

        if condition_a and condition_b_1 and condition_b_2:
            mask = mask_temp
            condition_b_1_min = count
            condition_b_2_max = overlap

    return mask


def avoid_isolation(silhouette, npz, size=[512,512]):
    # Define heatmap
    junc_heatmap, line_heatmap = get_heatmap(npz, size)

    # Avoid-isolation algorithm
    if random() > NUM_TYPE1_RATIO:
        mask = hole_type_1(silhouette, npz, junc_heatmap, line_heatmap)
        if mask is not None:
            return mask

    mask = hole_type_2(silhouette, npz, junc_heatmap, line_heatmap)

    if mask is None:
        mask = hole_type_3(silhouette, npz, junc_heatmap, line_heatmap)

    return mask
