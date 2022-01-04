import cv2
import PIL
from skimage import io, color
import numpy as np
from math import log2, ceil

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def upgrade_type(arr):
    """
    BSD-3-Clause License
    https://github.com/william-silversmith/countless/blob/master/python/countless2d.py
    """
    dtype = arr.dtype

    if dtype == np.uint8:
        return arr.astype(np.uint16), True
    elif dtype == np.uint16:
        return arr.astype(np.uint32), True
    elif dtype == np.uint32:
        return arr.astype(np.uint64), True

    return arr, False
  

def downgrade_type(arr):
    """
    BSD-3-Clause License
    https://github.com/william-silversmith/countless/blob/master/python/countless2d.py
    """
    dtype = arr.dtype

    if dtype == np.uint64:
        return arr.astype(np.uint32)
    elif dtype == np.uint32:
        return arr.astype(np.uint16)
    elif dtype == np.uint16:
        return arr.astype(np.uint8)

    return arr


def zero_corrected_countless(data):
    """
    BSD-3-Clause License
    https://github.com/william-silversmith/countless/blob/master/python/countless2d.py
    """
    """
    Vectorized implementation of downsampling a 2D 
    image by 2 on each side using the COUNTLESS algorithm.

    data is a 2D numpy array with even dimensions.
    """
    # allows us to prevent losing 1/2 a bit of information 
    # at the top end by using a bigger type. Without this 255 is handled incorrectly.
    data, upgraded = upgrade_type(data) 

    # offset from zero, raw countless doesn't handle 0 correctly
    # we'll remove the extra 1 at the end.
    data += 1 

    sections = []

    # This loop splits the 2D array apart into four arrays that are
    # all the result of striding by 2 and offset by (0,0), (0,1), (1,0), 
    # and (1,1) representing the A, B, C, and D positions from Figure 1.
    factor = (2,2)
    for offset in np.ndindex(factor):
        part = data[tuple(np.s_[o::f] for o, f in zip(offset, factor))]
        sections.append(part)

    a, b, c, d = sections

    ab = a * (a == b) # PICK(A,B)
    ac = a * (a == c) # PICK(A,C)
    bc = b * (b == c) # PICK(B,C)

    a = ab | ac | bc # Bitwise OR, safe b/c non-matches are zeroed

    result = a + (a == 0) * d - 1 # a or d - 1

    if upgraded:
        return downgrade_type(result)

    # only need to reset data if we weren't upgraded 
    # b/c no copy was made in that case
    data -= 1

    return result


def load_predictor(checkpoint_name="COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"):
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"
    cfg.merge_from_file(model_zoo.get_config_file(checkpoint_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_name)
    predictor = DefaultPredictor(cfg)

    return predictor


def real_object_silhouettes(predictor, path):
    # 1. Read image
    image = io.imread(path)
    if image.ndim == 2:
        image = color.gray2rgb(image)

    # 2. Apply Detectron2
    panoptic_seg, segments_info = predictor(image)["panoptic_seg"]
    panoptic_seg = panoptic_seg.detach().cpu().numpy()

    # 3. Clean detected labels and collect 'things' only (less than 30% size)
    shape_in_power2 = [num if log2(num).is_integer() else 2 ** int(ceil(log2(num))) for num in panoptic_seg.shape]
    panoptic_seg_width64 = PIL.Image.fromarray(panoptic_seg).resize(shape_in_power2[::-1], PIL.Image.NEAREST)
    panoptic_seg_width64 = np.array(panoptic_seg_width64, dtype=np.int32)
    while panoptic_seg_width64.shape[1] > 64:
        panoptic_seg_width64 = zero_corrected_countless(panoptic_seg_width64)

    things = list()
    for sinfo in segments_info:
        if sinfo["isthing"] and np.sum(panoptic_seg_width64 == sinfo["id"]):
            thing = np.array(panoptic_seg == sinfo["id"], dtype=np.uint8)

            percent = np.array(thing, dtype=np.float32).sum().item() / np.prod(panoptic_seg.shape) * 100
            if percent >= 30.0:
                continue

            things.append(thing)

    # 4. Post-process segments
    silhouettes = list()
    for thing in things:
        ret, _, _, _ = cv2.connectedComponentsWithStats(thing)
        if ret > 2:
            # Discard if it has multiple isolated regions
            thing = np.empty(1)
        else:
            # Fill hole
            img_floodfill = thing.copy()
            cv2.floodFill(img_floodfill, np.zeros((thing.shape[0]+2, thing.shape[1]+2), np.uint8), (0,0), 255)

            thing = thing | cv2.bitwise_not(img_floodfill)

            # Crop tight bounding box around silhouette map
            sum_row = np.sum(thing, axis=1).squeeze()
            sum_col = np.sum(thing, axis=0).squeeze()

            crop_top = next((i for i, x in enumerate(sum_row) if x), None)
            crop_bottom = thing.shape[0] - next((i for i, x in enumerate(sum_row[::-1]) if x), None)
            crop_left = next((i for i, x in enumerate(sum_col) if x), None)
            crop_right = thing.shape[1] - next((i for i, x in enumerate(sum_col[::-1]) if x), None)

            thing = thing[crop_top:crop_bottom, crop_left:crop_right]

        silhouettes.append(thing)

    return silhouettes
