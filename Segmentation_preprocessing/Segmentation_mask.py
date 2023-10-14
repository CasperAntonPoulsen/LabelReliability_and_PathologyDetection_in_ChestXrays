import pandas as pd
import numpy as np

import cv2
from tqdm import tqdm
import json


def rle2mask(mask_rle: str, label=1, shape=(3520,4280)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)# Needed to align to RLE direction


def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def decode_both_lungs(row, label=1):

    right = rle2mask(
        mask_rle=row["Right Lung"],
        label=label,
        shape=(int(row["Height"]),int(row["Width"]))
    )

    left = rle2mask(
        mask_rle=row["Left Lung"],
        label=label,
        shape=(int(row["Height"]),int(row["Width"]))
    )

    return right + left

def fast_dilate(image, dilation_rate=1):

    #kernel_shape = (dilation_rate*2)+1

    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dilate = cv2.dilate(image, kernel, iterations=dilation_rate)

    return dilate

def bounding_box(image, label=1):

    segmentation = np.where(image == label)

    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

    return cv2.rectangle(image, (x_min, y_min), (x_max, y_max), 1,-1)

def bbox_both_lungs(row, label=1):

    right = bounding_box(rle2mask(
        mask_rle=row["Right Lung"],
        label=label,
        shape=(int(row["Height"]),int(row["Width"]))
    ))

    left = bounding_box(rle2mask(
        mask_rle=row["Left Lung"],
        label=label,
        shape=(int(row["Height"]),int(row["Width"]))
    ))

    return right + left


def apply_all_augmentations(row, dilation_factor=300000):
    
    dilation_rate = int(np.round((row["Height"] * row["Width"]) / dilation_factor))

    mask = decode_both_lungs(row)
    bbox = bounding_box(mask)
    bbox_both = bbox_both_lungs(row)

    dilated_masks = [fast_dilate(mask, dilation_rate=dilation_rate*(i+1)) for i in range(4)]

    output_row = {
        "ImageID":row["ImageID"],
        "original_mask":mask2rle(mask),
        "bbox_mask":mask2rle(bbox),
        "bbox_both_mask":mask2rle(bbox_both),
        "dilated_mask_1":mask2rle(dilated_masks[0]),
        "dilated_mask_2":mask2rle(dilated_masks[1]),
        "dilated_mask_3":mask2rle(dilated_masks[2]),
        "dilated_mask_4":mask2rle(dilated_masks[3]),
    }

    return output_row


if __name__ == "__main__":
    files = "/home/data_shares/purrlab/physionet.org/files/chexmask-cxr-segmentation-data/0.2"

    padchest_masks = pd.read_csv(files+ "/OriginalResolution/Padchest.csv")

    for idx, row in tqdm(padchest_masks.iterrows()):
        output_row = apply_all_augmentations(row)

        with open("/home/caap/LabelReliability_and_PathologyDetection_in_ChestXrays/Data/Masks/masks.ndjson", "a") as file:
            file.write(json.dumps(output_row) + "\n")