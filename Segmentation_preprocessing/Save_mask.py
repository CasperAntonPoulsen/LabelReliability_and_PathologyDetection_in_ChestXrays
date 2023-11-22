import pandas as pd
import numpy as np
import tensorflow as tf

from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from tqdm import tqdm


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

    _image = image.copy()
    segmentation = np.where(_image == label)

    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

    return cv2.rectangle(_image, (x_min, y_min), (x_max, y_max), 1,-1)

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

def preprocess_img(img, printing=False):
    


    # resize with padding:
    try:
        img = tf.convert_to_tensor(img)
        img = tf.expand_dims(img, -1)
        img = tf.image.resize_with_pad(img, 512, 512)
    except:
        return "Not an Image"
    
    return img

def crop_image(image, mask):


    _mask = mask.copy()
    segmentation = np.where(_mask == 1)

    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))


    return image.crop((x_min, y_min, x_max, y_max))

def apply_all_augmentations_no_encode(row, dilation_factor=300000):
    
    dilation_rate = int(np.round((row["Height"] * row["Width"]) / dilation_factor))

    mask = decode_both_lungs(row)
    bbox = bounding_box(mask)
    bbox_both = bbox_both_lungs(row)

    dilated_masks = [fast_dilate(mask, dilation_rate=dilation_rate*(i+1)) for i in range(4)]

    output_row = {
        "ImageID":row["ImageID"],
        "original_mask":mask,
        "bbox_mask":bbox,
        "bbox_both_mask":bbox_both,
        "dilated_mask_1":dilated_masks[0],
        "dilated_mask_2":dilated_masks[1],
        "dilated_mask_3":dilated_masks[2],
        "dilated_mask_4":dilated_masks[3],
    }

    return output_row

def remove_lungs(image, mask):
    
    _mask = np.where(mask==0, 1, 0)
    
    return image * _mask

def only_lungs(image, mask):    
    return image * mask

if __name__ == "__main__":
    files = "/home/data_shares/purrlab/physionet.org/files/chexmask-cxr-segmentation-data/0.2"

    padchest_masks = pd.read_csv(files+ "/OriginalResolution/Padchest.csv")
    test_set = pd.read_csv("../Data/Data_splits/pathology_detection-test.csv", index_col=0)
    annotations = pd.read_csv('../Annotation/Annotations_aggregated.csv', index_col=0)
    annotations["ImagePath"] = annotations["ImagePath"].apply(lambda x : x.replace("../../Data","/home/data_shares/purrlab_students") )

    test_data = pd.concat([test_set, annotations])
    test_set_masks = pd.merge(padchest_masks, test_set, how="inner", on= "ImageID")

    for idx in tqdm(range(len(test_set_masks))):
        masks = apply_all_augmentations_no_encode(test_set_masks.iloc[idx])
        image = Image.open(test_set_masks.iloc[idx]["ImagePath"])
        image_as_array = np.asarray(image)


        purrlab_path , image_path = test_set_masks.iloc[idx]["ImagePath"].split("padchest-preprocessed")

        mask_names = [
                "original_mask", 
                "bbox_mask",
                "bbox_both_mask",
                "dilated_mask_1",
                "dilated_mask_2",
                "dilated_mask_3",
                "dilated_mask_4"
        ]
        print(test_set_masks.iloc[idx]["ImagePath"])
        for mask in mask_names:

            Path(f"{purrlab_path}Modified_segmentation_masks/{mask}/inside/{image_path.split('/')[1]}/").mkdir(parents=True, exist_ok=True)
            Path(f"{purrlab_path}Modified_segmentation_masks/{mask}/outside/{image_path.split('/')[1]}/").mkdir(parents=True, exist_ok=True)

            mask_preprocessed = np.asarray(preprocess_img(masks[mask]), dtype=np.uint8)[:,:,0]
            with open(f"{purrlab_path}Modified_segmentation_masks/{mask}/inside{image_path}", "+wb") as file:
                

                inside_only = only_lungs(image=image_as_array, mask=mask_preprocessed)    

                inside_only_image = Image.fromarray(inside_only)

                inside_only_image.save(file)

            with open(f"{purrlab_path}Modified_segmentation_masks/{mask}/outside{image_path}", "+wb") as file:
                outside_only = remove_lungs(image=image_as_array, mask=mask_preprocessed)

                outside_only_image = Image.fromarray(outside_only.astype(np.uint8))

                outside_only_image.save(file)
                