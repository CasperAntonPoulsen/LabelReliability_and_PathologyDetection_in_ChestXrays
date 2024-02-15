import pandas as pd
import numpy as np
from tqdm import tqdm
import os
os.environ["KERAS_BACKEND"] = "jax" 


import keras
from VisionTransformer import Patches, PatchEncoder

# Function to change the image paths
def change_paths(df, data_directory):
    df = df.copy()


    df["ImagePath"] = df["ImagePath"].apply(lambda x: x.replace("padchest-preprocessed", data_directory))

    return df

def crop_augment(
        im, 
        model,
        model_input_size =224,
        original_image_size=512, 
        padding=10
    ):


    image = im.resize((model_input_size,model_input_size))
    #_im = keras.utils.img_to_array(m)

    preds = model.predict(np.expand_dims(keras.utils.img_to_array(image), axis=0))[0]

    top_left_x, top_left_y = max(int(preds[0] * original_image_size) - padding , 0) , max(int(preds[1] * original_image_size) - padding , 0)
 
    bottom_right_x, bottom_right_y = min(int(preds[2] * original_image_size), original_image_size) + padding , min(int(preds[3] * original_image_size) + padding ,original_image_size) 

    img_crop = im.crop([top_left_x, top_left_y, bottom_right_x, bottom_right_y]).resize((original_image_size,original_image_size))



    return img_crop


model = keras.saving.load_model("/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/vit_object_detector_50000_samples.keras")


#data_dir = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/Data/"

# pathology_detection_train = pd.read_csv(data_dir + 'Data_splits/pathology_detection-train.csv', index_col=0)
# tube_detection_train = pd.read_csv(data_dir + "Data_splits/tube_detection-finetuning.csv", index_col=0)

# train_df = pd.concat([pathology_detection_train, tube_detection_train]).reset_index(drop=True)

# pathology_detection_val = pd.read_csv(data_dir + 'Data_splits/pathology_detection-val.csv', index_col=0)
# tube_detection_val = pd.read_csv(data_dir + "Data_splits/tube_detection-finetuning_val.csv", index_col=0)

# val_df = pd.concat([pathology_detection_val, tube_detection_val]).reset_index(drop=True)


#df = pd.concat((train_df, val_df)).reset_index(drop=True)


data_dir = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/"

pathology_detection_test = pd.read_csv(data_dir + 'Data/Data_splits/pathology_detection-test.csv', index_col=0)
pathology_detection_test["ImagePath"] = pathology_detection_test["ImagePath"].apply(lambda x: x.replace("/home/data_shares/purrlab_students/", data_dir + "Data/"))


tube_detection_test = pd.read_csv("/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/" + "Annotation/Annotations_aggregated.csv", index_col=0)
tube_detection_test = tube_detection_test.rename({"Chest_drain":"Chest_drain_tube"},axis=1)
tube_detection_test["ImagePath"] = tube_detection_test["ImagePath"].apply(lambda x: x.replace("../../", data_dir))
tube_detection_test = tube_detection_test.replace({-1:0})



test_df = pd.concat([pathology_detection_test, tube_detection_test])

df_org = change_paths(test_df, "padchest-preprocessed")
df_crop = change_paths(test_df, "padchest-cropped")



source_paths = list(df_org["ImagePath"])
output_paths = list(df_crop["ImagePath"])


for idx in tqdm(range(len(source_paths))):
    #print(source_paths[idx])
    cropped_image = crop_augment( keras.utils.load_img(source_paths[idx]),model )
    #print(output_paths[idx])  
    with open(output_paths[idx], "+wb") as file:

    #     print(output_paths[idx])  
        #
        keras.utils.save_img(file, cropped_image)
