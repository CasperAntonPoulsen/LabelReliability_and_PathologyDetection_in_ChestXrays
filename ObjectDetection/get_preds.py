# Imports
import pandas as pd
import numpy as np
#import tensorflow as tf
import os
import sys

os.environ["KERAS_BACKEND"] = "jax" 
import keras
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model
from tqdm import tqdm



# Function to change the image paths
def change_paths(df, data_directory):
    df = df.copy()


    df["ImagePath"] = df["ImagePath"].apply(lambda x: x.replace("padchest-preprocessed", data_directory))

    return df




def main():
    mt_names = [
        'mt_gamma_0.2_epochs_50.keras',
        'mt_gamma_0.4_epochs_50.keras',
        'mt_gamma_0.5_epochs_50.keras',
        'mt_gamma_0.6_epochs_50.keras',
        'mt_gamma_0.8_epochs_50.keras'
    ]

    mt_aug_names = [
        'mt_aug_gamma_0.2_epochs_50.keras',
        'mt_aug_gamma_0.4_epochs_50.keras',
        'mt_aug_gamma_0.5_epochs_50.keras',
        'mt_aug_gamma_0.6_epochs_50.keras',
        'mt_aug_gamma_0.8_epochs_50.keras'
    ]

    pd_name = 'pd_epochs_50.keras'
    pd_aug_name = 'pd_aug_epochs_50.keras'


    td_name = 'td_epochs_50.keras'
    td_aug_name = 'td_aug_epochs_50.keras'

    pd_labels = ['Effusion', 'Pneumothorax', 'Atelectasis', 'Cardiomegaly', 'Pneumonia']
    td_labels = ['Chest_drain_tube', 'NSG_tube', 'Endotracheal_tube', 'Tracheostomy_tube']


    data_dir = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/"
    model_dir =  "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/ObjectDetection/models/"
    output_dir = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/ObjectDetection/predictions/"


    pathology_detection_test = pd.read_csv(data_dir + 'Data/Data_splits/pathology_detection-test.csv', index_col=0)
    pathology_detection_test["ImagePath"] = pathology_detection_test["ImagePath"].apply(lambda x: x.replace("/home/data_shares/purrlab_students/", data_dir + "Data/"))

    x_test_pd = np.array([keras.utils.img_to_array(keras.utils.load_img(i)) for i in tqdm(pathology_detection_test["ImagePath"])])/255


    # MT
    for name in mt_names:
        model = keras.saving.load_model(model_dir + name)


        predictions = model.predict(x_test_pd)

        pred_df = pd.DataFrame(predictions[0], columns=pd_labels)

        pred_df.to_json(output_dir + "pd_" + name.replace(".keras", ".json"))

    # PD

    model = keras.saving.load_model(model_dir + pd_name)

    predictions = model.predict(x_test_pd)

    pred_df = pd.DataFrame(predictions, columns=pd_labels)

    pred_df.to_json(output_dir + pd_name.replace(".keras", ".json"))


    tube_detection_test = pd.read_csv("/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/" + "Annotation/Annotations_aggregated.csv", index_col=0)
    tube_detection_test = tube_detection_test.rename({"Chest_drain":"Chest_drain_tube"},axis=1)
    tube_detection_test["ImagePath"] = tube_detection_test["ImagePath"].apply(lambda x: x.replace("../../", data_dir))
    tube_detection_test = tube_detection_test.replace({-1:0})

    x_test_td = np.array([keras.utils.img_to_array(keras.utils.load_img(i)) for i in tqdm(tube_detection_test["ImagePath"])])/255


    # MT

    for name in mt_names:
        model = keras.saving.load_model(model_dir + name)


        predictions = model.predict(x_test_td)

        pred_df = pd.DataFrame(predictions[1], columns=td_labels)

        pred_df.to_json(output_dir + "td_" + name.replace(".keras", ".json"))

    # TD
    
    model = keras.saving.load_model(model_dir + td_name)

    predictions = model.predict(x_test_td)


    pred_df = pd.DataFrame(predictions, columns=td_labels)

    pred_df.to_json(output_dir + td_name.replace(".keras", ".json"))


    pathology_detection_test_aug = change_paths(pathology_detection_test ,"padchest-cropped")
    x_test_aug_pd = np.array([keras.utils.img_to_array(keras.utils.load_img(i)) for i in tqdm(pathology_detection_test_aug ["ImagePath"])])/255


    # MT AUG

    for name in mt_aug_names:
        model = keras.saving.load_model(model_dir + name)


        predictions = model.predict(x_test_aug_pd)

        pred_df = pd.DataFrame(predictions[0], columns=pd_labels)

        pred_df.to_json(output_dir + "pd_" + name.replace(".keras", ".json"))


    # PD AUG

    model = keras.saving.load_model(model_dir + pd_aug_name)

    predictions = model.predict(x_test_aug_pd)

    pred_df = pd.DataFrame(predictions, columns=pd_labels)

    pred_df.to_json(output_dir + pd_aug_name.replace(".keras", ".json"))



    tube_detection_test_aug = change_paths(tube_detection_test ,"padchest-cropped")
    x_test_aug_td = np.array([keras.utils.img_to_array(keras.utils.load_img(i)) for i in tqdm(tube_detection_test_aug["ImagePath"])])/255


    # MT AUG

    for name in mt_aug_names:
        model = keras.saving.load_model(model_dir + name)


        predictions = model.predict(x_test_aug_td)

        pred_df = pd.DataFrame(predictions[1], columns=td_labels)

        pred_df.to_json(output_dir + "td_" + name.replace(".keras", ".json"))


    # TD AUG

    model = keras.saving.load_model(model_dir + td_aug_name)

    predictions = model.predict(x_test_aug_td)

    pred_df = pd.DataFrame(predictions, columns=td_labels)

    pred_df.to_json(output_dir + td_aug_name.replace(".keras", ".json"))




if __name__ == "__main__":
    main()