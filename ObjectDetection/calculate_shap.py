import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle
os.environ["KERAS_BACKEND"] = "jax" 


import keras
import shap

# Function to change the image paths
def change_paths(df, data_directory):
    df = df.copy()


    df["ImagePath"] = df["ImagePath"].apply(lambda x: x.replace("padchest-preprocessed", data_directory))

    return df

mt_name = 'mt_gamma_0.8_epochs_50.keras'
mt_aug_name = 'mt_aug_gamma_0.8_epochs_50.keras'


pd_name = 'pd_epochs_50.keras'
pd_aug_name = 'pd_aug_epochs_50.keras'


td_name = 'td_epochs_50.keras'
td_aug_name = 'td_aug_epochs_50.keras'

pd_labels = ['Effusion', 'Pneumothorax', 'Atelectasis', 'Cardiomegaly', 'Pneumonia']
td_labels = ['Chest_drain_tube', 'NSG_tube', 'Endotracheal_tube', 'Tracheostomy_tube']


data_dir = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/"
model_dir =  "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/ObjectDetection/models/"
output_dir = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/ObjectDetection/shap_values/"


pathology_detection_test = pd.read_csv(data_dir + 'Data/Data_splits/pathology_detection-test.csv', index_col=0)
pathology_detection_test["ImagePath"] = pathology_detection_test["ImagePath"].apply(lambda x: x.replace("/home/data_shares/purrlab_students/", data_dir + "Data/"))


tube_detection_test = pd.read_csv("/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/" + "Annotation/Annotations_aggregated.csv", index_col=0)
tube_detection_test = tube_detection_test.rename({"Chest_drain":"Chest_drain_tube"},axis=1)
tube_detection_test["ImagePath"] = tube_detection_test["ImagePath"].apply(lambda x: x.replace("../../", data_dir))
tube_detection_test = tube_detection_test.replace({-1:0})

pathology_detection_test_aug = change_paths(pathology_detection_test ,"padchest-cropped")
tube_detection_test_aug = change_paths(tube_detection_test ,"padchest-cropped")


x_test_pd = np.array([keras.utils.img_to_array(keras.utils.load_img(i)) for i in tqdm(pathology_detection_test["ImagePath"])])
x_test_aug_pd = np.array([keras.utils.img_to_array(keras.utils.load_img(i)) for i in tqdm(pathology_detection_test_aug["ImagePath"])])

masker = shap.maskers.Image("inpaint_telea", x_test_pd[0].shape)

results_dir = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/ObjectDetection/predictions/"
result_names = os.listdir("/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/ObjectDetection/predictions")

pd_result_names = [i for i in result_names if i.startswith("pd") and i.endswith(".json")]
pd_preds = {i:pd.read_json(results_dir+i) for i in pd_result_names}


def get_most_confident_correct_and_incorrect_ids(name, preds, true, labels):
    preds_df = preds[name.replace("keras", "json")]

    most_confident_correct = [preds_df[true[i] == 1][i].idxmax() for i in labels]
    most_confident_incorrect = [preds_df[true[i] == 0][i].idxmax() for i in labels]


    print(most_confident_correct, most_confident_incorrect)
    return most_confident_correct, most_confident_incorrect

def f(X):
    tmp = X.copy()
    #preprocess_input(tmp)
    return model(tmp)

def f_mt_pd(X):
    tmp = X.copy()
    #preprocess_input(tmp)
    return model(tmp)[0]


pd_noaug_confident_correct, pd_noaug_confident_incorrect = get_most_confident_correct_and_incorrect_ids(pd_name, pd_preds, pathology_detection_test, pd_labels)
pd_aug_confident_correct, pd_aug_confident_incorrect = get_most_confident_correct_and_incorrect_ids(pd_aug_name, pd_preds, pathology_detection_test_aug, pd_labels)
pd_mt_noaug_confident_correct, pd_mt_noaug_confident_incorrect = get_most_confident_correct_and_incorrect_ids("pd_"+mt_name, pd_preds, pathology_detection_test, pd_labels)
pd_mt_aug_confident_correct, pd_mt_aug_confident_incorrect = get_most_confident_correct_and_incorrect_ids("pd_"+mt_aug_name, pd_preds, pathology_detection_test_aug, pd_labels)

shappable_ids = sorted(list(set(np.array([pd_noaug_confident_correct, pd_noaug_confident_incorrect,
pd_aug_confident_correct, pd_aug_confident_incorrect,
pd_mt_noaug_confident_correct, pd_mt_noaug_confident_incorrect,
pd_mt_aug_confident_correct, pd_mt_aug_confident_incorrect]).flatten())))





x_noaug = np.array([x_test_pd[i] for i in shappable_ids])/255
x_aug = np.array([x_test_aug_pd[i] for i in shappable_ids])/255



model = keras.saving.load_model(model_dir + pd_name)

explainer = shap.PartitionExplainer(f, masker, output_names=pd_labels, )
shap_values = explainer(
    x_noaug, max_evals=1000, batch_size=32
)

with open(output_dir+pd_name.replace("keras","pickle"), "+wb") as file:
    pickle.dump(shap_values, file)

model = keras.saving.load_model(model_dir + pd_aug_name)

explainer = shap.PartitionExplainer(f, masker, output_names=pd_labels, )
shap_values = explainer(
    x_aug, max_evals=1000, batch_size=32
)

with open(output_dir+pd_aug_name.replace("keras","pickle"), "+wb") as file:
    pickle.dump(shap_values, file)

model = keras.saving.load_model(model_dir + mt_name)

explainer = shap.PartitionExplainer(f_mt_pd, masker, output_names=pd_labels, )
shap_values = explainer(
    x_noaug, max_evals=1000, batch_size=32
)

with open(output_dir+mt_name.replace("keras","pickle"), "+wb") as file:
    pickle.dump(shap_values, file)


model = keras.saving.load_model(model_dir + mt_aug_name)

explainer = shap.PartitionExplainer(f_mt_pd, masker, output_names=pd_labels, )
shap_values = explainer(
    x_aug, max_evals=1000, batch_size=32
)

with open(output_dir+ mt_aug_name.replace("keras","pickle"), "+wb") as file:
    pickle.dump(shap_values, file)