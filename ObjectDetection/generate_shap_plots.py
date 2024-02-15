import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

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
output_dir = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/ObjectDetection/shap_plots/"


pathology_detection_test = pd.read_csv(data_dir + 'Data/Data_splits/pathology_detection-test.csv', index_col=0)
pathology_detection_test["ImagePath"] = pathology_detection_test["ImagePath"].apply(lambda x: x.replace("/home/data_shares/purrlab_students/", data_dir + "Data/"))


tube_detection_test = pd.read_csv("/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/" + "Annotation/Annotations_aggregated.csv", index_col=0)
tube_detection_test = tube_detection_test.rename({"Chest_drain":"Chest_drain_tube"},axis=1)
tube_detection_test["ImagePath"] = tube_detection_test["ImagePath"].apply(lambda x: x.replace("../../", data_dir))
tube_detection_test = tube_detection_test.replace({-1:0})

pathology_detection_test_aug = change_paths(pathology_detection_test ,"padchest-cropped")
tube_detection_test_aug = change_paths(tube_detection_test ,"padchest-cropped")

results_dir = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/ObjectDetection/predictions/"
result_names = os.listdir("/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/ObjectDetection/predictions")
pd_result_names = [i for i in result_names if i.startswith("pd") and i.endswith(".json")]
pd_preds = {i:pd.read_json(results_dir+i) for i in pd_result_names}
td_result_names = [i for i in result_names if i.startswith("td") and i.endswith(".json")]
td_preds = {i:pd.read_json(results_dir+i) for i in td_result_names}

def get_most_confident_correct_and_incorrect_ids(name, preds, true, labels):
    preds_df = preds[name.replace("keras", "json")]

    most_confident_correct = [preds_df[true[i] == 1][i].idxmax() for i in labels]
    most_confident_incorrect = [preds_df[true[i] == 0][i].idxmax() for i in labels]


    print(most_confident_correct, most_confident_incorrect)
    return most_confident_correct, most_confident_incorrect

pd_noaug_confident_correct, pd_noaug_confident_incorrect = get_most_confident_correct_and_incorrect_ids(pd_name, pd_preds, pathology_detection_test, pd_labels)
pd_aug_confident_correct, pd_aug_confident_incorrect = get_most_confident_correct_and_incorrect_ids(pd_aug_name, pd_preds, pathology_detection_test_aug, pd_labels)
pd_mt_noaug_confident_correct, pd_mt_noaug_confident_incorrect = get_most_confident_correct_and_incorrect_ids("pd_"+mt_name, pd_preds, pathology_detection_test, pd_labels)
pd_mt_aug_confident_correct, pd_mt_aug_confident_incorrect = get_most_confident_correct_and_incorrect_ids("pd_"+mt_aug_name, pd_preds, pathology_detection_test_aug, pd_labels)


shappable_ids = sorted(list(set(np.array([pd_noaug_confident_correct, pd_noaug_confident_incorrect,
pd_aug_confident_correct, pd_aug_confident_incorrect,
pd_mt_noaug_confident_correct, pd_mt_noaug_confident_incorrect,
pd_mt_aug_confident_correct, pd_mt_aug_confident_incorrect]).flatten())))

shap_id_df = pd.DataFrame([pd_noaug_confident_correct, pd_noaug_confident_incorrect,
pd_aug_confident_correct, pd_aug_confident_incorrect,
pd_mt_noaug_confident_correct, pd_mt_noaug_confident_incorrect,
pd_mt_aug_confident_correct, pd_mt_aug_confident_incorrect], columns=pd_labels)


shap_id_df["name"]=[
    "pd_noaug",
    "pd_noaug",
    "pd_aug",
    "pd_aug",
    "mt_noaug",
    "mt_noaug",
    "mt_aug",
    "mt_aug",
]

shap_id_df["answered_correct?"]=[
    "correct",
    "incorrect",
    "correct",
    "incorrect",
    "correct",
    "incorrect",
    "correct",
    "incorrect",
]


shappable_index = {v:i for i, v in enumerate(shappable_ids)}

shap_folder = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/ObjectDetection/shap_values/"

with open(shap_folder + "pd_epochs_50.pickle", "rb") as file:
    shap_values_pd_noaug = pickle.load(file)

with open(shap_folder + "pd_aug_epochs_50.pickle", "rb") as file:
    shap_values_pd_aug  = pickle.load(file)

with open(shap_folder + "mt_gamma_0.8_epochs_50.pickle", "rb") as file:
    shap_values_mt_noaug  = pickle.load(file)

with open(shap_folder + "mt_aug_gamma_0.8_epochs_50.pickle", "rb") as file:
   shap_values_mt_aug  = pickle.load(file)

shap_dict ={
    "pd_noaug": shap_values_pd_noaug,
    "pd_aug": shap_values_pd_aug,
    "mt_noaug":shap_values_mt_noaug,
    "mt_aug":shap_values_mt_aug,
}

correct_df = shap_id_df[shap_id_df["answered_correct?"] == "correct"]
incorrect_df = shap_id_df[shap_id_df["answered_correct?"] == "incorrect"]

seen_ids = list()

for label in pd_labels:
    print(label)

    for i in range(0,2):

        xray_id = correct_df.iloc[i][label]
        if xray_id in seen_ids:
            continue
        print(f"For X-ray with id: {xray_id}")

        
        for index in  shap_id_df[shap_id_df == xray_id].dropna(how="all").dropna(axis=1, how="all").index:
            
            name = shap_id_df.iloc[index]["name"]
            was_correct = shap_id_df.iloc[index]["answered_correct?"]
            confident_guesses = list(shap_id_df[shap_id_df == xray_id].iloc[index].dropna(how="all").keys())
            if was_correct == "correct":
                print(f"Model: {name} most confident correct answer was {confident_guesses} for this x-ray")
            else:
                print(f"Model: {name} most confident wrong answers were {confident_guesses} for this x-ray")
    
        
        for key in shap_dict.keys():
            #print(key)
            shap_index = shappable_index[xray_id]
            shap.image_plot(shap_dict[key][shap_index:(shap_index+1)], show=False)
            plt.savefig(output_dir+f"{xray_id}_{key}.png")
            plt.close()


        seen_ids.append(xray_id)

