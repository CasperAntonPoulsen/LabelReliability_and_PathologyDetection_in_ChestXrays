### Imports
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

# For the cluster specifically
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


np.set_printoptions(threshold=10000000)   # Making sure it will return all preds and not just "x,y...z"


### Helper functions

# Function for loading the models
def load_model(file_json, file_h5):
    json_file = open(file_json, 'r')               
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(file_h5)
    
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Function for making predictions
def make_predictions(model, generator):
    predictions = (model.predict(generator))
    return predictions


# Function for creating the n-hot encoding
def get_n_hot_encoding(df, labels_to_encode):
    enc = np.zeros((len(df), len(labels_to_encode)))
    for idx, row in df.iterrows():
        for ldx, l in enumerate(labels_to_encode):
            if row[l] == 1:
                enc[idx][ldx] = 1
    return enc


## Data loading and preprocessing

test_padchest = pd.read_csv('../Data/Data_splits/pathology_detection-test.csv', index_col=0)
#annotations = pd.read_csv('../Data/Annotations/Annotations_aggregated.csv', index_col=0)

#test_df = pd.concat([test_padchest, annotations])

img_generator = image.ImageDataGenerator(rescale=1./255)  # Normalizing the data

generator_test_padchest = img_generator.flow_from_dataframe(dataframe = test_padchest, 
    x_col='ImagePath',
    y_col='Pneumothorax',
    target_size=(512, 512),
    classes=None,
    class_mode='raw',
    batch_size=256,
    shuffle=False,
    validate_filenames=False)


### To store the predictions
path = "/home/caap/LabelReliability_and_PathologyDetection_in_ChestXrays/Models/"
all_dict = {"Model_name": [], "Val_data": [], "Preds": []}
df_acc = pd.DataFrame(data=all_dict)
filename = "/home/caap/LabelReliability_and_PathologyDetection_in_ChestXrays/Predictions/MT_preds_PD.csv"
df_acc.to_csv(filename, mode='a', sep=',')


## Get predictions

gamma_list = [0.2, 0.4, 0.5, 0.6]



#json = [path+'MT_model1.json', path+'MT_model2.json', path+'MT_model3.json']
#h5 = [path+'MT_model1.h5', path+'MT_model2.h5', path+'MT_model3.h5']

### Adding the predictions to the dataframe
all_dict = {"Model_name": [], "Val_data": [], "Preds": [] }


for gamma in gamma_list:
    model = load_model(path + f"MT_model1_gamma_{gamma}_op.json", path + f"MT_model1_gamma_{gamma}_op.h5")
    pred = make_predictions(model, generator_test_padchest)[0]


    all_dict["Model_name"].append(f'MT, Multiclass, gamma {gamma}')
    all_dict["Val_data"].append('PadChest, PD')
    k = "Preds"
    all_dict[k].append(pred)


df_acc = pd.DataFrame(data=all_dict)
df_acc.to_csv(filename, mode='a', header=False, sep=',')

