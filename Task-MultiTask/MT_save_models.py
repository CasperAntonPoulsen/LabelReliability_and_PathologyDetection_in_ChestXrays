# Imports
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model

# For the cluster specifically
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


np.set_printoptions(threshold=10000)   # Making sure it will return all preds and not just "x,y...z"


### Helper functions

# Function for creating the n-hot encoding
def get_n_hot_encoding(df, labels_to_encode):
    enc = np.zeros((len(df), len(labels_to_encode)))
    for idx, row in df.iterrows():
        for ldx, l in enumerate(labels_to_encode):
            if row[l] == 1:
                enc[idx][ldx] = 1
    return enc


# Function to change the image paths
def change_paths(df):
    df = df.copy()

    #df["ImagePath"] = ["../../Data/" + df["ImagePath"][idx][11:] for idx in range(len(df))]

    return df


## Data loading and preprocessing

train_data_padchest = pd.read_csv('../Data/Data_splits/pathology_detection-train.csv', index_col=0)
tube_detection_finetuning = pd.read_csv("../Data/Data_splits/tube_detection-finetuning.csv", index_col=0)
#augmented_masks_padchest = pd.read_json("../Data/Masks/augmented_masks.ndjson", orient="records")

# Concatenating the datasets for fine-tuning and shuffling
finetune_df = pd.concat([train_data_padchest, tube_detection_finetuning])
finetune_df = finetune_df.sample(frac=1, random_state=321).reset_index(drop=True)

# Changing the image paths, so they fit to res24
finetune_df = change_paths(finetune_df)


img_generator = image.ImageDataGenerator(rescale=1./255)  # Normalizing the data

generator_train_padchest_pd = img_generator.flow_from_dataframe(dataframe = finetune_df, 
    x_col='ImagePath',
    y_col='Pneumothorax',
    target_size=(512, 512),
    classes=None,
    class_mode='raw',
    batch_size=128,
    shuffle=False,
    validate_filenames=False)

generator_train_padchest_td = img_generator.flow_from_dataframe(dataframe = finetune_df, 
    x_col='ImagePath',
    y_col='Pneumothorax',
    target_size=(512, 512),
    classes=None,
    class_mode='raw',
    batch_size=128,
    shuffle=False,
    validate_filenames=False)


# N-hot encoding the labels
labels_to_encode = ['Effusion', 'Pneumothorax', 'Atelectasis', 'Cardiomegaly', 'Pneumonia']
y_train_pd = get_n_hot_encoding(finetune_df, labels_to_encode)
generator_train_padchest_pd._targets=y_train_pd

labels_to_encode = ['Chest_drain_tube', 'NSG_tube', 'Endotracheal_tube', 'Tracheostomy_tube']
y_train_td = get_n_hot_encoding(finetune_df, labels_to_encode)
generator_train_padchest_td._targets=y_train_td

## Modeling

# Helper function for the multitask setup
def generator_double_output(gen1, gen2):
    while True:
        X1 = gen1.next()
        X2 = gen2.next()
        yield X1[0], [X1[1], X2[1]] 


# Function for running the multitask model
def PD_save_models(json_name, h5_name, gamma ,lr=0.00001):

    # Get the DenseNet ImageNet weights
    densenet_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(512,512,3))

    # Specify whether to train on the loaded weights
    for layer in densenet_model.layers[:-9]:
        layer.trainable = False
    for layer in densenet_model.layers[-9:]:
        layer.trainable = True

    # Building further upon the last layer (ReLu)
    input_tensor = densenet_model.output
    x = GlobalAveragePooling2D(keepdims=True)(input_tensor)
    x = Flatten()(x)

    # Segmentation layer

    # Object detection layer
    

    # Pathology detection layer
    PD = Dense(5, activation='sigmoid', name="PD_output")(x)
    
    # Tube detection layer
    TD = Dense(4, activation='sigmoid', name="TD_output")(x)
    
    model = Model(inputs=densenet_model.inputs, outputs=[PD, TD])

    adam = tf.optimizers.Adam(learning_rate = lr)
    model.compile(
        loss={"PD_output": tf.keras.losses.BinaryCrossentropy(), "TD_output": tf.keras.losses.BinaryCrossentropy()}, 
        loss_weights={"PD_output":gamma, "TD_output":1-gamma}, 
        optimizer=adam, 
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    history = model.fit(generator_double_output(generator_train_padchest_pd, generator_train_padchest_td), epochs=epochs, steps_per_epoch =len(finetune_df)/32,  verbose=1)

    # Saving the model 
    model_json = model.to_json()

    with open(json_name, "w") as json_file:
        json_file.write(model_json)

    model.save_weights(h5_name)
    

    # Get performances: train=PD, Val=TD
    train_acc = [float("{:.3f}".format(i)) for i in history.history["PD_output_accuracy"]]
    train_auc = [float("{:.3f}".format(i)) for i in history.history["PD_output_auc"]]
    train_loss = [float("{:.3f}".format(i)) for i in history.history["PD_output_loss"]]
    val_acc = [float("{:.3f}".format(i)) for i in history.history["TD_output_accuracy"]]
    val_auc = [float("{:.3f}".format(i)) for i in history.history["TD_output_auc"]]
    val_loss = [float("{:.3f}".format(i)) for i in history.history["TD_output_loss"]]    

    # Model details
    model_name = "Learning_rate: {}".format(lr)

    return [model_name, train_acc, train_auc, train_loss, val_acc, val_auc, val_loss]


# Defining the model hyperparameters
epochs = 30
learning_rate_list = [0.00001]
gamma_list = [ 0.8]
path = ""

# Running the model function
all_dict = {"Model name": [], "Train acc model1": [], "Train auc model1": [], "Train loss model1": [], "Train acc model2": [], "Train auc model2": [], "Train loss model2": [], "Train acc model3": [], "Train auc model3": [], "Train loss model3": [], "Train acc avg": [], "Train auc avg": [], "Train loss avg": [], "Val acc model1": [], "Val auc model1": [], "Val loss model1": [], "Val acc model2": [], "Val auc model2": [], "Val loss model2": [], "Val acc model3": [], "Val auc model3": [], "Val loss model3": [], "Val acc avg": [], "Val auc avg": [], "Val loss avg": []}

# Making csv with header
df_acc = pd.DataFrame(data=all_dict)

filename = "MT_save_models_optimized.csv"
df_acc.to_csv(filename, mode='a', sep=',')


for gamma in gamma_list:
    all_dict = {"Model name": [], "Train acc model1": [], "Train auc model1": [], "Train loss model1": [], "Train acc model2": [], "Train auc model2": [], "Train loss model2": [], "Train acc model3": [], "Train auc model3": [], "Train loss model3": [], "Train acc avg": [], "Train auc avg": [], "Train loss avg": [], "Val acc model1": [], "Val auc model1": [], "Val loss model1": [], "Val acc model2": [], "Val auc model2": [], "Val loss model2": [], "Val acc model3": [], "Val auc model3": [], "Val loss model3": [], "Val acc avg": [], "Val auc avg": [], "Val loss avg": []}

    model_acc1 = PD_save_models(json_name=path+f"MT_model1_gamma_{gamma}_op.json", h5_name=path+f"MT_model1_gamma_{gamma}_op.h5", gamma=gamma,lr=learning_rate_list[0])
    #model_acc2 = PD_save_models(json_name=path+f"MT_model2_gamma_{gamma}.json", h5_name=path+f"MT_model2_gamma_{gamma}.h5", gamma=gamma,lr=learning_rate_list[0])
    #model_acc3 = PD_save_models(json_name=path+f"MT_model3_gamma_{gamma}.json", h5_name=path+f"MT_model3_gamma_{gamma}.h5", gamma=gamma,lr=learning_rate_list[0])

    all_dict["Model name"].append(model_acc1[0])

    # train=PD, Val=TD
    all_dict["Train acc model1"].append(model_acc1[1])
    all_dict["Train auc model1"].append(model_acc1[2])
    all_dict["Train loss model1"].append(model_acc1[3])
    # all_dict["Train acc model2"].append(model_acc2[1])
    # all_dict["Train auc model2"].append(model_acc2[2])
    # all_dict["Train loss model2"].append(model_acc2[3])
    # all_dict["Train acc model3"].append(model_acc3[1])
    # all_dict["Train auc model3"].append(model_acc3[2])
    # all_dict["Train loss model3"].append(model_acc3[3])

    all_dict["Val acc model1"].append(model_acc1[4])
    all_dict["Val auc model1"].append(model_acc1[5])
    all_dict["Val loss model1"].append(model_acc1[6])
    # all_dict["Val acc model2"].append(model_acc2[4])
    # all_dict["Val auc model2"].append(model_acc2[5])
    # all_dict["Val loss model2"].append(model_acc2[6])
    # all_dict["Val acc model3"].append(model_acc3[4])
    # all_dict["Val auc model3"].append(model_acc3[5])
    # all_dict["Val loss model3"].append(model_acc3[6])

    all_avg_train_acc = []
    all_avg_train_auc = []
    all_avg_train_loss = []
    all_avg_val_acc = []
    all_avg_val_auc = []
    all_avg_val_loss = []


    # for i in range(epochs):
    #     avg_train_acc = sum([model_acc1[1][i], model_acc2[1][i], model_acc3[1][i]])/3
    #     all_avg_train_acc.append(float("{:.2f}".format(avg_train_acc)))

    #     avg_train_auc = sum([model_acc1[2][i], model_acc2[2][i], model_acc3[2][i]])/3
    #     all_avg_train_auc.append(float("{:.2f}".format(avg_train_auc)))

    #     avg_train_loss = sum([model_acc1[3][i], model_acc2[3][i], model_acc3[3][i]])/3
    #     all_avg_train_loss.append(float("{:.3f}".format(avg_train_loss)))

    #     avg_val_acc = sum([model_acc1[4][i], model_acc2[4][i], model_acc3[4][i]])/3
    #     all_avg_val_acc.append(float("{:.2f}".format(avg_val_acc)))

    #     avg_val_auc = sum([model_acc1[5][i], model_acc2[5][i], model_acc3[5][i]])/3
    #     all_avg_val_auc.append(float("{:.2f}".format(avg_val_auc)))

    #     avg_val_loss = sum([model_acc1[6][i], model_acc2[6][i], model_acc3[6][i]])/3
    #     all_avg_val_loss.append(float("{:.3f}".format(avg_val_loss)))

    # all_dict["Train acc avg"].append(all_avg_train_acc)
    # all_dict["Train auc avg"].append(all_avg_train_auc)
    # all_dict["Train loss avg"].append(all_avg_train_loss)

    # all_dict["Val acc avg"].append(all_avg_val_acc)
    # all_dict["Val auc avg"].append(all_avg_val_auc)
    # all_dict["Val loss avg"].append(all_avg_val_loss)


    # Making csv with each 3 models 
    df_acc = pd.DataFrame(data=all_dict)
    df_acc.to_csv(filename, mode='a', header=False, sep=',')
