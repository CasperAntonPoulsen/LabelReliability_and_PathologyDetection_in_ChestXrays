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

# Function for creating the n-hot encoding
def get_n_hot_encoding(df, labels_to_encode):
    enc = np.zeros((len(df), len(labels_to_encode)))
    for idx, row in df.iterrows():
        for ldx, l in enumerate(labels_to_encode):
            if row[l] == 1:
                enc[idx][ldx] = 1
    return enc

# Function to change the image paths
def change_paths(df, data_directory):
    df = df.copy()


    df["ImagePath"] = df["ImagePath"].apply(lambda x: x.replace("/home/data_shares/purrlab_students/", data_directory))

    return df


# Function for running the multitask model
def PD_save_models(
        x_train, 
        y_train_td, 
        x_val,
        y_val_td,
        epochs, 
        lr,name, 
        out_dir
    ):

    # Get the DenseNet ImageNet weights
    densenet_model = keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(512,512,3))

    # Specify whether to train on the loaded weights
    for layer in densenet_model.layers[:-9]:
        layer.trainable = False
    for layer in densenet_model.layers[-9:]:
        layer.trainable = True

    # Building further upon the last layer (ReLu)
    input_tensor = densenet_model.output
    x = GlobalAveragePooling2D(keepdims=True)(input_tensor)
    x = Flatten()(x)


    # Tube detection layer
    TD = Dense(4, activation='sigmoid', name="TD_output")(x)
    
    model = Model(inputs=densenet_model.inputs, outputs=TD)

    adam = keras.optimizers.Adam(learning_rate = lr)

    checkpoint_filepath = out_dir + f"checkpoint.{name}_epochs_{epochs}.keras"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_auc',
    mode='max',
    save_best_only=True)

    model.compile(
        loss={ "TD_output": keras.losses.BinaryCrossentropy()}, 
        optimizer=adam, 
        metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.BinaryCrossentropy(name="loss")]
    )
    model.summary()
    history = model.fit(
        x_train,
        y_train_td, 
        batch_size=128,
        epochs=epochs,  
        verbose=2,
        validation_data=(x_val,y_val_td),
        callbacks=[model_checkpoint_callback]
    )

    model.save(out_dir + f"{name}_epochs_{epochs}.keras")
    print(history.history)
    # Get performances: train=PD, Val=TD

    return history



def main():
    # Defining the model hyperparameters
    name = sys.argv[1]
    epochs = 50
    learning_rate = 0.00001
    data_dir = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/Data/"
    output_dir = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/ObjectDetection/models/"


    tube_detection_train = pd.read_csv(data_dir + "Data_splits/tube_detection-finetuning.csv", index_col=0)
    tube_detection_val = pd.read_csv(data_dir + "Data_splits/tube_detection-finetuning_val.csv", index_col=0)

    # Concatenating the datasets for fine-tuning and shuffling
    train_df = pd.concat([tube_detection_train])
    train_df = train_df.sample(frac=1, random_state=321).reset_index(drop=True)

    val_df = pd.concat([tube_detection_val])
    val_df = val_df.sample(frac=1, random_state=321).reset_index(drop=True)

    # Changing the image paths, so they fit to res24
    train_df = change_paths(train_df, data_dir)
    val_df = change_paths(val_df, data_dir)  

    # N-hot encoding the labels
    labels_to_encode = ['Chest_drain_tube', 'NSG_tube', 'Endotracheal_tube', 'Tracheostomy_tube']
    y_train_td = get_n_hot_encoding(train_df, labels_to_encode)
    y_val_td = get_n_hot_encoding(val_df, labels_to_encode)

    #rescaler = keras.layers.Rescaling(scale=1./255)
    # Load data into CPU memory
    x_train = np.array([keras.utils.img_to_array(keras.utils.load_img(i)) for i in tqdm(train_df["ImagePath"])])/255
    x_val = np.array([keras.utils.img_to_array(keras.utils.load_img(i)) for i in tqdm(val_df["ImagePath"])])/255


    # Train model
    model_history = PD_save_models(
        x_train, 
        y_train_td,
        x_val,
        y_val_td,
        epochs=epochs,
        lr=learning_rate, 
        name=name,
        out_dir=output_dir
    )

    # Save metrics
    filename = f"{name}_history_epochs_{epochs}.json"
    df_history = pd.DataFrame(data=model_history.history)
    df_history.to_json(output_dir+filename)


if __name__ == "__main__":
    main()