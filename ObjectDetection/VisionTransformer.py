# Code from https://keras.io/examples/vision/object_detection_using_vision_transformer/
#
import os
os.environ["KERAS_BACKEND"] = "jax" 

import numpy as np
import keras
from keras import layers
from keras import ops
import matplotlib.pyplot as plt
import numpy as np
import cv2

import scipy.io
import shutil
import pandas as pd
import pickle
from tqdm import tqdm
# For the cluster specifically 
# Set global variable

image_size = 224  # resize input images to this size
patch_size = 64  # Size of the patches to be extracted from the input images
input_shape = (image_size, image_size, 3)  # input image shape
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 64
num_epochs = 50
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
num_patches = (image_size // patch_size) ** 2



# Size of the transformer layers
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
transformer_layers = 4
mlp_head_units = [2048, 1024, 512, 64, 32]  # Size of the dense layers



def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


@keras.saving.register_keras_serializable(name=None)
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
    
@keras.saving.register_keras_serializable(name=None)
class PatchEncoder(layers.Layer):
    def __init__(self,input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers, mlp_head_units,**kwargs):
        super().__init__(**kwargs)

        self.input_shape = input_shape
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)

        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

        self.num_heads = num_heads
        self.transformer_units = transformer_units
        self.transformer_layers = transformer_layers
        self.mlp_head_units = mlp_head_units

    # Override function to avoid error while saving model
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "input_shape": self.input_shape,
                "patch_size": self.patch_size,
                "num_patches": self.num_patches,
                "projection_dim": self.projection_dim,
                "num_heads": self.num_heads,
                "transformer_units": self.transformer_units,
                "transformer_layers": self.transformer_layers,
                "mlp_head_units": self.mlp_head_units,
            }
        )
        return config

    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded
    


def create_vit_object_detector(
    input_shape,
    patch_size,
    num_patches,
    projection_dim,
    num_heads,
    transformer_units,
    transformer_layers,
    mlp_head_units,
):
    inputs = keras.Input(shape=input_shape)
    # Create patches
    patches = Patches(patch_size)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers, mlp_head_units)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)

    bounding_box = layers.Dense(4)(
        features
    )  # Final four neurons that output bounding box

    # return Keras model.
    return keras.Model(inputs=inputs, outputs=bounding_box)


def run_experiment(model, learning_rate, weight_decay, batch_size, num_epochs, x_train, y_train):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    # Compile model.
    model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())

    checkpoint_filepath = "vit_object_detector.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[
            checkpoint_callback,
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        ],
    )

    return history


# To calculate IoU (intersection over union, given two bounding boxes)
def bounding_box_intersection_over_union(box_predicted, box_truth):
    # get (x, y) coordinates of intersection of bounding boxes
    top_x_intersect = max(box_predicted[0], box_truth[0])
    top_y_intersect = max(box_predicted[1], box_truth[1])
    bottom_x_intersect = min(box_predicted[2], box_truth[2])
    bottom_y_intersect = min(box_predicted[3], box_truth[3])

    # calculate area of the intersection bb (bounding box)
    intersection_area = max(0, bottom_x_intersect - top_x_intersect + 1) * max(
        0, bottom_y_intersect - top_y_intersect + 1
    )

    # calculate area of the prediction bb and ground-truth bb
    box_predicted_area = (box_predicted[2] - box_predicted[0] + 1) * (
        box_predicted[3] - box_predicted[1] + 1
    )
    box_truth_area = (box_truth[2] - box_truth[0] + 1) * (
        box_truth[3] - box_truth[1] + 1
    )

    # calculate intersection over union by taking intersection
    # area and dividing it by the sum of predicted bb and ground truth
    # bb areas subtracted by  the interesection area

    # return ioU
    return intersection_area / float(
        box_predicted_area + box_truth_area - intersection_area
    )



def main():


    data_path = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/ObjectDetection/padchest_object_detection.json"

    df = pd.read_json(data_path).sample(n=50000, random_state=1)

    image_paths = list(df["ImagePath"])
    annots = list(df["bbox_512"])



    images, targets = [], []

    # loop over the annotations and images, preprocess them and store in lists
    for i in tqdm(range(0, len(annots))):
        # Access bounding box coordinates
        #annot = scipy.io.loadmat(path_annot + annot_paths[i])["box_coord"][0]
        annot = annots[i] 

        # top_left_x, top_left_y = annot[2], annot[0]
        # bottom_right_x, bottom_right_y = annot[3], annot[1]

        top_left_x, top_left_y = annot[0], annot[1]
        bottom_right_x, bottom_right_y = annot[2], annot[3]

        image = keras.utils.load_img(
            image_paths[i],
        )
        (w, h) = image.size[:2]

        # resize images
        image = image.resize((image_size, image_size))

        # convert image to array and append to list
        images.append(keras.utils.img_to_array(image))

        # apply relative scaling to bounding boxes as per given image and append to list
        targets.append(
            (
                float(top_left_x) / w,
                float(top_left_y) / h,
                float(bottom_right_x) / w,
                float(bottom_right_y) / h,
            )
        )

    # Convert the list to numpy array, split to train and test dataset
    (x_train), (y_train) = (
        np.asarray(images[: int(len(images) * 0.8)]),
        np.asarray(targets[: int(len(targets) * 0.8)]),
    )
    (x_test), (y_test) = (
        np.asarray(images[int(len(images) * 0.8) :]),
        np.asarray(targets[int(len(targets) * 0.8) :]),
    )

    history = []

    vit_object_detector = create_vit_object_detector(
        input_shape,
        patch_size,
        num_patches,
        projection_dim,
        num_heads,
        transformer_units,
        transformer_layers,
        mlp_head_units,
    )

    # Train model
    history = run_experiment(
        vit_object_detector, learning_rate, weight_decay, batch_size, num_epochs, x_train, y_train
    )
    try:
        pickle.dump(history, "history_100it_50000sample.pickle")
    except Exception:
        print("Could not pickle")
    finally:
        print(history)


    vit_object_detector.save("vit_object_detector_50000_samples.keras")

    i, mean_iou = 0, 0

    # Compare results for 10 images in the test set
    for input_image in x_test[:10]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
        im = input_image

        # Display the image
        ax1.imshow(im.astype("uint8"))
        ax2.imshow(im.astype("uint8"))

        input_image = cv2.resize(
            input_image, (image_size, image_size), interpolation=cv2.INTER_AREA
        )
        input_image = np.expand_dims(input_image, axis=0)
        preds = vit_object_detector.predict(input_image)[0]

        (h, w) = (im).shape[0:2]

        top_left_x, top_left_y = int(preds[0] * w), int(preds[1] * h)

        bottom_right_x, bottom_right_y = int(preds[2] * w), int(preds[3] * h)

        box_predicted = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        # Create the bounding box

        top_left_x, top_left_y = int(y_test[i][0] * w), int(y_test[i][1] * h)

        bottom_right_x, bottom_right_y = int(y_test[i][2] * w), int(y_test[i][3] * h)

        box_truth = top_left_x, top_left_y, bottom_right_x, bottom_right_y

        mean_iou += bounding_box_intersection_over_union(box_predicted, box_truth)

        i = i + 1

    print("mean_iou: " + str(mean_iou / len(x_test[:10])))


if __name__ == "__main__":
    main()