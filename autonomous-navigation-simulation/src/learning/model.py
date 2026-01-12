import keras
from keras import layers, models

def get_model():
    """
    Compiles and builds a branching CNN model for direction prediction
    with less than 10M trainable parameters.

    Returns:
      The compiled Keras model.
    """
    # Input: A sequence of 5 grayscale images of size 240x320.
    input_layer = layers.Input(shape=(5, 240, 320, 1))

    ### Branch 1: Spatial features extraction from the last frame ###
    # For this branch, use only the last frame.
    spatial = layers.Cropping3D(cropping=((4, 0), (0, 0), (0, 0)),
                                data_format="channels_last")(input_layer)
    # Reshape from (1, 240, 320, 1) to (240, 320, 1)
    spatial = layers.Reshape((240, 320, 1))(spatial)

    # Use a couple of Conv2D layers with fewer filters
    spatial = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding="same")(spatial)
    spatial = layers.Dropout(rate=0.3)(spatial)
    spatial = layers.MaxPooling2D(pool_size=(3, 3))(spatial)

    spatial = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same")(spatial)
    spatial = layers.Dropout(rate=0.3)(spatial)
    spatial = layers.MaxPooling2D(pool_size=(3, 3))(spatial)

    spatial = layers.Flatten()(spatial)
    spatial = layers.Dense(32, activation='relu')(spatial)

    ### Branch 2: Temporal and spatiotemporal feature extraction using 3D convolutions ###
    # Use the full 5-frame sequence.
    temporal = layers.Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', padding="same")(input_layer)
    temporal = layers.Dropout(rate=0.3)(temporal)
    temporal = layers.MaxPooling3D(pool_size=(1, 3, 3), padding="same")(temporal)

    temporal = layers.Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu', padding="same")(temporal)
    temporal = layers.Dropout(rate=0.3)(temporal)
    temporal = layers.MaxPooling3D(pool_size=(1, 3, 3), padding="same")(temporal)

    temporal = layers.Flatten()(temporal)
    temporal = layers.Dense(16, activation='relu')(temporal)

    ### Merge branches ###
    merged = layers.concatenate([spatial, temporal])

    ### Dense layers ###
    x = layers.Dense(32, activation='relu')(merged)
    x = layers.Dropout(rate=0.1)(x)
    # Final classification over 3 classes.
    output = layers.Dense(3, activation='softmax')(x)

    # Model creation and compilation
    model = models.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

