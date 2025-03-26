
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
def unet_model_with_pooling(output_channels: int):
    """
    U-Net model with pooling layers for image-to-image tasks, updated for 256x256 input/output size.

    Args:
        output_channels: Number of output channels (e.g., 3 for RGB images).

    Returns:
        A tf.keras.Model representing the U-Net with pooling.
    """
    # Input layer
    inputs = tf.keras.layers.Input(shape=[512, 512, 3])

    # Downsampling layers (added MaxPooling layers)
    down_stack = [
        downsample_with_pooling(64, 3, apply_batchnorm=False),  # 256x256 -> 128x128
        downsample_with_pooling(64, 3, apply_batchnorm=False),  # 256x256 -> 128x128
        downsample_with_pooling(128, 3),                       # 128x128 -> 64x64
        downsample_with_pooling(256, 3),                       # 64x64 -> 32x32
        downsample_with_pooling(512, 3),                       # 32x32 -> 16x16
        downsample_with_pooling(512, 3),                       # 16x16 -> 8x8
        downsample_with_pooling(512, 3),                       # 8x8 -> 4x4
    ]

    # Upsampling layers
    up_stack = [
        upsample(512, 3),  # 4x4 -> 8x8
        upsample(512, 3),  # 8x8 -> 16x16
        upsample(512, 3),  # 16x16 -> 32x32
        upsample(256, 3),  # 32x32 -> 64x64
        upsample(128, 3),  # 64x64 -> 128x128
        upsample(64, 3),   # 128x128 -> 256x256
        upsample(64, 3),   # 256x256 -> 512x512
    ]

    # Downsampling through the model
    skips = []
    x = inputs
    for down in down_stack:
        x = down(x)
        skips.append(x)

    # The last element in `skips` corresponds to the smallest feature map.
    # For skip connections, exclude the last element (the bottleneck layer).
    cnt = len(skips) - 2  # Index of the last skip connection (4 for this example).

    # Upsampling and skip connections
    for up in up_stack:
        x = up(x)
        print("PHILIP:", skips)
        if cnt >= 0:  # Check if there are any skips left
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skips[cnt]])
            cnt -= 1

    # Output layer
    last = tf.keras.layers.Conv2D(
        filters=output_channels, kernel_size=1, strides=1, padding='same', activation='sigmoid'
    )

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

# Downsampling block

def downsample_with_pooling(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    # Depthwise Convolution
    result.add(tf.keras.layers.DepthwiseConv2D(kernel_size=size, strides=1, padding='same',
                                               depthwise_initializer=initializer, use_bias=False))
    
    # Pointwise Convolution
    result.add(tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same',
                                      kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    result.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result
# 2. Define SSIM loss function
@tf.keras.utils.register_keras_serializable()
def ssim_loss(y_true, y_pred):
    """
    SSIM Loss for image quality enhancement.
    Args:
        y_true: Ground truth images.
        y_pred: Predicted images.
    Returns:
        A scalar loss value based on SSIM.
    """
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return 1.0 - tf.reduce_mean(ssim)

@tf.keras.utils.register_keras_serializable()
def hybrid_loss(y_true, y_pred):
    """
    Hybrid loss combining SSIM and MAE.
    Args:
        y_true: Ground truth images.
        y_pred: Predicted images.
    Returns:
        A scalar loss value combining SSIM and MAE.
    """
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    ssim_loss = 1.0 - tf.reduce_mean(ssim)
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    return ssim_loss + mae_loss



# 5. Dataset preparation (for training)
def load_image(image_file):
    """
    Loads and preprocesses an input image.
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [512, 512])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize image to [0, 1]
    return image
def load_and_preprocess_image(file_path):
    # Ensure the file path is correctly handled as a string
    image = tf.io.read_file(tf.strings.join([file_path]))  # Concatenate to ensure string type
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [512, 512])
    image = tf.cast(image, tf.float32) / 255.0
    return image
def prepare_dataset(input_dir, gt_dir, batch_size=32):
    # Create file paths and ensure they are handled as strings
    input_images = [os.path.join(input_dir, fname) for fname in sorted(os.listdir(input_dir))]
    gt_images = [os.path.join(gt_dir, fname) for fname in sorted(os.listdir(gt_dir))]

    # Convert lists to TensorFlow constants to enforce the dtype as string
    input_images = tf.constant(input_images, dtype=tf.string)
    gt_images = tf.constant(gt_images, dtype=tf.string)

    # Create Dataset objects for input and ground truth
    input_dataset = tf.data.Dataset.from_tensor_slices(input_images).map(load_and_preprocess_image)
    gt_dataset = tf.data.Dataset.from_tensor_slices(gt_images).map(load_and_preprocess_image)

    # Pair and batch the datasets
    dataset = tf.data.Dataset.zip((input_dataset, gt_dataset))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# 6. Training the model
def train_model(model, train_dataset, epochs, batch_size, output_dir):
    """
    Trains the model with a given dataset and saves the results.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Compile the model with the hybrid loss function
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs
    )

    return history



if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    # Define paths to your training and validation directories
    input_dir = 'MasterNetverk/LSUI/input'  # Path to training images
    gt_dir = 'MasterNetverk/LSUI/GT'      # Path to validation images


    # Define parameters
    batch_size = 8
    epochs =100 
    output_channels = 3  # RGB images
    output_dir = './output'

    # Prepare datasets
    train_dataset = prepare_dataset(input_dir, gt_dir,batch_size)
    #val_dataset = prepare_dataset_from_directory(val_dir, batch_size, shuffle=False)
    
    # Initialize the U-Net model
    model = unet_model_with_pooling(output_channels)

    # Train the model
    history = train_model(model, train_dataset, epochs, batch_size, output_dir)
    model.save("modelpoolingDWPWMSE512.h5")
    model.save("modelpoolingDWPWMSE512.keras")
    print("Final model saved.")
