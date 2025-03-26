import tensorflow as tf
from tensorflow.keras import layers, Model

def create_simple_cnn(input_shape=(16, 16, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Simple CNN with one convolutional layer
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flattening before fully connected layer
    x = layers.GlobalMaxPooling2D()(x)
    
    # Single neuron output for binary classification
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

# Create and compile the model
model = create_simple_cnn()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.save("MasterVitis-AI/Vitis-AI/modelSimpleCNN.h5")
model.save_weights("MasterVitis-AI/Vitis-AI/cnn_weights.h5")