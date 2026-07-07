import os
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# Define the Keras ConvBlock equivalent
def keras_conv_block(out_channels, pool=False):
    block_layers = [
        layers.Conv2D(out_channels, kernel_size=3, padding='same', use_bias=True),
        layers.BatchNormalization(epsilon=1e-5, momentum=0.9),
        layers.ReLU()
    ]
    if pool:
        block_layers.append(layers.MaxPooling2D(pool_size=4, strides=4, padding='valid'))
    return Sequential(block_layers)

# Define Keras ResNet9 equivalent architecture
class KerasResNet9(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = keras_conv_block(64)
        self.conv2 = keras_conv_block(128, pool=True)
        self.res1_0 = keras_conv_block(128)
        self.res1_1 = keras_conv_block(128)
        
        self.conv3 = keras_conv_block(256, pool=True)
        self.conv4 = keras_conv_block(512, pool=True)
        self.res2_0 = keras_conv_block(512)
        self.res2_1 = keras_conv_block(512)
        
        self.pool = layers.MaxPooling2D(pool_size=4, strides=4, padding='valid')
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(num_classes)
        
    def call(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        
        res = self.res1_0(out)
        res = self.res1_1(res)
        out = layers.add([out, res])
        
        out = self.conv3(out)
        out = self.conv4(out)
        
        res = self.res2_0(out)
        res = self.res2_1(res)
        out = layers.add([out, res])
        
        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out

def convert(pth_path, keras_path):
    print(f"Loading PyTorch state dict from: {pth_path}")
    pytorch_state = torch.load(pth_path, map_location='cpu')
    
    # The PyTorch model classifies 38 diseases
    num_classes = 38
    print(f"Initializing Keras ResNet9 with {num_classes} classes...")
    keras_model = KerasResNet9(num_classes)
    
    # Build Keras model variables by running dummy data
    dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
    _ = keras_model(dummy_input)
    print("Keras model initialized and built successfully.")
    
    # Helper to set weights of Keras ConvBlock from PyTorch ConvBlock
    def set_conv_block_weights(keras_block, pytorch_prefix):
        # 1. Conv2D (layer 0 in Keras block)
        conv_weight = pytorch_state[f"{pytorch_prefix}.0.weight"].numpy()
        conv_bias = pytorch_state[f"{pytorch_prefix}.0.bias"].numpy()
        # Transpose PyTorch [out_channels, in_channels, kh, kw] -> Keras [kh, kw, in_channels, out_channels]
        conv_weight_keras = np.transpose(conv_weight, (2, 3, 1, 0))
        keras_block.layers[0].set_weights([conv_weight_keras, conv_bias])
        
        # 2. BatchNorm2D (layer 1 in Keras block)
        bn_weight = pytorch_state[f"{pytorch_prefix}.1.weight"].numpy() # gamma
        bn_bias = pytorch_state[f"{pytorch_prefix}.1.bias"].numpy() # beta
        bn_mean = pytorch_state[f"{pytorch_prefix}.1.running_mean"].numpy() # moving mean
        bn_var = pytorch_state[f"{pytorch_prefix}.1.running_var"].numpy() # moving variance
        keras_block.layers[1].set_weights([bn_weight, bn_bias, bn_mean, bn_var])

    print("Copying weights layer by layer...")
    # Copy all convolutional blocks
    set_conv_block_weights(keras_model.conv1, "conv1")
    set_conv_block_weights(keras_model.conv2, "conv2")
    
    set_conv_block_weights(keras_model.res1_0, "res1.0")
    set_conv_block_weights(keras_model.res1_1, "res1.1")
    
    set_conv_block_weights(keras_model.conv3, "conv3")
    set_conv_block_weights(keras_model.conv4, "conv4")
    
    set_conv_block_weights(keras_model.res2_0, "res2.0")
    set_conv_block_weights(keras_model.res2_1, "res2.1")
    
    # Copy classifier linear (Dense) layer
    fc_weight = pytorch_state["classifier.2.weight"].numpy()
    fc_bias = pytorch_state["classifier.2.bias"].numpy()
    # Transpose PyTorch [out_features, in_features] -> Keras [in_features, out_features]
    fc_weight_keras = np.transpose(fc_weight, (1, 0))
    keras_model.fc.set_weights([fc_weight_keras, fc_bias])
    
    print(f"Saving converted Keras model to: {keras_path}")
    keras_model.save(keras_path)
    print("Conversion completed successfully!")

if __name__ == '__main__':
    pth_model = 'models/plant_disease_model.pth'
    keras_model_file = 'models/plant_disease_recog_model_pwp.keras'
    
    if not os.path.exists(pth_model):
        print(f"Error: PyTorch model file '{pth_model}' not found.")
    else:
        convert(pth_model, keras_model_file)
