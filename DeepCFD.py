# -*- coding: utf-8 -*-

"""
Created on Tue Oct 10 03:39:17 2023

@author: Mehras Amirhakimi
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, BatchNormalization, MaxPooling2D, Concatenate, UpSampling2D, Input
from tensorflow.keras.models import Model


# Helper function for creating layers
def create_layer(in_channels, out_channels, kernel_size, use_weight_norm=True, use_batch_norm=True,
                 activation=tf.keras.layers.ReLU, convolution=tf.keras.layers.Conv2D):
    assert kernel_size % 2 == 1, "Kernel size should be odd for consistent padding"
    layers = []
    
    # Convolution layer with padding
    conv_layer = convolution(filters=out_channels, kernel_size=kernel_size, padding='same')
    if use_weight_norm:
        conv_layer = tf.keras.layers.experimental.SyncBatchNormalization()(conv_layer)
    layers.append(conv_layer)
    
    # Activation
    if activation is not None:
        layers.append(activation())
    
    # Batch Normalization
    if use_batch_norm:
        layers.append(tf.keras.layers.BatchNormalization())
        
    return tf.keras.Sequential(layers)

# Autoencoder model in
class AutoEncoder(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64],
                 use_weight_norm=True, use_batch_norm=True, activation=tf.keras.layers.ReLU, final_activation=None):
        super(AutoEncoder, self).__init__()
        assert len(filters) > 0, "Filters list must not be empty"
        
        self.encoder = []
        self.decoder = []
        
        # Build encoder
        for i, filter_size in enumerate(filters):
            if i == 0:
                self.encoder.append(create_layer(in_channels, filter_size, kernel_size, use_weight_norm, use_batch_norm, activation, tf.keras.layers.Conv2D))
            else:
                self.encoder.append(create_layer(filters[i-1], filter_size, kernel_size, use_weight_norm, use_batch_norm, activation, tf.keras.layers.Conv2D))
        
        # Build decoder
        for i in range(len(filters) - 1, -1, -1):
            if i == len(filters) - 1:
                self.decoder.append(create_layer(filters[i], out_channels, kernel_size, use_weight_norm, False, final_activation, tf.keras.layers.Conv2DTranspose))
            else:
                self.decoder.append(create_layer(filters[i+1], filters[i], kernel_size, use_weight_norm, use_batch_norm, activation, tf.keras.layers.Conv2DTranspose))
        
        self.encoder = tf.keras.Sequential(self.encoder)
        self.decoder = tf.keras.Sequential(self.decoder)

    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)

# AutoEncoderEx model
class AutoEncoderEx(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[8, 16, 16, 32],
                 use_weight_norm=True, use_batch_norm=True, activation=tf.keras.layers.ReLU, final_activation=None):
        super(AutoEncoderEx, self).__init__()
        assert len(filters) > 0, "Filters list must not be empty"
        
        self.encoder = []
        self.decoders = [[] for _ in range(out_channels)]
        
        # Build encoder
        for i, filter_size in enumerate(filters):
            if i == 0:
                encoder_layer = create_layer(in_channels, filter_size, kernel_size, use_weight_norm, use_batch_norm, activation, tf.keras.layers.Conv2D)
            else:
                encoder_layer = create_layer(filters[i-1], filter_size, kernel_size, use_weight_norm, use_batch_norm, activation, tf.keras.layers.Conv2D)
            self.encoder.append(encoder_layer)
            
            # Build decoder layers
            for c in range(out_channels):
                if i == 0:
                    decoder_layer = create_layer(filter_size, 1, kernel_size, use_weight_norm, False, final_activation, tf.keras.layers.Conv2DTranspose)
                else:
                    decoder_layer = create_layer(filters[i], filters[i-1], kernel_size, use_weight_norm, use_batch_norm, activation, tf.keras.layers.Conv2DTranspose)
                self.decoders[c].insert(0, decoder_layer)
        
        self.encoder = tf.keras.Sequential(self.encoder)
        self.decoders = [tf.keras.Sequential(dec) for dec in self.decoders]

    def call(self, inputs):
        x = self.encoder(inputs)
        y = [decoder(x) for decoder in self.decoders]
        return tf.concat(y, axis=-1)

# Example of creating a model and checking its summary
model = AutoEncoder(in_channels=1, out_channels=1)
model.build(input_shape=(None, 50, 50, 1))
model.summary()

# AutoEncoderEx model summary
auto_encoder_ex = AutoEncoderEx(in_channels=1, out_channels=2)
auto_encoder_ex.build(input_shape=(None, 50, 50, 1))
auto_encoder_ex.summary()


import tensorflow as tf

# Helper functions for building encoder and decoder blocks
def create_encoder_block(in_channels, out_channels, kernel_size, use_weight_norm=True, use_batch_norm=True,
                         activation=tf.keras.layers.ReLU, layers=2):
    encoder = []
    for i in range(layers):
        _in = out_channels if i > 0 else in_channels
        _out = out_channels
        encoder.append(create_layer(_in, _out, kernel_size, use_weight_norm, use_batch_norm, activation, tf.keras.layers.Conv2D))
    return tf.keras.Sequential(encoder)

def create_decoder_block(in_channels, out_channels, kernel_size, use_weight_norm=True, use_batch_norm=True,
                         activation=tf.keras.layers.ReLU, layers=2, final_layer=False):
    decoder = []
    for i in range(layers):
        _in = in_channels * 2 if i == 0 else in_channels
        _out = out_channels if i == layers - 1 else in_channels
        _bn = False if (i == layers - 1 and final_layer) else use_batch_norm
        _activation = None if (i == layers - 1 and final_layer) else activation
        decoder.append(create_layer(_in, _out, kernel_size, use_weight_norm, _bn, _activation, tf.keras.layers.Conv2DTranspose))
    return tf.keras.Sequential(decoder)

def create_encoder(in_channels, filters, kernel_size, use_weight_norm=True, use_batch_norm=True,
                   activation=tf.keras.layers.ReLU, layers=2):
    encoder = []
    for i, out_channels in enumerate(filters):
        encoder_block = create_encoder_block(in_channels if i == 0 else filters[i-1], out_channels, kernel_size,
                                             use_weight_norm, use_batch_norm, activation, layers)
        encoder.append(encoder_block)
    return tf.keras.Sequential(encoder)

def create_decoder(out_channels, filters, kernel_size, use_weight_norm=True, use_batch_norm=True,
                   activation=tf.keras.layers.ReLU, layers=2):
    decoder = []
    for i in range(len(filters)):
        decoder_block = create_decoder_block(filters[i] if i == 0 else filters[i-1], 
                                             out_channels if i == 0 else filters[i-1], kernel_size,
                                             use_weight_norm, use_batch_norm, activation, layers,
                                             final_layer=(i == 0))
        decoder.insert(0, decoder_block)
    return tf.keras.Sequential(decoder)

# UNet model
class UNet(tf.keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64], layers=2,
                 use_weight_norm=True, use_batch_norm=True, activation=tf.keras.layers.ReLU, final_activation=None):
        super(UNet, self).__init__()
        assert len(filters) > 0, "Filters list must not be empty"
        
        self.final_activation = final_activation
        self.encoder = create_encoder(in_channels, filters, kernel_size, use_weight_norm, use_batch_norm, activation, layers)
        self.decoder = create_decoder(out_channels, filters, kernel_size, use_weight_norm, use_batch_norm, activation, layers)

    def encode(self, x):
        tensors, indices, sizes = [], [], []
        for encoder_layer in self.encoder.layers:
            x = encoder_layer(x)
            sizes.append(tf.shape(x))
            tensors.append(x)
            x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding="SAME")
            indices.append(x)
        return x, tensors, indices, sizes

    def decode(self, x, tensors, indices, sizes):
        for decoder_layer in self.decoder.layers:
            tensor = tensors.pop()
            size = sizes.pop()
            x = tf.image.resize(x, size[1:3])
            x = tf.concat([tensor, x], axis=-1)
            x = decoder_layer(x)
        return x

    def call(self, inputs):
        x, tensors, indices, sizes = self.encode(inputs)
        x = self.decode(x, tensors, indices, sizes)
        if self.final_activation:
            x = self.final_activation(x)
        return x

# Example usage of the UNet model and summary
unet = UNet(in_channels=4, out_channels=4)
unet.build(input_shape=(None, 128, 128, 4))  # Modify input shape as per the requirements
unet.summary()


def create_layer(in_channels, out_channels, kernel_size, use_bn=True, activation=ReLU, transpose=False):
    layers = []
    if transpose:
        conv_layer = Conv2DTranspose(out_channels, kernel_size, strides=2, padding='same')
    else:
        conv_layer = Conv2D(out_channels, kernel_size, padding='same')
    
    layers.append(conv_layer)
    if use_bn:
        layers.append(BatchNormalization())
    if activation is not None:
        layers.append(activation())
    return layers

def create_encoder_block(in_channels, out_channels, kernel_size, use_bn=True, activation=ReLU, layers=2):
    block = []
    for i in range(layers):
        block.extend(create_layer(in_channels if i == 0 else out_channels, out_channels, kernel_size, use_bn, activation))
    return tf.keras.Sequential(block)

def create_decoder_block(in_channels, out_channels, kernel_size, use_bn=True, activation=ReLU, layers=2, final_layer=False):
    block = []
    for i in range(layers):
        _in_channels = in_channels * 2 if i == 0 else in_channels
        _out_channels = out_channels if i == layers - 1 else in_channels
        _activation = None if final_layer and i == layers - 1 else activation
        block.extend(create_layer(_in_channels, _out_channels, kernel_size, use_bn, _activation, transpose=i == 0))
    return tf.keras.Sequential(block)

def create_encoder(in_channels, filters, kernel_size, use_bn=True, activation=ReLU, layers=2):
    encoder = []
    for i, f in enumerate(filters):
        encoder.append(create_encoder_block(in_channels if i == 0 else filters[i - 1], f, kernel_size, use_bn, activation, layers))
    return encoder

def create_decoder(out_channels, filters, kernel_size, use_bn=True, activation=ReLU, layers=2):
    decoder = []
    for i, f in enumerate(reversed(filters)):
        final_layer = (i == 0)
        out_ch = out_channels if final_layer else filters[len(filters) - i - 2]
        decoder.append(create_decoder_block(f, out_ch, kernel_size, use_bn, activation, layers, final_layer=final_layer))
    return decoder

class UNetEx(Model):
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64], layers=2, use_bn=True, activation=ReLU, final_activation=None):
        super().__init__()
        self.final_activation = final_activation
        self.encoder_blocks = create_encoder(in_channels, filters, kernel_size, use_bn, activation, layers)
        self.decoder_blocks = create_decoder(out_channels, filters, kernel_size, use_bn, activation, layers)
        self.pool = MaxPooling2D(pool_size=(2, 2))

    def call(self, inputs):
        tensors, sizes = [], []
        x = inputs
        # Encoder Path
        for enc_block in self.encoder_blocks:
            x = enc_block(x)
            sizes.append(tf.shape(x))
            tensors.append(x)
            x = self.pool(x)

        # Decoder Path
        for dec_block in self.decoder_blocks:
            size = sizes.pop()
            tensor = tensors.pop()
            x = UpSampling2D(size=(2, 2))(x)
            x = Concatenate()([tensor, x])
            x = dec_block(x)

        if self.final_activation:
            x = self.final_activation(x)
        return x

# Instantiate and build the model
input_tensor = Input(shape=(128, 128, 4))  # Example input shape
unet_model = UNetEx(in_channels=4, out_channels=4)
output_tensor = unet_model(input_tensor)
model = Model(inputs=input_tensor, outputs=output_tensor)
model.summary()

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU, LeakyReLU, BatchNormalization, MaxPooling2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model


# Helper function to create individual layers
def create_layer(in_channels, out_channels, kernel_size, use_bn=True, activation=LeakyReLU, conv_layer=Conv2D, transpose=False):
    layers = []
    conv = Conv2DTranspose if transpose else Conv2D
    conv_layer = conv(out_channels, kernel_size, padding='same')
    layers.append(conv_layer)
    if use_bn:
        layers.append(BatchNormalization())
    if activation is not None:
        layers.append(activation())
    return layers

# Encoder block
def create_encoder_block(in_channels, out_channels, kernel_size, use_bn=True, activation=LeakyReLU, layers=2):
    block = []
    for i in range(layers):
        _in_channels = in_channels if i == 0 else out_channels
        block.extend(create_layer(_in_channels, out_channels, kernel_size, use_bn, activation))
    return tf.keras.Sequential(block)

# Decoder block
def create_decoder_block(in_channels, out_channels, kernel_size, use_bn=True, activation=LeakyReLU, layers=2, final_layer=False):
    block = []
    for i in range(layers):
        _in_channels = in_channels * 2 if i == 0 else in_channels
        _out_channels = out_channels if i == layers - 1 else in_channels
        _activation = None if final_layer and i == layers - 1 else activation
        block.extend(create_layer(_in_channels, _out_channels, kernel_size, use_bn, _activation, transpose=(i == 0)))
    return tf.keras.Sequential(block)

# Create encoder
def create_encoder(in_channels, filters, kernel_size, use_bn=True, activation=LeakyReLU, layers=2):
    encoder = []
    for i, f in enumerate(filters):
        encoder_block = create_encoder_block(in_channels if i == 0 else filters[i - 1], f, kernel_size, use_bn, activation, layers)
        encoder.append(encoder_block)
    return encoder

# Create decoder
def create_decoder(out_channels, filters, kernel_size, use_bn=True, activation=LeakyReLU, layers=2):
    decoder = []
    for i, f in enumerate(reversed(filters)):
        final_layer = (i == 0)
        out_ch = out_channels if final_layer else filters[len(filters) - i - 2]
        decoder_block = create_decoder_block(f, out_ch, kernel_size, use_bn, activation, layers, final_layer=final_layer)
        decoder.insert(0, decoder_block)
    return decoder

# UNetExMod model
class UNetExMod(Model):
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64], layers=3, use_bn=True, activation=LeakyReLU, final_activation=None):
        super(UNetExMod, self).__init__()
        self.final_activation = final_activation
        self.encoder = create_encoder(in_channels, filters, kernel_size, use_bn, activation, layers)
        self.decoders = [create_decoder(1, filters, kernel_size, use_bn, activation, layers) for _ in range(out_channels)]
        self.pool = MaxPooling2D(pool_size=(2, 2))
        self.upsample = UpSampling2D(size=(2, 2))

    def encode(self, x):
        tensors, sizes = [], []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            sizes.append(tf.shape(x))
            tensors.append(x)
            x = self.pool(x)
        return x, tensors, sizes

    def decode(self, x, tensors, sizes):
        outputs = []
        for decoder in self.decoders:
            _x = x
            _tensors = tensors[:]
            _sizes = sizes[:]
            for decoder_block in decoder:
                tensor = _tensors.pop()
                size = _sizes.pop()
                _x = self.upsample(_x)
                _x = Concatenate()([tensor, _x])
                _x = decoder_block(_x)
            outputs.append(_x)
        return tf.concat(outputs, axis=-1)

    def call(self, inputs):
        x, tensors, sizes = self.encode(inputs)
        x = self.decode(x, tensors, sizes)
        if self.final_activation:
            x = self.final_activation(x)
        return x

# Instantiate and build the model
input_tensor = Input(shape=(128, 128, 4))  # Example input shape
unet_model = UNetExMod(in_channels=4, out_channels=4)
output_tensor = unet_model(input_tensor)
model = Model(inputs=input_tensor, outputs=output_tensor)
model.summary()


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

# Functions for data splitting and initialization
def split_tensors(*tensors, ratio):
    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        split1.append(tensor[:int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2

def initialize(model):
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            tf.keras.initializers.GlorotNormal()(layer.weights[0])
            if layer.bias is not None:
                tf.keras.initializers.RandomNormal(stddev=0.02)(layer.bias)


##functions

import numpy as np
from matplotlib import pyplot as plt

def split_tensors(*tensors, ratio):
    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        split1.append(tensor[:int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2

def initialize(model, gain=1, std=0.02):
    for module in model.modules():
        if type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            nn.init.xavier_normal_(module.weight, gain)
            if module.bias is not None:
                nn.init.normal_(module.bias, 0, std)

def visualize(sample_y, out_y, error, s):

    minu = np.min(sample_y[s, 0, :, :])
    maxu = np.max(sample_y[s, 0, :, :])

    minv = np.min(sample_y[s, 1, :, :])
    maxv = np.max(sample_y[s, 1, :, :])

    minp = np.min(sample_y[s, 2, :, :])
    maxp = np.max(sample_y[s, 2, :, :])

    mineu = np.min(error[s, 0, :, :])
    maxeu = np.max(error[s, 0, :, :])

    minev = np.min(error[s, 1, :, :])
    maxev = np.max(error[s, 1, :, :])

    minep = np.min(error[s, 2, :, :])
    maxep = np.max(error[s, 2, :, :])

    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.subplot(3, 3, 1)
    plt.title('CFD', fontsize=18)
    plt.imshow(np.transpose(sample_y[s, 0, :, :]), cmap='jet', vmin = minu, vmax = maxu, origin='lower', extent=[0,260,0,120])
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Ux', fontsize=18)
    plt.subplot(3, 3, 2)
    plt.title('CNN', fontsize=18)
    plt.imshow(np.transpose(out_y[s, 0, :, :]), cmap='jet', vmin = minu, vmax = maxu, origin='lower', extent=[0,260,0,120])
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 3)
    plt.title('Error', fontsize=18)
    plt.imshow(np.transpose(error[s, 0, :, :]), cmap='jet', vmin = mineu, vmax = maxeu, origin='lower', extent=[0,260,0,120])
    plt.colorbar(orientation='horizontal')

    plt.subplot(3, 3, 4)
    plt.imshow(np.transpose(sample_y[s, 1, :, :]), cmap='jet', vmin = minv, vmax = maxv, origin='lower', extent=[0,260,0,120])
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Uy', fontsize=18)
    plt.subplot(3, 3, 5)
    plt.imshow(np.transpose(out_y[s, 1, :, :]), cmap='jet', vmin = minv, vmax = maxv, origin='lower', extent=[0,260,0,120])
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 6)
    plt.imshow(np.transpose(error[s, 1, :, :]), cmap='jet', vmin = minev, vmax = maxev, origin='lower', extent=[0,260,0,120])
    plt.colorbar(orientation='horizontal')

    plt.subplot(3, 3, 7)
    plt.imshow(np.transpose(sample_y[s, 2, :, :]), cmap='jet', vmin = minp, vmax = maxp, origin='lower', extent=[0,260,0,120])
    plt.colorbar(orientation='horizontal')
    plt.ylabel('p', fontsize=18)
    plt.subplot(3, 3, 8)
    plt.imshow(np.transpose(out_y[s, 2, :, :]), cmap='jet', vmin = minp, vmax = maxp, origin='lower', extent=[0,260,0,120])
    plt.colorbar(orientation='horizontal')
    plt.subplot(3, 3, 9)
    plt.imshow(np.transpose(error[s, 2, :, :]), cmap='jet', vmin = minep, vmax = maxep, origin='lower', extent=[0,260,0,120])
    plt.colorbar(orientation='horizontal')
    plt.tight_layout()
    plt.show()


# Early Stopping implementation using callbacks
class EarlyStoppingCustom(tf.keras.callbacks.Callback):
    def __init__(self, patience=7, verbose=False, delta=0):
        super(EarlyStoppingCustom, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.model.stop_training = True
                if self.verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs")
        else:
            self.best_score = score
            self.counter = 0

# Training and Evaluation functions
def train_model(model, train_dataset, val_dataset, optimizer, loss_func, epochs=100, batch_size=256, patience=10):
    early_stopping = EarlyStoppingCustom(patience=patience, verbose=True)
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )
    return model, history

# Load dataset and split
x = pickle.load(open("/content/drive/MyDrive/DeepCFD/dataX.pkl", "rb"))
y = pickle.load(open("/content/drive/MyDrive/DeepCFD/dataY.pkl", "rb"))
x = np.array(x, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Define model, optimizer, and loss function
def create_model(input_shape):

    pass  # Define architecture based on UNetExMod or equivalent

model = create_model(input_shape=(128, 128, 3))  # example shape
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_func = tf.keras.losses.MeanSquaredError()

# Compile model
model.compile(optimizer=optimizer, loss=loss_func, metrics=["mse"])

# Training
train_data, test_data = split_tensors(x, y, ratio=0.7)
train_dataset = tf.data.Dataset.from_tensor_slices((train_data[0], train_data[1])).batch(256)
val_dataset = tf.data.Dataset.from_tensor_slices((test_data[0], test_data[1])).batch(256)

# Train model
trained_model, history = train_model(
    model, train_dataset, val_dataset,
    optimizer=optimizer, loss_func=loss_func,
    epochs=100, batch_size=256, patience=10
)

# Save the model if needed
model.save("trained_model")

