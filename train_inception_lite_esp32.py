import sys
import os
import argparse
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    MaxPooling2D,
    SeparableConv2D,
    GlobalAveragePooling2D,
    Dropout,
    ReLU,
    BatchNormalization,
    Reshape,
    Cropping1D,
    concatenate,
    Add,
)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau


# =============================================================================
# 1. PARAMETER SETUP (UNCHANGED)
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--data-directory", type=str, default=".")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--learning-rate", type=float, default=0.001)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--ensure-determinism", action="store_true")

args, unknown = parser.parse_known_args()

EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
ENSURE_DETERMINISM = args.ensure_determinism

N_FREQ = 41
n_time = int(input_length / N_FREQ)
expected_len = n_time * N_FREQ
cropping_amount = input_length - expected_len


# =============================================================================
# 2. DATA PIPELINE (UNCHANGED)
# =============================================================================

def crop_for_aug(data, label):
    return data[:expected_len], label


def pad_after_aug(data, label):
    paddings = [[0, cropping_amount]]
    data = tf.pad(data, paddings, "CONSTANT")
    return data, label


train_dataset = train_dataset.map(crop_for_aug)

if cropping_amount > 0:
    train_dataset = train_dataset.map(pad_after_aug)

if not ENSURE_DETERMINISM:
    train_dataset = train_dataset.shuffle(BATCH_SIZE * 4)

train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)


# =============================================================================
# 3. OPTIMIZED MODULES (TINYML)
# =============================================================================

def conv_block(x, filters, kernel_size, strides=(1, 1)):
    """
    Standard convolution block for the stem layer.
    """
    x = Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6.0)(x)  # ReLU6 for int8-friendly quantization
    return x


def sep_block(
    x,
    filters,
    kernel_size,
    strides=(1, 1),
    dilation_rate=(1, 1),
):
    """
    Separable convolution block.
    Much lighter than standard Conv2D (~9x fewer MACs).
    """
    x = SeparableConv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        dilation_rate=dilation_rate,
        use_bias=False,
        depth_multiplier=1,
    )(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6.0)(x)
    return x


def inception_lite_module(x, filters_base):
    """
    Ultra-lightweight Inception-style module.

    - Reduced number of branches
    - Heavy use of separable convolutions
    - Mandatory bottleneck after concatenation
    """

    # Branch 1: 1x1 Conv (channel mixing)
    b1 = Conv2D(
        filters_base,
        (1, 1),
        padding="same",
        use_bias=False,
    )(x)
    b1 = BatchNormalization()(b1)
    b1 = ReLU(max_value=6.0)(b1)

    # Branch 2: 3x3 Separable Conv (local spatial features)
    b2 = sep_block(x, filters_base, (3, 3))

    # Branch 3: Dilated Separable Conv (longer temporal context)
    # Dilation (2, 1): stretch time axis, keep frequency axis
    b3 = sep_block(
        x,
        filters_base,
        (3, 3),
        dilation_rate=(2, 1),
    )

    # Concatenate branches
    x = concatenate([b1, b2, b3], axis=-1)

    # Bottleneck: reduce channel explosion immediately
    x = Conv2D(
        filters_base,
        (1, 1),
        padding="same",
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6.0)(x)

    return x


# =============================================================================
# 4. MODEL ARCHITECTURE
# =============================================================================

def build_model():
    input_tensor = Input(shape=(input_length,), name="x_input")
    x = input_tensor

    if cropping_amount > 0:
        x = Reshape((input_length, 1))(x)
        x = Cropping1D(cropping=(0, cropping_amount))(x)
        x = Reshape((expected_len,))(x)

    x = Reshape((n_time, N_FREQ, 1))(x)

    # ---------------------------------------------------------------------
    # STEM
    # Downsample early to reduce compute cost
    # ---------------------------------------------------------------------
    x = conv_block(x, 16, (3, 3), strides=(2, 2))

    # ---------------------------------------------------------------------
    # INCEPTION LITE BLOCKS
    # ---------------------------------------------------------------------

    # Block 1
    x = inception_lite_module(x, filters_base=32)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Dropout(0.2)(x)

    # Block 2
    x = inception_lite_module(x, filters_base=48)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Dropout(0.2)(x)

    # Block 3
    # SeparableConv is cheap, so we add depth for accuracy
    x = inception_lite_module(x, filters_base=64)

    # ---------------------------------------------------------------------
    # HEAD
    # ---------------------------------------------------------------------
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)

    output = Dense(
        classes,
        activation="softmax",
        kernel_regularizer=l2(1e-4),
        name="y_pred",
    )(x)

    return Model(input_tensor, output, name="Inception_Lite_ESP32")


model = build_model()


# =============================================================================
# 5. TRAINING
# =============================================================================

optimizer = Adam(learning_rate=LEARNING_RATE)

# More aggressive LR reduction for faster convergence
callbacks.append(
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )
)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    verbose=2,
    callbacks=callbacks,
)

disable_per_channel_quantization = False
