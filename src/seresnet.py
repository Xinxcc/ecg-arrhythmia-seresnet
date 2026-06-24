"""A compact 2-D Squeeze-and-Excitation ResNet for log-Mel ECG features."""
from __future__ import annotations

from tensorflow.keras import layers, Model


def se_block(x, reduction: int = 8):
    """Channel attention: squeeze with global average pooling, excite with a
    small bottleneck MLP, then re-scale the input channels."""
    channels = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(channels // reduction, 4), activation="relu")(s)
    s = layers.Dense(channels, activation="sigmoid")(s)
    s = layers.Reshape((1, 1, channels))(s)
    return layers.Multiply()([x, s])


def residual_se_block(x, filters: int, stride: int = 1):
    """Pre-activation residual block with an SE gate on the residual path."""
    shortcut = x
    y = layers.BatchNormalization()(x)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 3, strides=stride, padding="same",
                      kernel_initializer="he_normal")(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, 3, strides=1, padding="same",
                      kernel_initializer="he_normal")(y)
    y = se_block(y)

    # match shapes on the shortcut when channels or resolution change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same",
                                 kernel_initializer="he_normal")(shortcut)
    return layers.Add()([shortcut, y])


def build_se_resnet(input_shape, n_classes: int,
                    stages=(2, 2, 2, 2), base_filters: int = 32,
                    dropout: float = 0.3, name: str = "se_resnet") -> Model:
    """SE-ResNet: stem conv -> 4 residual stages (downsampling each) ->
    global average pooling -> dropout -> softmax."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(base_filters, 5, strides=2, padding="same",
                      kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    filters = base_filters
    for stage, n_blocks in enumerate(stages):
        for block in range(n_blocks):
            stride = 2 if (block == 0 and stage > 0) else 1
            x = residual_se_block(x, filters, stride=stride)
        filters *= 2

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D(name="embedding")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inputs, outputs, name=name)


def replace_head(backbone: Model, n_classes: int, dropout: float = 0.3) -> Model:
    """Reuse a trained backbone up to the `embedding` layer and attach a fresh
    classification head (used for the binary -> 4-class transfer step)."""
    feat = backbone.get_layer("embedding").output
    x = layers.Dropout(dropout)(feat)
    outputs = layers.Dense(n_classes, activation="softmax", name="head_4class")(x)
    return Model(backbone.input, outputs, name="se_resnet_finetune")
