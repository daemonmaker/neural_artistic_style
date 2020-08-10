import tensorflow as tf

from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.python.keras.models import Model

# Let's take some inspiration from
#     https://github.com/krasserm/super-resolution/blob/master/model/edsr.py
# but cut back because we don't want something too complicated here for now.
def make_style_transfer_model(num_filters=64, num_res_blocks=8, res_block_scaling=None):
    nbr_input_channels = 3
    x_in = Input(shape=(None, None, nbr_input_channels))
    x = b = Conv2D(num_filters, 3, padding='same')(x_in)
    for _ in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)

    # let's try this instead
    b = Conv2D(nbr_input_channels, 3, padding='same')(b)
    x = Add()([x_in, b])

    # old code, which was problematic because there were 64 channels
    # coming out, insead of the 3 channels that we'd expect in an image
    #  b = Conv2D(num_filters, 3, padding='same')(b)
    #  x = Add()([x, b])
    return Model(x_in, x, name="basic_style_resnet")

def res_block(x_in, filters, scaling):
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
       x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x



# What's the format of the images when they enter this pipeline?
