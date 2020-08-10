import tensorflow as tf

import numpy as np
import PIL.Image
import time
import functools

from gatys.Styler import (StyleContentModel, style_content_loss)
from singlenet.model import make_style_transfer_model

# this is not supposed to be needed, and adding this line
# doesn't seem to change anything, but then why am I finding
# myself dealing with some error message that talks about '_in_graph_mode' ?
tf.executing_eagerly()

####
# Temporary measure : I'll just copy those over here and then I'll talk
# to Dustin about organizing paths.
####
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model
####
####
# from utils import clip_0_1, gram_matrix, vgg_layers


def run_styler(
    style_image,
    content_image,
    iterations=3,
    learning_rate=0.02,
    style_weight=1e-2,
    content_weight=1e4,
    total_variation_weight=10,
    content_layers = ['block5_conv2'],
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1'],
):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    
    print('Layers:')
    for layer in vgg.layers:
        print(layer.name)

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)
    extractor = StyleContentModel(style_layers, content_layers)

    # these are computed once, and once only
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    style_transfer_model = make_style_transfer_model()
    opt = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)

    @tf.function()
    def f(content_image):
        # this is the fwdprop segment that contains the parameters to train
        output_image = style_transfer_model(content_image)

        # Not sure what to do with scale of the domain,
        # if we should care about [0,1] or [0, 255],
        # or just forget about it because the style_transfer_model
        # will take care of scaling the values in the
        # correct format expected by the next step.

        # then we plop this into the vgg network
        outputs = extractor(output_image)
        loss = style_content_loss(outputs, style_targets, style_weight, content_weight, num_style_layers, content_targets, num_content_layers)
        loss += total_variation_weight*tf.image.total_variation(output_image)

    # def train_step():
    #     with tf.GradientTape() as tape:

    #         tape.watch(style_transfer_model.trainable_variables)

    #         # this is the fwdprop segment that contains the parameters to train
    #         output_image = style_transfer_model(content_image)

    #         # Not sure what to do with scale of the domain,
    #         # if we should care about [0,1] or [0, 255],
    #         # or just forget about it because the style_transfer_model
    #         # will take care of scaling the values in the
    #         # correct format expected by the next step.

    #         # then we plop this into the vgg network
    #         outputs = extractor(output_image)
    #         loss = style_content_loss(outputs, style_targets, style_weight, content_weight, num_style_layers, content_targets, num_content_layers)
    #         loss += total_variation_weight*tf.image.total_variation(output_image)

    #         # TODO : this is probably something like the parameters of the model
    #         grad = tape.gradient(loss, style_transfer_model.trainable_variables)
    #         opt.apply_gradients([(grad, style_transfer_model.trainable_variables)])

    with tf.GradientTape() as tape:
        loss = f(content_image)
        grads = tape.gradient(loss, style_transfer_model.trainable_variables)
        import pdb; pdb.set_trace()

    # What am I even supposed to do with this whole thing? I no longer know how to Tensorflow!!
    # opt.apply_gradients([(grad, style_transfer_model.trainable_variables)])

    print('')
    print('Styling:')
    for idx in range(iterations):
        print(idx+1, '/', iterations)
        train_step()

    # At this point we're not returning much.
    # Actually, there's something lacking with this current setup,
    # which is that we're using the same content image each time.
    # We're supposed to iterate over many content images,
    # but with the style image fixed.
    #
    # This isn't at all reflected in the current code.
    # The current code is just a kind of stepping step.


if __name__ == "__main__":
    
    content_path = tf.keras.utils.get_file('dog.jpg', 'file://localhost/tf/images/dog.jpg')
    style_path = tf.keras.utils.get_file('abstract.jpg', 'file://localhost/tf/images/abstract.jpg')
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    run_styler(style_image, content_image)
    
    print("yay")