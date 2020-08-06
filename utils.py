import numpy as np
import PIL.Image
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False


def tensor_to_image(tensor):
    """Convert a tensor to an image.

    Converts a TensorFlow Tensor into a PIL Image.

    Args:
        tensor: Tensor representing a single image.

    Returns:
        A PIL.Image object.
    """
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    """Loads an image.

    Loads an image from a file into a TensorFlow Tensor. It also converts pixel values to values in the range [0, 1] and
    resizes the image such that the longer side is a maximum of 512 pixels for compatibility with a VGG16 deep learning
    model.

    Args:
        path_to_img: A path to an image.

    Returns:
        A tf.Tensor object representing the image in the file specified.
    """
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


def imshow(image, title=None):
    """Plots an image.

    Plots an image using matplotlib.pyplot imported as `plt`.

    Args:
        image: An image to plot.
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


def clip_0_1(image):
    """Clips the values of an image.

    Clips the values of an image to the range [0, 1]

    Args:
        image: An image with values potentially outside the range [0, 1].

    Returns:
        An image wherein the pixel values are in the range [0, 1].
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def gram_matrix(input_tensor):
    """Computes the Grammian of a tensor.

    Computes the Gram Matrix/Tensor of the tensor specified.

    Args:
        input_tensor: A tf.Tensor.

    Returns:
        A tf.Tensor representing the Grammian of the supplied tf.Tensor.
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


def vgg_layers(layer_names, vgg=None):
    """Returns a list layers from a VGG model.

    Creates a VGG model that returns a list of intermediate output values.

    Args:
        layer_names: The names of the layers for which the intermediate values are desired.
        vgg: An optional VGG model. If not specified then a VGG19 is loaded without the head and with the ImageNet
        weights.

    Returns:
       A list of tf.Tensor objects representing the intermediate values of a VGG model.
    """
    # Load our model. Load pretrained VGG, trained on imagenet data
    if vgg is None:
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        print('vgg created')
        vgg.trainable = False
        print('vgg marked not trainable')

    outputs = [vgg.get_layer(name).output for name in layer_names]
    print('outputs: ', outputs)

    model = tf.keras.Model([vgg.input], outputs)
    print('model built')
    return model
