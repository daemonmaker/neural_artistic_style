import tensorflow as tf

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

from utils import clip_0_1, gram_matrix, vgg_layers

class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
    
    return {'content':content_dict, 'style':style_dict}

def style_content_loss(outputs, style_targets, style_weight, content_weight, num_style_layers, content_targets, num_content_layers):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(image, style_targets, style_weight, content_weight, total_variation_weight, num_style_layers, content_targets, num_content_layers, extractor, opt):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs, style_targets, style_weight, content_weight, num_style_layers, content_targets, num_content_layers)
    loss += total_variation_weight*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))


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

    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image*255)

    extractor = StyleContentModel(style_layers, content_layers)

    results = extractor(tf.constant(content_image))

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)
    #image = tf.Variable(style_image)

    opt = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)
 
    print('')
    print('Styling:')
    for idx in range(iterations):
        print(idx+1, '/', iterations)
        train_step(image, style_targets, style_weight, content_weight, total_variation_weight, num_style_layers, content_targets, num_content_layers, extractor, opt)

    return image
