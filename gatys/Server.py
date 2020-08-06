from flask import Flask, url_for, request
from celery import Celery

from utils import *

import tensorflow as tf
from gatys import run_styler

import time
import random

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


@app.route('/')
def get_root():
    #task = add.delay(random.randint(-1000, 1000), random.randint(-1000, 1000))
    task2 = start_processing.delay(3)
    return '<img src="' + url_for('static', filename='dog.jpg') + '" /><br />' \
    'Image task: <b><a href="' + url_for('check_status', task_id=task2.id) + '">' + str(task2.id) + '</a></b><br />'
    #'Add task: <b><a href="' + url_for('check_status', task_id=task.id) + '">' + str(task.id) + '</a></b><br />' \


@celery.task(bind=True)
def start_processing(self, iterations):
    self.update_state(state='STARTED', meta={'type': 'image'})
    time.sleep(1)

    content_path = tf.keras.utils.get_file('dog.jpg', 'file://localhost/tf/images/dog.jpg')
    style_path = tf.keras.utils.get_file('abstract.jpg', 'file://localhost/tf/images/abstract.jpg')

    print(content_path)
    print(style_path)

    self.update_state(state='LOADING_CONTENT', meta={'type': 'image'})
    content_image = load_img(content_path)
    '''
    max_dim = 512
    img = tf.io.read_file(content_path)
    self.update_state(state='READ_FILE', meta={'type': 'image'})
    img = tf.image.decode_image(img, channels=3)
    self.update_state(state='DECODED_IMAGE', meta={'type': 'image'})
    #img = tf.image.convert_image_dtype(img, tf.float32)
    self.update_state(state='CONVERTED_IMAGE_DTYPE', meta={'type': 'image'})

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    self.update_state(state='CAST', meta={'type': 'image'})
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    self.update_state(state='CAST', meta={'type': 'image'})

    img = tf.image.resize(img, new_shape)
    self.update_state(state='RESIZED', meta={'type': 'image'})
    content_image = img[tf.newaxis, :]
    '''

    self.update_state(state='LOADING_STYLE', meta={'type': 'image'})
    style_image = load_img(style_path)

    self.update_state(state='BUILDING_MODEL', meta={'type': 'image'})
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    self.update_state(state='STYLING', meta={'type': 'image'})
    image = run_styler(vgg=vgg, style_image=style_image, content_image=content_image, iterations=iterations, convert_result_to_image=True)

    self.update_state(state='WRITING_IMAGE', meta={'type': 'image'})
    image.save('/tf/gatys/static/result.jpg', 'JPEG')

    self.update_state(state='SUCCESS', meta={'type': 'image'})

    return {'type': 'image', 'value': image}


@celery.task(bind=True)
def add(self, term_a, term_b):
    result = 0
    self.update_state(state='STARTED', meta={'type': 'add', 'term_a': term_a, 'term_b': term_b, 'sum': result})
    result += term_a + term_b
    time.sleep(random.randint(0, 5))
    self.update_state(state='SUCCESS', meta={'type': 'add', 'term_a': term_a, 'term_b': term_b, 'sum': result})
    return {'type': 'add', 'term_a': term_a, 'term_b': term_b, 'sum': result}


@app.route('/status/<task_id>')
def check_status(task_id):
    print('task_id: ', task_id)
    task = add.AsyncResult(task_id)
    result = {
        'status': task.state,
    }
    if task.ready() and task.get('type', None) == 'add':
        result['term_a'] = task.info.get('term_a', None)
        result['term_b'] = task.info.get('term_b', None)
    if task.state == 'SUCCESS':
        result['info'] = task.info
        result = str(result) + '<br /><img src="/static/result.jpg" />'
    return result
