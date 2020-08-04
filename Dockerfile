FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN pip install pillow tensorflow_hub flask
