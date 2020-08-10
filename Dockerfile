FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN pip install pillow tensorflow_hub flask

# so we can refer to different modules inside by name
ENV PYTHONPATH "${PYTHONPATH}:/tf"