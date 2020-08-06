FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN mkdir /scripts && mkdir /logs && apt --yes install procps redis-server vim &&  pip install pillow tensorflow_hub flask celery redis
COPY scripts/*.sh /scripts/
