FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN pip install pillow tensorflow_hub flask
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

