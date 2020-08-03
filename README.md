# neural_artistic_style
A playground for experimenting with different methods of neural artistic style transfer.

<<<<<<< HEAD
# some docker commands
## without GPU

docker build -t neural_artistic_style -f nas.Dockerfile .

docker build -t neural_artistic_style_gpu -f nasgpu.Dockerfile .
=======
## Docker
The simplest way to get started with this repository is to use docker. Just build the image and create the container by executing the following commands from within the repository:

```bash
./build_image && ./run_image
```

Once the container is online it will give you the address for the jupyter notebook.
>>>>>>> 7eaab23c32e38deb685433f3f9a982e8626c7111
