# neural_artistic_style
A playground for experimenting with different methods of neural artistic style transfer.

## Docker
The simplest way to get started with this repository is to use docker. Just build the image and create the container by executing the following commands from within the repository:

```bash
./scripts/build_image && ./scripts/run_image
```

Once the container is online it will give you the address for the jupyter notebook.

### GPU support
The `run_image` script takes a single parameter that specifies the number of GPUs. For example, executing the following command tells the docker container to use one GPU:

```bash
./scripts/run_image 1
```
