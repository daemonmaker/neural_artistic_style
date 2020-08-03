# neural_artistic_style
A playground for experimenting with different methods of neural artistic style transfer.

Currently, only the Gatys method (i.e. the original methodology, https://arxiv.org/abs/1508.06576) is supported. There are two implementations. The first being the notebook from the TensorFlow tutorial called Neural Style Transfer (https://www.tensorflow.org/tutorials/generative/style_transfer) which is found in `style_transfer.ipynb`. The other is in the notebook `gatys.ipynb` and is merely a refactorization of this code such that the utility functions are in `utils.py` and the logic specific to the the method is found in the `gatys` python module in this repository.

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
