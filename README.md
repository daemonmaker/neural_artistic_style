# neural_artistic_style
A playground for experimenting with different methods of neural artistic style transfer.

Currently, only the Gatys method (i.e. the original methodology, https://arxiv.org/abs/1508.06576) is supported. There are two implementations. The first being the notebook from the TensorFlow tutorial called Neural Style Transfer (https://www.tensorflow.org/tutorials/generative/style_transfer) which is found in `style_transfer.ipynb`. The other is in the notebook `gatys.ipynb` and is merely a refactorization of this code such that the utility functions are in `utils.py` and the logic specific to the the method is found in the `gatys` python module in this repository.

## Docker image
The simplest way to get started with this repository is to use docker. Execute the following command to build the docker image:

```bash
./scripts/build_image
```

The docker image is named `neural_artistic_style` and it supports two modes; one that exposes Jupyter notebooks and another that exposes a Flask server. These modes can be activated by executing `./scripts/run_jupyter` and `./scripts/run_flask` respectively. These scripts need to be executed from within the root of this repository because they mount this repository at `/tf` within the container.

*Note:* The resulting containers are not retained once they are shutdown.

### GPU support
The `run_<mode>` scripts take a single parameter that specifies the number of GPUs. For example, executing the following command tells the docker container to start a container in Jupyter mode using one GPU:

```bash
./scripts/run_jupyter 1
```
