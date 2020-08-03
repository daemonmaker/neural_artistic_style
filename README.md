# neural_artistic_style
A playground for experimenting with different methods of neural artistic style transfer.

# some docker commands
## without GPU

docker build -t neural_artistic_style -f nas.Dockerfile .

docker build -t neural_artistic_style_gpu -f nasgpu.Dockerfile .