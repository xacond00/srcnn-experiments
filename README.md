# KNN Projekt
RESULTS:
https://drive.google.com/drive/folders/16FCOaLuFvgzC1v0jG7NbsmuZq_6kACBz?hl=cs

## Execution instructions
### Requirements

Pytorch (CUDA)
Torchvision
PIL
ptflops
timm

Download gdrive files and place them in this directory
(`gan_base` for training and `test` for testing) 

### Training

To train auxiliary upscaler run:
`python train_aux.py`
To train custom SrResNet upscaler run:
`python train_srcnn.py`
To train custom SRGAN upscaler run:
`python train_srgan.py`
To train custom RUnet upscaler run:
`python train_runet.py`
To train ResShift upscaler see:
`python rs/README.MD`

### Testing

